import os
import torch
import random
import faiss
import h5py
import numpy as np
import pickle as pkl

from sklearn.decomposition import PCA
from pytorch_lightning import Trainer
from typing import Optional, Dict, List
from omegaconf import DictConfig, OmegaConf
from transformers import AutoTokenizer
from tqdm import tqdm
from collections import OrderedDict

from nemo.collections.nlp.models.nlp_model import NLPModel
from nemo.collections.nlp.modules.common.lm_utils import get_lm_model
from nemo.collections.nlp.data import EntityLinkingDataset
from nemo.collections.common.losses import MultiSimilarityLoss
from nemo.core.classes.exportable import Exportable
from nemo.core.neural_types import NeuralType
from nemo.core.classes.common import typecheck
from nemo.utils import logging

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

__all__ = ['EntityLinkingModel']

#@experimental
class EntityLinkingModel(NLPModel, Exportable):
    """
    Second stage pretraining of BERT based language model
    for entity linking task.
    """

    @property
    def input_types(self) -> Optional[Dict[str, NeuralType]]:
        return self.model.input_types

    @property 
    def output_types(self) :
        pass

    def __init__(self, cfg: DictConfig, trainer: Trainer = None):
        """Initializes the SAP-BERT model for entity linking."""

        # tokenizer needed before super().__init__() so dataset and loader can process data
        self._setup_tokenizer(cfg.tokenizer)

        super().__init__(cfg=cfg, trainer=trainer)

        self.model = get_lm_model(
            pretrained_model_name=cfg.language_model.pretrained_model_name,
            config_file=cfg.language_model.config_file,
            config_dict=cfg.language_model.config,
            checkpoint_file=cfg.language_model.lm_checkpoint,
        )

        self.model = self.model.to(device)

        # Token to use for the self-alignment loss, typically the first token, [CLS]
        self._idx_conditioned_on = 0
        self.loss = MultiSimilarityLoss()
        self.index = None

        
    @typecheck()
    def forward(self, input_ids, token_type_ids, attention_mask):
        hidden_states = self.model(
            input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask
        )

        # normalize to unit sphere
        logits = torch.nn.functional.normalize(hidden_states[:,self._idx_conditioned_on], p=2, dim=1)
        return logits

    def training_step(self, batch, batch_idx):
        """
        Lightning calls this inside the training loop with the data from the training dataloader
        passed in as `batch`.
        """
        input_ids, token_type_ids, attention_mask, concept_ids = batch
        logits = self.forward(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        train_loss = self.loss(logits=logits, concept_ids=concept_ids)

        # No hard examples found in batch, 
        # shouldn't use this batch to update model weights
        if train_loss == 0:
            train_loss = None
            lr = None

        else:
            lr = self._optimizer.param_groups[0]["lr"]
            self.log("train_loss", train_loss)
            self.log("lr", lr, prog_bar=True)

        return {"loss": train_loss, "lr": lr}

    def validation_step(self, batch, batch_idx):
        """
        Lightning calls this inside the validation loop with the data from the validation dataloader
        passed in as `batch`.
        """
        input_ids, input_type_ids, input_mask, concept_ids = batch
        with torch.no_grad():
            logits = self.forward(input_ids=input_ids, token_type_ids=input_type_ids, attention_mask=input_mask)
            val_loss = self.loss(logits=logits, concept_ids=concept_ids)

        # No hard examples found in batch, 
        # shouldn't use this batch to update model weights
        if val_loss == 0:
            val_loss = None
        else:
            self.log("val_loss", val_loss)
            logging.info(f"val loss: {val_loss}")

        return {"val_loss": val_loss}

    def validation_epoch_end(self, outputs):
        """
        Called at the end of validation to aggregate outputs.

        Args:
            outputs: list of individual outputs of each validation step.
        Returns:
            
        """
        if outputs:
            avg_loss = torch.stack([x[f"val_loss"] for x in outputs]).mean()
            self.log(f"val_loss", avg_loss, prog_bar=True)
            
            return {"val_loss": avg_loss}


    def setup_training_data(self, train_data_config: Optional[DictConfig]):
        return
        if not train_data_config or not train_data_config.file_path:
            logging.info(
                f"Dataloader config or file_path or processed data path for the train dataset is missing, \
                        so no data loader for train is created!"
            )

            self._train_dl = None
            return

        self._train_dl = self._setup_dataloader(cfg=train_data_config)

    def setup_validation_data(self, val_data_config: Optional[DictConfig]):
        return
        if not val_data_config or not val_data_config.file_path:
            logging.info(
                f"Dataloader config or file_path or processed data path for the val dataset is missing, \
                        so no data loader for validation is created!"
            )

            self._validation_dl = None
            return

        self._validation_dl = self._setup_dataloader(cfg=val_data_config)

    def setup_index_data(self, index_data_config: Optional[DictConfig]):
        if not index_data_config or not index_data_config.file_path:
            logging.info(
                f"Dataloader config or file_path or processed data path for the index dataset is missing, \
                        so no data loader for the index is created!"
            )

        self._index_dl = self._setup_dataloader(cfg=index_data_config, is_index_data=True)

    def query_index(self, query: str, top_n: int = 5) -> Dict:
        """
        Query the nearest neighbor index of entities to find the 
        concepts in the index dataset that are most similar to the 
        query.

        Args:
            query (str): entity to look up in the index
            top_n (int): max number of nearest neighbor entites returned

        Returns:
            A dictionary with the concept ids of the index's most similar 
            entities as the keys and a tuple containing the string 
            representation of that concept and its cosine similarity to 
            the query as the values. 
        """
        if self.index is None:
            logging.info("Please use model.load_index(cfg) first, then try again")
            return

        query_emb = self._get_query_embedding(query).detach().cpu().numpy()

        if self.index_cfg.apply_pca:
            query_emb = self.pca.transform(query_emb)

        dist, neighbors = self.index.search(query_emb.astype(np.float32), self.index_cfg.query_num_factor*top_n)
        dist, neighbors = dist[0], neighbors[0]
        unique_ids = OrderedDict()
        neighbor_idx = 0

        # Many of nearest neighbors could map to the same concept id, their idx is their unique identifier
        while len(unique_ids) < top_n and neighbor_idx < len(neighbors):
            concept_id_idx = neighbors[neighbor_idx]
            concept_id = self.idx2id[concept_id_idx]

            # Only want one instance of each unique concept
            if concept_id not in unique_ids:
                concept = self.id2string[concept_id]
                unique_ids[concept_id] = (concept, 1 - dist[neighbor_idx])

            neighbor_idx += 1

        unique_ids = dict(unique_ids)

        return unique_ids

    def load_index(self, cfg: DictConfig):
        """
        Loads faiss index.

        Args:
            cfg: Config file specifying index parameters
        """

        if not os.path.isfile(cfg.index_save_name):
            self.build_index(cfg)

        logging.info("Loading index and associated files")
        self.index_cfg = cfg
        self.index = faiss.read_index(cfg.index_save_name)
        self.idx2id = pkl.load(open(cfg.idx_to_id, "rb"))
        self.id2string = pkl.load(open(cfg.id_to_string, "rb"))

        if cfg.apply_pca:
            self.pca = pkl.load(open(cfg.pca.pca_save_name, "rb"))


    def build_index(self, cfg: DictConfig):
        """
        Builds faiss index from index dataset specified in the config.
            
        Args:
            cfg: Config file specifying index parameters
        """
        if os.path.isfile(cfg.index_save_name):
            logging.info("Index file already exists, try loading the index instead")
            return

        # Get index dataset embeddings 
        if os.path.isfile(cfg.embedding_save_name) and os.path.isfile(cfg.concept_id_save_name):
            logging.info("Loading previously extracted index dataset embeddings")
            embeddings = h5py.File(cfg.embedding_save_name, "r")
            embeddings = embeddings[cfg.dataset.name][:]
            concept_ids = pkl.load(open(cfg.concept_id_save_name, "rb"))

        else:
            logging.info("Encoding index dataset, this may take a while")
            self.setup_index_data(cfg.dataset)
            embeddings, concept_ids = self._get_index_embeddings(cfg)

        # Map each index dataset example to its unique index
        if not os.path.isfile(cfg.idx_to_id):
            logging.info("Mapping concept_ids to their unique indices")
            idx2id = {idx : cui for idx, cui in tqdm(enumerate(concept_ids), total=len(concept_ids))}
            pkl.dump(idx2id, open(cfg.idx_to_id, "wb"))

        # Create pca model to reduce dimensionality of index dataset and decrease memory footprint
        if cfg.apply_pca and not os.path.isfile(cfg.pca_embeddings_save_name):
            if not os.path.isfile(cfg.pca.pca_save_name):
                logging.info("Fitting PCA model for embedding dimensionality reduction")
                pca_train_set = random.sample(list(embeddings), k=int(len(embeddings)*cfg.pca.sample_fraction))
                pca = PCA(n_components=cfg.pca.output_dim)
                pca.fit(pca_train_set)
                pkl.dump(pca, open(cfg.pca.pca_save_name, "wb"))
            
            else:
                pca = pkl.load(open(cfg.pca.pca_save_name, "rb"))

            logging.info("Applying PCA transformation to entire index dataset")
            embeddings = np.array(pca.transform(embeddings), dtype=np.float32)
            emb_file = h5py.File(cfg.pca_embeddings_save_name, "w")
            emb_file.create_dataset(cfg.dataset.name, data=embeddings)
            emb_file.close()

        elif cfg.apply_pca:
            logging.info("Loading reduced dimensionality embeddings")
            embeddings = h5py.File(cfg.pca_embeddings_save_name, "r")
            embeddings = embeddings[cfg.dataset.name][:] 


        # Build faiss index from embeddings
        logging.info(f"Training index with embedding dim size {cfg.dims} using {faiss.get_num_gpus()} gpus")
        quantizer = faiss.IndexFlatL2(cfg.dims)
        index = faiss.IndexIVFFlat(quantizer, cfg.dims, cfg.nlist)
        index = faiss.index_cpu_to_all_gpus(index)
        index.train(embeddings)

        logging.info("Adding dataset embeddings to index")
        for i in tqdm(range(0, embeddings.shape[0], cfg.index_batch_size)):
            index.add(embeddings[i:i+cfg.index_batch_size])

        logging.info("Saving index")
        faiss.write_index(faiss.index_gpu_to_cpu(index), cfg.index_save_name)

        logging.info("Index built and saved")


    def _setup_tokenizer(self, cfg: DictConfig):
        tokenizer = AutoTokenizer.from_pretrained(
            cfg.tokenizer_name,
            vocab_file=cfg.vocab_file,
            do_lower_case=cfg.do_lower_case)

        self.tokenizer = tokenizer

    def _setup_dataloader(self, cfg: Dict, is_index_data: bool = False) -> 'torch.utils.data.DataLoader':
        
        dataset = EntityLinkingDataset( 
                tokenizer=self.tokenizer,
                data_file=cfg.data_file,
                max_seq_length=cfg.max_seq_length,
                is_index_data=is_index_data,
            )
    
        return torch.utils.data.DataLoader(
                dataset=dataset,
                batch_size=cfg.batch_size,
                collate_fn=dataset.collate_fn,
                shuffle=cfg.get("shuffle", True),
                num_workers=cfg.get("num_wokers", 2),
                pin_memory=cfg.get("pin_memory", False),
                drop_last=cfg.get("drop_last", False)
                )

    def _get_index_embeddings(self, cfg: DictConfig):
        embeddings = []
        concept_ids = []
                        
        with torch.no_grad():
            for batch in tqdm(self._index_dl):
                input_ids, token_type_ids, input_mask, batch_concept_ids = batch
                input_ids = input_ids.to(device)
                token_type_ids = token_type_ids.to(device)
                input_mask = input_mask.to(device)
                batch_embeddings = self.forward(input_ids=input_ids,
                                      token_type_ids=token_type_ids,
                                      attention_mask=input_mask)

                embeddings.extend(batch_embeddings.detach().cpu().numpy())
                concept_ids.extend(batch_concept_ids.numpy())

        emb_file = h5py.File(cfg.embedding_save_name, "w")
        emb_file.create_dataset(cfg.dataset.name, data=embeddings)
        emb_file.close()

        pkl.dump(concept_ids, open(cfg.concept_id_save_name, "wb"))

        return embeddings, concept_ids 

    def _get_query_embedding(self, query):
        model_input = self.tokenizer(query,
                                    add_special_tokens = True,
                                    padding = True,
                                    truncation = True,
                                    max_length = 512,
                                    return_token_type_ids = True,
                                    return_attention_mask = True,
                                    )

        query_emb = self.forward(input_ids=torch.LongTensor([model_input["input_ids"]]).to(device),
                                 token_type_ids=torch.LongTensor([model_input["token_type_ids"]]).to(device),
                                 attention_mask=torch.LongTensor([model_input["attention_mask"]]).to(device))

        return query_emb


    @classmethod
    def list_available_models(cls) -> Optional[Dict[str, str]]:
        pass

    @classmethod
    def from_pretrained(cls, name: str):
        pass

