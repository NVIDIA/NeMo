import torch
import pickle as pkl
from torch.utils.data import DistributedSampler

from pytorch_lightning import Trainer
from typing import Optional, Dict, List
from omegaconf import DictConfig, OmegaConf
from transformers import AutoTokenizer
from self_alignment_pretraining_dataset_no_pretokenization import SelfAlignmentPretrainingDataset
from multi_similarity_loss import MultiSimilarityLoss
from annoy import AnnoyIndex
from eval_classes import *
from tqdm import tqdm

from nemo.collections.nlp.models.nlp_model import NLPModel
from nemo.collections.nlp.modules.common.lm_utils import get_lm_model
#from nemo.collections.nlp.modules.common.tokenizer_utils import get_tokenizer
from nemo.core.classes.exportable import Exportable
from nemo.core.neural_types import NeuralType
from nemo.core.classes.common import typecheck
from nemo.utils import logging
from megatron.initialize import initialize_megatron

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def check_memory():
    logging.info('GPU memory: %.1f' % (torch.cuda.memory_allocated() // 1024 ** 2))


class SelfAlignmentPretrainingModel(NLPModel, Exportable):
    @property
    def input_types(self) -> Optional[Dict[str, NeuralType]]:
        return self.bert_model.input_types

    def __init__(self, cfg: DictConfig, trainer: Trainer = None):
        """Initializes the SAP-BERT model."""

        # tokenizer needed before super().__init__() so dataset and loader can process data
        self.dataset_cfg = cfg.dataset
        self._setup_tokenizer(cfg.tokenizer)
        #self.tokenizer = AutoTokenizer.from_pretrained(cfg.dataset.tokenizer_name)
        self.config = cfg

        super().__init__(cfg=cfg, trainer=trainer)

        print("intializing model...")
        self.bert_model = get_lm_model(
            pretrained_model_name=cfg.language_model.pretrained_model_name,
            config_file=cfg.language_model.config_file,
            config_dict=cfg.language_model.config,
            checkpoint_file=cfg.language_model.lm_checkpoint,
        )
        print("model initialized")
        self.bert_model = self.bert_model.to(device)

        # Token to use for the self-alignment loss, typically the first token, [CLS]
        self._idx_conditioned_on = 0
        self.loss = MultiSimilarityLoss()

    def _setup_tokenizer(self, cfg: DictConfig):
        tokenizer = AutoTokenizer.from_pretrained(
            cfg.tokenizer_name,
            vocab_file=cfg.vocab_file,
            do_lower_case=cfg.do_lower_case)

        #tokenizer = get_tokenizer(
        #    tokenizer_name=cfg.tokenizer_name,
        #    vocab_file=self.register_artifact(config_path='tokenizer.vocab_file', src=cfg.vocab_file),
        #    special_tokens=OmegaConf.to_container(cfg.special_tokens) if cfg.special_tokens else None,
        #    tokenizer_model=self.register_artifact(config_path='tokenizer.tokenizer_model', src=cfg.tokenizer_model),
        #)

        self.tokenizer = tokenizer
        
    @typecheck()
    def forward(self, input_ids, token_type_ids, attention_mask):
        hidden_states = self.bert_model(
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
        input_ids, token_type_ids, attention_mask, labels = batch
        logits = self.forward(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        train_loss = self.loss(logits=logits, labels=labels)

        if train_loss == 0:
            return {"loss": train_loss, "lr": None}

        lr = self._optimizer.param_groups[0]["lr"]

        self.log("train_loss", train_loss)
        self.log("lr", lr, prog_bar=True)

        return {"loss": train_loss, "lr": lr}

    def validation_step(self, batch, batch_idx):
        """
        Lightning calls this inside the validation loop with the data from the validation dataloader
        passed in as `batch`.
        """
        input_ids, input_type_ids, input_mask, labels = batch
        with torch.no_grad():
            logits = self.forward(input_ids=input_ids, token_type_ids=input_type_ids, attention_mask=input_mask)
            val_loss = self.loss(logits=logits, labels=labels)

        if val_loss == 0:
            return None

        self.log("val_loss", val_loss)
        logging.info(f"val loss: {val_loss}")

        return {"val_loss": val_loss}

    def validation_epoch_end(self, outputs):
        """
        Called at the end of validation to aggregate outputs.
        :param outputs: list of individual outputs of each validation step.
        """
        if not outputs:
            return {}

        if self.testing:
            prefix = 'test'
        else:
            prefix = 'val'

        avg_loss = torch.stack([x[f'val_loss'] for x in outputs]).mean()
        self.log(f'{prefix}_loss', avg_loss, prog_bar=True)


    def setup_training_data(self, train_data_config: Optional[DictConfig]):
        if not train_data_config and not (train_data_config.file_path or train_data_config.processed_datapath):
            logging.info(
                f"Dataloader config or file_path or processed data path for the train dataset is missing, \
                        so no data loader for test is created!"
            )
            
            self._train_dl = None
            return

        self._train_dl = self._setup_dataloader_from_config(cfg=train_data_config)

    def setup_validation_data(self, val_data_config: Optional[DictConfig]):
        if not val_data_config and not (val_data_config.file_path or val_data_config.processed_datapath):
            logging.info(
                f"Dataloader config or file_path or processed data path for the val dataset is missing, \
                        so no data loader for test is created!"
            )

            self._validation_dl = None
            return

        self._validation_dl = self._setup_dataloader_from_config(cfg=val_data_config)


    def _setup_dataloader_from_config(self, cfg: Dict) -> 'torch.utils.data.DataLoader':
        
        dataset = SelfAlignmentPretrainingDataset(
                name=cfg.name,
                datafile=cfg.datafile,
                tokenizer=self.tokenizer,
                max_len=cfg.max_seq_length,
                batch_size=cfg.batch_size,
                shuffle=cfg.shuffle,
            )
        
        return torch.utils.data.DataLoader(
                dataset=dataset,
                batch_size=dataset.batch_size,
                collate_fn=dataset.collate_fn,
                shuffle=False,
                pin_memory=cfg.get("pin_memory", False),
                num_workers=cfg.get("num_wokers", 0),
                )
    

    def _get_batch_embeddings(self, batch, batch_idx):
        """
        Lightning calls this inside the validation loop with the data from the validation dataloader
        passed in as `batch`.
        """
        input_ids, input_type_ids, input_mask, labels = batch
        input_ids = input_ids.to(device)
        input_type_ids = input_type_ids.to(device)
        input_mask = input_mask.to(device)
        logits = self.forward(input_ids=input_ids, 
                              token_type_ids=input_type_ids, 
                              attention_mask=input_mask)

        return logits

    def get_dataset_embeddings(self, rank, world_size, dataloader_type="train"):
        #if not torch.distributed.is_initialized():
        #    torch.distributed.init_process_group("gloo", rank=rank, world_size=world_size)
        #    print("Process initialized...")

        embeddings = []
        labels = []
        dataloaders = {"train": self._train_dl,
                       "val": self._validation_dl}
                        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(dataloaders[dataloader_type])):
                batch_embeddings = self._get_batch_embeddings(batch, batch_idx)
                embeddings.extend(batch_embeddings.cpu().numpy())
                labels.extend(batch[-1].numpy())

        pkl.dump(embeddings, open(self.config.index.embedding_save_name, "wb"))
        pkl.dump(labels, open(self.config.index.label_save_name, "wb"))

        return embeddings, labels


    def get_query_embedding(self, query):
        inputs = self.tokenizer(query,
                              add_special_tokens = True,
                              truncation = True,
                              max_length = self.dataset_cfg.max_seq_length,
                              return_token_type_ids = True,
                              return_attention_mask = True,
                              return_length = True)

        embedding = self.forward(input_ids=torch.LongTensor([inputs["input_ids"]]).to(device),
                                token_type_ids=torch.LongTensor([inputs["token_type_ids"]]).to(device),
                                attention_mask=torch.LongTensor([inputs["attention_mask"]]).to(device))
        
        return embedding[0]

    @classmethod
    def list_available_models(cls) -> Optional[Dict[str, str]]:
        pass

    @classmethod
    def from_pretrained(cls, name: str):
        pass


    
class SelfAlignmentPretrainingEvalModel(SelfAlignmentPretrainingModel):
    
    def setup_training_data(self, train_data_config: Optional[DictConfig]):
        return
    
    def setup_validation_data(self, val_data_config: Optional[DictConfig]):
        return
    
    def setup_test_data(self, test_data_config: Optional[DictConfig]):
        if not test_data_config and not (test_data_config.file_path or test_data_config.processed_datapath):
            logging.info(
                f"Dataloader config or file_path or processed data path for the val dataset is missing, \
                        so no data loader for test is created!"
            )

            self._test_dl = None
            return
        
        print("setting up test data")
        self._data_dl = self._setup_dataloader_from_config(cfg=test_data_config.data_ds, dataset_type="data")
        
        print("setting up query data")
        self._query_dl = self._setup_dataloader_from_config(cfg=test_data_config.query_ds, dataset_type="query")

    def _setup_dataloader_from_config(self, cfg: Dict, dataset_type=None) -> 'torch.utils.data.DataLoader':

        datasets = {"train": SelfAlignmentPretrainingDataset,
                    "val": SelfAlignmentPretrainingDataset,
                    "ncbi": SAP_NCBIEvalDataset,
                    "ncbi_query": SAP_QueryNCBIDataset,
                    "bc5cdr-d": SAP_NCBIEvalDataset,
                    "bc5cdr-d_query": SAP_QueryNCBIDataset,
                    "bc5cdr-c": SAP_NCBIEvalDataset,
                    "bc5cdr-c_query": SAP_QueryNCBIDataset,
                    "askapatient": SAP_AskAPatientDataset,
                    "askapatient_query": SAP_AskAPatientDataset,
                    "twadr-l": SAP_TWADRLDataset,
                    "twadr-l_query": SAP_TWADRLDataset,
                    "cometa": SAP_CometaDataset,
                    "cometa_query": SAP_CometaDataset,
                    }
        
        dataset = datasets[cfg.name](
                name=cfg.name,
                datafile=cfg.datafile,
                tokenizer=self.tokenizer,
                max_len=cfg.max_seq_length,
                batch_size=cfg.batch_size,
                shuffle=cfg.shuffle,
            )
        
        if dataset_type == "data":
            self.data_id2string = dataset.id2string
        elif dataset_type == "query":
            self.query_id2string = dataset.id2string
         
        return torch.utils.data.DataLoader(
                dataset=dataset,
                batch_size=dataset.batch_size,
                collate_fn=dataset.collate_fn,
                shuffle=False,
                pin_memory=cfg.get("pin_memory", False),
                num_workers=cfg.get("num_wokers", 0),
                )
    
    def get_dataset_embeddings(self, rank, world_size, dataloader_type="train"):
        #if not torch.distributed.is_initialized():
        #    torch.distributed.init_process_group("gloo", rank=rank, world_size=world_size)
        #    print("Process initialized...")

        embeddings = []
        labels = []
                        
        dataloaders = {"data": self._data_dl,
                       "query": self._query_dl}

        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(dataloaders[dataloader_type])):
                batch_embeddings = self._get_batch_embeddings(batch, batch_idx)
                embeddings.extend(batch_embeddings.cpu().numpy())
                labels.extend(batch[-1].numpy())

        #pkl.dump(embeddings, open(self.config.eval.embedding_save_name, "wb"))
        #pkl.dump(labels, open(self.config.eval.label_save_name, "wb"))

        return embeddings, labels
    
    def check_label(self, pred_label, query_label):
        """
        Some composite annotation didn't consider orders
        So, set label '1' if any cui is matched within composite cui (or single cui)
        Otherwise, set label '0'
        """
        return int(len(set(pred_label.split("|")).intersection(set(query_label.split("|"))))>0)

    def evaluate(self, ks, test_data_config, baseline=False, exact=False):
        data_name = test_data_config.data_ds.name
        print("getting query embeddings")
        query_embeds, query_label_ids = self.get_dataset_embeddings(rank=0,
                                                        world_size=1,
                                                        dataloader_type="query")
        
        if exact:
            query_labels = [self.query_id2string[label_id] for label_id in query_label_ids]
            
            print("Exact Nearest Neighbors, getting data embeddings")
            data_embeds, data_label_ids = self.get_dataset_embeddings(rank=0, 
                                                           world_size=1, 
                                                           dataloader_type="data")
            
            data_labels = [self.data_id2string[label_id] for label_id in data_label_ids]

            score_matrix = np.matmul(np.array(query_embeds), np.array(data_embeds).T)
            accs = {k : 0 for k in ks}
            
            for query_idx in tqdm(range(len(query_labels))):
                query = query_embeds[query_idx]
                query_label = query_labels[query_idx]
                query_scores = score_matrix[query_idx]
                
                for k in ks:
                    topk_idxs = np.argpartition(query_scores, -k)[-k:]
                    topk_labels = [data_labels[idx] for idx in topk_idxs]
                    matches = int(np.any([self.check_label(pred_label, query_label) for pred_label in topk_labels]))
                    accs[k] += matches
                    
            for k in ks:
                accs[k] /= len(query_labels)
        
        else:
            query_labels = query_label_ids
            index = AnnoyIndex(test_data_config.index_dims, "angular")
            if baseline:
                index.load(test_data_config.base_index)
            else:
                index.load(test_data_config.sap_index)
                
            accs = {k : 0 for k in ks}
            
            for query_idx in tqdm(range(len(query_labels))):
                query, query_label = query_embeds[query_idx], query_labels[query_idx]
                topk_labels = index.get_nns_by_vector(query, ks[-1]) # getting max topk number of nearest neighbors
                
                for k in ks:
                    matches = int(np.any([int(pred_label == query_label) for pred_label in topk_labels[:k]]))
                    accs[k] += matches 
                                                                  
            # Finish out accuarcy calculation for each k
            for k in ks:
                accs[k] /= len(query_labels)
        
        return accs
