import os
import torch
import random
import faiss
import h5py
import numpy as np
import pickle as pkl

from tqdm import tqdm
from argparse import ArgumentParser
from sklearn.decomposition import PCA
from typing import Optional, Dict, List
from omegaconf import DictConfig, OmegaConf
from collections import OrderedDict
from nemo.collections.nlp.data import EntityLinkingDataset
from nemo.collections.nlp.models import EntityLinkingModel
from nemo.utils import logging

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def build_index(cfg: DictConfig, model: object): 
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
        index_dataloader = model.setup_dataloader(cfg.dataset, is_index_data=True)
        embeddings, concept_ids = get_index_embeddings(cfg, index_dataloader, model)

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


def get_index_embeddings(cfg: DictConfig, dataloader, model: object):
    embeddings = []
    concept_ids = []
                    
    with torch.no_grad():
        for batch in tqdm(dataloader):
            input_ids, token_type_ids, input_mask, batch_concept_ids = batch
            input_ids = input_ids.to(device)
            token_type_ids = token_type_ids.to(device)
            input_mask = input_mask.to(device)
            batch_embeddings = model.forward(input_ids=input_ids,
                                   token_type_ids=token_type_ids,
                                   attention_mask=input_mask)

            embeddings.extend(batch_embeddings.detach().cpu().numpy())
            concept_ids.extend(batch_concept_ids.numpy())

    emb_file = h5py.File(cfg.embedding_save_name, "w")
    emb_file.create_dataset(cfg.dataset.name, data=embeddings)
    emb_file.close()

    pkl.dump(concept_ids, open(cfg.concept_id_save_name, "wb"))

    return embeddings, concept_ids 


def get_query_embedding(query, model):
    model_input = model.tokenizer(query,
                                add_special_tokens = True,
                                padding = True,
                                truncation = True,
                                max_length = 512,
                                return_token_type_ids = True,
                                return_attention_mask = True,
                                )

    query_emb = model.forward(input_ids=torch.LongTensor([model_input["input_ids"]]).to(device),
                             token_type_ids=torch.LongTensor([model_input["token_type_ids"]]).to(device),
                             attention_mask=torch.LongTensor([model_input["attention_mask"]]).to(device))

    return query_emb


def query_index(query: str, 
                top_n: int,
                cfg: DictConfig, 
                model: object, 
                index: object,
                pca: object,
                idx2id: dict,
                id2string: dict,
                ) -> Dict:
    """
    Query the nearest neighbor index of entities to find the 
    concepts in the index dataset that are most similar to the 
    query.

    Args:
        query (str): entity to look up in the index
        cfg (DictConfig): config object to specifiy query parameters
        model (EntityLinkingModel): entity linking encoder model
        index: faiss index
        idx2id (dict): dictionary mapping unique concept dataset index to 
                       its CUI
        id2string (dict): dictionary mapping each unqiue CUI to a 
                          representative english description of
                          the concept
    Returns:
        A dictionary with the concept ids of the index's most similar 
        entities as the keys and a tuple containing the string 
        representation of that concept and its cosine similarity to 
        the query as the values. 
    """
    query_emb = get_query_embedding(query, model).detach().cpu().numpy()

    if cfg.apply_pca:
        query_emb = pca.transform(query_emb)

    dist, neighbors = index.search(query_emb.astype(np.float32), cfg.query_num_factor*top_n)
    dist, neighbors = dist[0], neighbors[0]
    unique_ids = OrderedDict()
    neighbor_idx = 0

    # Many of nearest neighbors could map to the same concept id, their idx is their unique identifier
    while len(unique_ids) < top_n and neighbor_idx < len(neighbors):
        concept_id_idx = neighbors[neighbor_idx]
        concept_id = idx2id[concept_id_idx]

        # Only want one instance of each unique concept
        if concept_id not in unique_ids:
            concept = id2string[concept_id]
            unique_ids[concept_id] = (concept, 1 - dist[neighbor_idx])

        neighbor_idx += 1

    unique_ids = dict(unique_ids)

    return unique_ids


def main(cfg: DictConfig, top_n: int, restore: bool):
    """
    Loads faiss index and allows commandline queries 
    to the index. Builds new index if one hasn't been built yet.

    Args:
        cfg: Config file specifying index parameters
    """

    logging.info("Loading entity linking encoder model")
    if restore:
        model = EntityLinkingModel.restore_from(cfg.model.nemo_path)
    else:
        cfg.model.train_ds = None
        cfg.model.validation_ds = None
        cfg.model.test_ds = None
        model = EntityLinkingModel(cfg.model)

    model = model.to(device)

    if not os.path.isfile(cfg.index.index_save_name):
        build_index(cfg.index, model)

    logging.info("Loading index and associated files")
    index = faiss.read_index(cfg.index.index_save_name)
    idx2id = pkl.load(open(cfg.index.idx_to_id, "rb"))
    id2string = pkl.load(open(cfg.index.id_to_string, "rb"))

    if cfg.index.apply_pca:
        pca = pkl.load(open(cfg.index.pca.pca_save_name, "rb"))

    while True:
        query = input("enter index query: ")
        output = query_index(query, top_n, cfg.index, model, index, pca, idx2id, id2string)

        if query == "exit":
            break

        for concept_id in output:
            concept_details = output[concept_id]
            concept_id = "C" + str(concept_id).zfill(7)
            print(concept_id, concept_details)

        print("----------------\n")

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--restore", action="store_true", help="Whether to restore encoder model weights from nemo path") 
    parser.add_argument("--cfg", required=False, type=str, default="./conf/medical_entity_linking_config_pubmed.yaml")
    parser.add_argument("--top_n", required=False, type=int, default=5, help="Max number of items returned per query") 
    args = parser.parse_args()

    cfg = OmegaConf.load(args.cfg)
    main(cfg, args.top_n, args.restore)
