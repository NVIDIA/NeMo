# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import pickle as pkl
import random
from argparse import ArgumentParser

import h5py
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from sklearn.decomposition import PCA
from tqdm import tqdm

from nemo.collections.nlp.models import EntityLinkingModel
from nemo.utils import logging

try:
    import faiss
except ModuleNotFoundError:
    logging.warning("Faiss is required for building the index. Please install faiss-gpu")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def build_index(cfg: DictConfig, model: object):
    """
    Builds faiss index from index dataset specified in the config.
        
    Args:
        cfg (DictConfig): Config file specifying index parameters
        model (object): Encoder model
    """

    # Get index dataset embeddings
    # PCA model exists and index embeddings have already been PCAed, no need to re-extract/PCA them
    if cfg.apply_pca and os.path.isfile(cfg.pca.pca_save_name) and os.path.isfile(cfg.pca_embeddings_save_name):
        logging.info("Loading reduced dimensionality embeddings")
        embeddings = h5py.File(cfg.pca_embeddings_save_name, "r")
        embeddings = embeddings[cfg.index_ds.name][:]

    elif os.path.isfile(cfg.embedding_save_name):
        logging.info("Loading previously extracted index dataset embeddings")
        embeddings = h5py.File(cfg.embedding_save_name, "r")
        embeddings = embeddings[cfg.index_ds.name][:]

    else:
        logging.info("Encoding index dataset, this may take a while")
        index_dataloader = model.setup_dataloader(cfg.index_ds, is_index_data=True)
        embeddings, concept_ids = get_index_embeddings(cfg, index_dataloader, model)

    # Create pca model to reduce dimensionality of index dataset and decrease memory footprint
    if cfg.apply_pca:

        # Need to train PCA model and apply PCA transformation with newly trained model
        if not os.path.isfile(cfg.pca.pca_save_name):
            logging.info("Fitting PCA model for embedding dimensionality reduction")
            pca_train_set = random.sample(list(embeddings), k=int(len(embeddings) * cfg.pca.sample_fraction))
            pca = PCA(n_components=cfg.pca.output_dim)
            pca.fit(pca_train_set)
            pkl.dump(pca, open(cfg.pca.pca_save_name, "wb"))
            embeddings = reduce_embedding_dim(pca, embeddings, cfg)

        # PCA model already trained, just need to reduce dimensionality of all embeddings
        elif not os.path.isfile(cfg.pca_embeddings_save_name):
            pca = pkl.load(open(cfg.pca.pca_save_name, "rb"))
            embeddings = reduce_embedding_dim(pca, embeddings, cfg)

    # Build faiss index from embeddings
    logging.info(f"Training index with embedding dim size {cfg.dims} using {faiss.get_num_gpus()} gpus")
    quantizer = faiss.IndexFlatL2(cfg.dims)
    index = faiss.IndexIVFFlat(quantizer, cfg.dims, cfg.nlist)
    index = faiss.index_cpu_to_all_gpus(index)
    index.train(embeddings)

    logging.info("Adding dataset embeddings to index")
    for i in tqdm(range(0, embeddings.shape[0], cfg.index_batch_size)):
        index.add(embeddings[i : i + cfg.index_batch_size])

    logging.info("Saving index")
    faiss.write_index(faiss.index_gpu_to_cpu(index), cfg.index_save_name)
    logging.info("Index built and saved")


def reduce_embedding_dim(pca, embeddings, cfg):
    """Apply PCA transformation to index dataset embeddings"""

    logging.info("Applying PCA transformation to entire index dataset")
    embeddings = np.array(pca.transform(embeddings), dtype=np.float32)
    emb_file = h5py.File(cfg.pca_embeddings_save_name, "w")
    emb_file.create_dataset(cfg.index_ds.name, data=embeddings)
    emb_file.close()

    return embeddings


def get_index_embeddings(cfg: DictConfig, dataloader: object, model: object):
    """Use entity linking encoder to get embeddings for full index dataset"""
    embeddings = []
    concept_ids = []

    with torch.no_grad():
        for batch in tqdm(dataloader):
            input_ids, token_type_ids, input_mask, batch_concept_ids = batch
            input_ids = input_ids.to(device)
            token_type_ids = token_type_ids.to(device)
            input_mask = input_mask.to(device)
            batch_embeddings = model.forward(
                input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=input_mask
            )

            embeddings.extend(batch_embeddings.detach().cpu().numpy())
            concept_ids.extend(batch_concept_ids.numpy())

    emb_file = h5py.File(cfg.embedding_save_name, "w")
    emb_file.create_dataset(cfg.index_ds.name, data=embeddings)
    emb_file.close()

    pkl.dump(concept_ids, open(cfg.concept_id_save_name, "wb"))

    return embeddings, concept_ids


def load_model(cfg: DictConfig, restore: bool):
    """
    Loads encoder model.

    Args:
        cfg: Config file specifying model parameters
        restore: Whether to restore model weights trained
                 by the user. Otherwise will load weights
                 used before self alignment pretraining. 
    """

    if restore:
        model = EntityLinkingModel.restore_from(cfg.nemo_path)
    else:
        cfg.train_ds = None
        cfg.validation_ds = None
        cfg.test_ds = None
        model = EntityLinkingModel(cfg)

    model = model.to(device)

    return model


def main(cfg: DictConfig, restore: bool):
    """
    Builds new index if one hasn't been built yet.

    Args:
        cfg: Config file specifying index parameters
        restore: Whether to restore model weights trained
                 by the user. Otherwise will load weights
                 used before self alignment pretraining.
    """

    logging.info("Loading entity linking encoder model")
    model = load_model(cfg.model, restore)

    if not os.path.isfile(cfg.index.index_save_name) or (
        cfg.apply_pca and not os.path.isfile(cfg.index.pca.pca_save_name)
    ):
        logging.info("Building index")
        build_index(cfg.index, model)
    else:
        logging.info("Index and pca model (if required) already exists. Skipping build index step.")

    if not os.path.isfile(cfg.index.idx_to_id):
        logging.info("Mapping entity index postions to ids")
        map_idx_to_ids(cfg.index)
    else:
        logging.info("Map from concept index to id already exists. Skipping mapping step.")


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument(
        "--restore", action="store_true", help="Whether to restore encoder model weights from nemo path"
    )
    parser.add_argument("--project_dir", required=False, type=str, default=".")
    parser.add_argument("--cfg", required=False, type=str, default="./conf/umls_medical_entity_linking_config.yaml")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.cfg)
    cfg.project_dir = args.project_dir

    main(cfg, args.restore)
