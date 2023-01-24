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
from argparse import ArgumentParser
from collections import OrderedDict
from typing import Dict

import numpy as np
import torch
from build_index import load_model
from omegaconf import DictConfig, OmegaConf

from nemo.utils import logging

try:
    import faiss
except ModuleNotFoundError:
    logging.warning("Faiss is required for building the index. Please install faiss-gpu")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_query_embedding(query, model):
    """Use entity linking encoder to get embedding for index query"""
    model_input = model.tokenizer(
        query,
        add_special_tokens=True,
        padding=True,
        truncation=True,
        max_length=512,
        return_token_type_ids=True,
        return_attention_mask=True,
    )

    query_emb = model.forward(
        input_ids=torch.LongTensor([model_input["input_ids"]]).to(device),
        token_type_ids=torch.LongTensor([model_input["token_type_ids"]]).to(device),
        attention_mask=torch.LongTensor([model_input["attention_mask"]]).to(device),
    )

    return query_emb


def query_index(
    query: str, cfg: DictConfig, model: object, index: object, pca: object, idx2id: dict, id2string: dict,
) -> Dict:

    """
    Query the nearest neighbor index of entities to find the 
    concepts in the index dataset that are most similar to the 
    query.

    Args:
        query (str): entity to look up in the index
        cfg (DictConfig): config object to specifiy query parameters
        model (EntityLinkingModel): entity linking encoder model
        index (object): faiss index
        pca (object): sklearn pca transformation to be applied to queries 
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

    dist, neighbors = index.search(query_emb.astype(np.float32), cfg.query_num_factor * cfg.top_n)
    dist, neighbors = dist[0], neighbors[0]
    unique_ids = OrderedDict()
    neighbor_idx = 0

    # Many of nearest neighbors could map to the same concept id, their idx is their unique identifier
    while len(unique_ids) < cfg.top_n and neighbor_idx < len(neighbors):
        concept_id_idx = neighbors[neighbor_idx]
        concept_id = idx2id[concept_id_idx]

        # Only want one instance of each unique concept
        if concept_id not in unique_ids:
            concept = id2string[concept_id]
            unique_ids[concept_id] = (concept, 1 - dist[neighbor_idx])

        neighbor_idx += 1

    unique_ids = dict(unique_ids)

    return unique_ids


def main(cfg: DictConfig, restore: bool):
    """
    Loads faiss index and allows commandline queries 
    to the index. Builds new index if one hasn't been built yet.

    Args:
        cfg: Config file specifying index parameters
        restore: Whether to restore model weights trained
                 by the user. Otherwise will load weights
                 used before self alignment pretraining.
    """

    if not os.path.isfile(cfg.index.index_save_name) or (
        cfg.apply_pca and not os.path.isfile(cfg.index.pca.pca_save_name) or not os.path.isfile(cfg.index.idx_to_id)
    ):
        logging.warning("Either no index and/or no mapping from entity idx to ids exists. Please run `build_index.py`")
        return

    logging.info("Loading entity linking encoder model")
    model = load_model(cfg.model, restore)

    logging.info("Loading index and associated files")
    index = faiss.read_index(cfg.index.index_save_name)
    idx2id = pkl.load(open(cfg.index.idx_to_id, "rb"))
    id2string = pkl.load(open(cfg.index.id_to_string, "rb"))  # Should be created during dataset prep

    if cfg.index.apply_pca:
        pca = pkl.load(open(cfg.index.pca.pca_save_name, "rb"))

    while True:
        query = input("enter index query: ")
        output = query_index(query, cfg.top_n, cfg.index, model, index, pca, idx2id, id2string)

        if query == "exit":
            break

        for concept_id in output:
            concept_details = output[concept_id]
            concept_id = "C" + str(concept_id).zfill(7)
            print(concept_id, concept_details)

        print("----------------\n")


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
