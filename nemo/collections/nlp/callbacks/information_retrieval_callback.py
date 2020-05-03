# =============================================================================
# Copyright 2020 NVIDIA. All Rights Reserved.
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
# =============================================================================

import numpy as np
import pickle
from nemo import logging

__all__ = ["eval_iter_callback", "eval_epochs_done_callback"]

GLOBAL_KEYS = ["scores", "query_id", "passage_ids"]


def eval_iter_callback(tensors, global_vars):
    for key in GLOBAL_KEYS:
        if key not in global_vars.keys():
            global_vars[key] = []

    for kv, v in tensors.items():

        if "scores" in kv:
            for tensor in v:
                global_vars["scores"].append(tensor[0].detach().cpu().numpy())

        if "query_id" in kv:
            for tensor in v:
                global_vars["query_id"].append(tensor.item())

        if "passage_ids" in kv:
            for tensor in v:
                global_vars["passage_ids"].append(tensor[0].detach().cpu().numpy())


def eval_epochs_done_callback(global_vars, query2rel, topk=[1, 10],
                              baseline_name="bm25", save_scores=None):

    query2passages = {}
    for i in range(len(global_vars["scores"])):
        query_id = global_vars["query_id"][i]

        if query_id in query2passages:
            query2passages[query_id]["psg_ids"] = np.concatenate(
                (query2passages[query_id]["psg_ids"], global_vars["passage_ids"][i]))
            query2passages[query_id]["scores"] = np.concatenate(
                (query2passages[query_id]["scores"], global_vars["scores"][i]))
        else:
            query2passages[query_id] = {
                "psg_ids": global_vars["passage_ids"][i],
                "scores": global_vars["scores"][i]}

    if save_scores is not None:
        pickle.dump(query2passages, open(save_scores, "wb"))

    if save_scores is not None:
        pickle.dump(query2passages, open(save_scores, "wb"))

    rrs = calculate_mrrs(query2passages, query2rel)
    oracle_mrr = np.mean(rrs["oracle"])

    logging.info("--------------------")
    logging.info(f"{baseline_name.upper()} oracle MRR: {np.round(oracle_mrr, 3)}")
    for k in topk:
        model_mrr = np.mean([(rr >= 1/k) * rr for rr in rrs["model"]])
        logging.info(f"{baseline_name.upper()} method MRR@{k}: {np.round(model_mrr, 3)}")
    logging.info("--------------------")
    logging.info(f"{baseline_name.upper()} oracle MRR: {np.round(oracle_mrr, 3)}")
    for k in topk:
        model_mrr = np.mean([(rr >= 1/k) * rr for rr in rrs["model"]])
        logging.info(f"{baseline_name.upper()} method MRR@{k}: {np.round(model_mrr, 3)}")
    logging.info("--------------------")

    for key in GLOBAL_KEYS:
        global_vars[key] = []
    metrics = dict({"mrr": model_mrr})

    return metrics


def calculate_mrrs(query2passages, query2rel, topk=[10, 100]):
    """
    Args:
        doc_relevances: binary matrix of shape [n_queries x n_docs]
            with 1 for relevant documents and 0 otherwise
        doc_scores: matrix of shape [n_samples x n_docs] with
            relevance scores for all the documents for each query
        topk: list of ints {l_i} to compute MRR@{l_i}
    """

    oracle_rrs, rrs = [], []

    for query in query2passages:
        indices = np.argsort(query2passages[query]["scores"])[::-1]
        sorted_psgs = query2passages[query]["psg_ids"][indices]

        oracle_rrs.append(0)
        rrs.append(0)
        for i, psg_id in enumerate(sorted_psgs):
            if psg_id in query2rel[query]:
                rrs[-1] = 1 / (i + 1)
                oracle_rrs[-1] = 1
                break
    rrs = {"oracle": oracle_rrs, "model": rrs}
    return rrs
