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

from nemo import logging

__all__ = ['eval_iter_callback', 'eval_epochs_done_callback']

GLOBAL_KEYS = ["scores", "doc_rels"]


def eval_iter_callback(tensors, global_vars):
    for key in GLOBAL_KEYS:
        if key not in global_vars.keys():
            global_vars[key] = []

    for kv, v in tensors.items():

        if "scores" in kv:
            for scores in v:
                global_vars["scores"].append(scores[0].detach().cpu().numpy())

        if "rels" in kv:
            for doc_rels in v:
                global_vars["doc_rels"].append(doc_rels[0].detach().cpu().numpy())


def eval_epochs_done_callback(global_vars, topk=[10, 100]):
    doc_relevances = np.stack(global_vars["doc_rels"])
    doc_scores = np.stack(global_vars["scores"])
    mrrs = calculate_mrrs(doc_relevances, doc_scores, topk)

    logging.info("--------------------")
    for k in topk:
        oracle, bm25, model = mrrs[k]["oracle"], mrrs[k]["bm25"], mrrs[k]["model"]
        logging.info(f" Oracle MRR@{k}: {np.round(oracle, 3)}")
        logging.info(f"   BM25 MRR@{k}: {np.round(bm25, 3)}")
        logging.info(f"    Our MRR@{k}: {np.round(model, 3)}")
        logging.info("--------------------")

    for key in GLOBAL_KEYS:
        global_vars[key] = []

    metrics = dict({"mrr": mrrs[k]["model"]})

    return metrics


def calculate_mrrs(doc_relevances, doc_scores, topk=[10, 100]):
    """
    Args:
        doc_relevances: binary matrix of shape [n_queries x n_docs]
            with 1 for relevant documents and 0 otherwise
        doc_scores: matrix of shape [n_samples x n_docs] with
            relevance scores for all the documents for each query
        topk: list of ints {l_i} to compute MRR@{l_i}
    """

    inv_ids = (1 / np.arange(1, doc_relevances.shape[1] + 1))[None, :]
    mrrs = {}
    
    for k in topk:
        rels, invs, scores = doc_relevances[:, :k], inv_ids[:, :k], doc_scores[:, :k]
        oracle_mrr = np.mean(np.max(rels, axis=1))
        bm25_mrr = np.mean(np.max(rels * invs, axis=1))
        new_rels = np.array([
            [p[1] for p in sorted(zip(scores[i], rels[i]), key=lambda x: -x[0])]
            for i in range(len(scores))])
        new_mrr = np.mean(np.max(new_rels * invs, axis=1))
        mrrs[k] = {"oracle": oracle_mrr, "bm25": bm25_mrr, "model": new_mrr}
    return mrrs
