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

        if "logits" in kv:
            for scores in v:
                global_vars["scores"].append(scores.detach().cpu().numpy())

        if "doc_rels" in kv:
            for doc_rels in v:
                global_vars["doc_rels"].append(doc_rels.detach().cpu().numpy())


def eval_epochs_done_callback(global_vars):
    rel_mat = np.stack(global_vars["doc_rels"])
    inv_ids = 1 / np.arange(1, rel_mat.shape[1] + 1)
    oracle_mrr = np.mean(np.max(rel_mat, axis=1))
    bm25_mrr = np.mean(np.max(rel_mat * inv_ids[None, :], axis=1))

    new_rel_mat = np.array([
        [p[1] for p in sorted(zip(global_vars["scores"][i], global_vars["doc_rels"][i]), key=lambda x: -x[0])]
        for i in range(len(global_vars["scores"]))
    ])
    mrr = np.mean(np.max(new_rel_mat * inv_ids[None, :], axis=1))

    logging.info("--------------------")
    logging.info(" Oracle MRR: {0}".format(np.round(oracle_mrr, 3)))
    logging.info("   BM25 MRR: {0}".format(np.round(bm25_mrr, 3)))
    logging.info("    Our MRR: {0}".format(np.round(mrr, 3)))
    logging.info("--------------------")

    for key in GLOBAL_KEYS:
        global_vars[key] = []

    metrics = dict({"mrr": mrr})

    return metrics


# def eval_iter_callback(tensors, global_vars):
#     for key in GLOBAL_KEYS:
#         if key not in global_vars.keys():
#             global_vars[key] = []

#     for kv, v in tensors.items():
#         if "loss" in kv:
#             for eval_loss in v:
#                 global_vars["eval_loss"].append(eval_loss.item())


# def eval_epochs_done_callback(global_vars):
#     eval_loss = np.mean(global_vars["eval_loss"])

#     logging.info("------------------------------------------------------------")
#     logging.info("Validation loss: {0}".format(np.round(eval_loss, 3)))
#     logging.info("------------------------------------------------------------")

#     for key in GLOBAL_KEYS:
#         global_vars[key] = []

#     metrics = dict({"eval_loss": eval_loss})

#     return metrics
