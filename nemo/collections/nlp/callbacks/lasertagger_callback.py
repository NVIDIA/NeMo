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
import torch

from nemo import logging

__all__ = ['eval_iter_callback', 'eval_epochs_done_callback']

GLOBAL_KEYS = [
    "tgt_ids",
    "input_ids",
    "loss",
    "per_example_loss",
    "log_softmax",
    "beam_results",
    "src_ids",
    "src_first_tokens",
    "pred",
]


def eval_iter_callback(tensors, global_vars, tokenizer):
    for key in GLOBAL_KEYS:
        if key not in global_vars.keys():
            global_vars[key] = []

    for kv, v in tensors.items():

        if "tgt" in kv:
            ref = []
            for tgt in v:
                nonpad_tokens = (tgt != tokenizer.pad_id).sum().item()
                tgt_sentences = tgt.cpu().numpy().tolist()
                for sentence in tgt_sentences:
                    ref.append(tokenizer.ids_to_text(sentence))
                global_vars["nonpad_tokens"].append(nonpad_tokens)
            global_vars["ref"].append(ref)

        if "loss" in kv:
            for eval_loss in v:
                global_vars["loss"].append(eval_loss.item())

        if "per_example_loss" in kv:
            for eval_loss in v:
                global_vars["per_example_loss"].append(eval_loss.item())

        if "log_softmax" in kv:
            for eval_loss in v:
                global_vars["log_softmax"].append(eval_loss.item())

        if "src_ids" in kv:
            for token in v:
                global_vars["src_tokens"].append(token)


def eval_epochs_done_callback(global_vars, validation_dataset=None):
    losses = np.array(global_vars["loss"])
    counts = np.array(global_vars["nonpad_tokens"])
    eval_loss = np.sum(losses * counts) / np.sum(counts)

    all_sys = [j for i in global_vars["sys"] for j in i]
    _, indices = np.unique(global_vars["sent_ids"], return_index=True)
    all_sys = [all_sys[i] for i in indices]

    if validation_dataset is not None:
        all_ref = [open(validation_dataset, "r").readlines()]
    else:
        all_ref = [[j for i in global_vars["ref"] for j in i]]

    logging.info("------------------------------------------------------------")
    logging.info("Validation loss: {0}".format(np.round(eval_loss, 3)))
    logging.info("------------------------------------------------------------")

    for key in GLOBAL_KEYS:
        global_vars[key] = []

    return None
