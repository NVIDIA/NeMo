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
    "loss",
    "per_example_loss",
    "beam_results",
    "src_ids",
    "src_first_tokens",
    "pred",
    "labels",
    "labels_mask",
]


def eval_iter_callback(tensors, global_vars, tokenizer):
    for key in GLOBAL_KEYS:
        if key not in global_vars.keys():
            global_vars[key] = []

    for kv, v in tensors.items():

        if "crossentropylossnm1" in kv:
            for per_example_loss in v:
                pel = per_example_loss.cpu().numpy().tolist()
                global_vars["per_example_loss"].extend(pel)

        if "logits" in kv:
            for pred in v:
                p = torch.argmax(pred, dim=-1).int().cpu().numpy().tolist()
                global_vars["pred"].extend(p)

        if "labels~" in kv:
            for label in v:
                l = label.cpu().numpy().tolist()
                global_vars["labels"].extend(l)

        if "labels_mask" in kv:
            for mask in v:
                m = mask.cpu().numpy().tolist()
                global_vars["labels_mask"].extend(m)


def eval_epochs_done_callback(global_vars, validation_dataset=None):
    losses = np.array(global_vars["per_example_loss"])
    eval_loss = np.mean(losses)
    global_vars["per_example_loss"] = []

    labels = np.array([np.array(n) for n in global_vars["labels"]])
    predictions = np.array([np.array(n) for n in global_vars["pred"]])
    labels_mask = np.array([np.array(n) for n in global_vars["labels_mask"]])
    for key in GLOBAL_KEYS:
        global_vars[key] = []

    lor = np.logical_or(labels == predictions, ~labels_mask.astype(np.bool))
    accuracy = np.mean(np.all(lor, axis=1).astype(np.float32))

    logging.info("------------------------------------------------------------")
    logging.info("Validation loss: {0}".format(np.round(eval_loss, 3)))
    logging.info("Sentence level accuracy: {0}".format(accuracy))
    logging.info("------------------------------------------------------------")

    return dict({"eval_loss": eval_loss})
