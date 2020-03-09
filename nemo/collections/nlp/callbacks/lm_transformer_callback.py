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
GLOBAL_KEYS = ["eval_loss", "sys"]


def eval_iter_callback(tensors, global_vars):
    for key in GLOBAL_KEYS:
        if key not in global_vars.keys():
            global_vars[key] = []

    for kv, v in tensors.items():
        if "loss" in kv:
            for eval_loss in v:
                global_vars["eval_loss"].append(eval_loss.item())


def eval_epochs_done_callback(global_vars):
    eval_loss = np.mean(global_vars["eval_loss"])
    eval_ppl = np.exp(eval_loss)

    logging.info("------------------------------------------------------")
    logging.info("Eval loss: {0}".format(np.round(eval_loss, 3)))
    logging.info("Eval  ppl: {0}".format(np.round(eval_ppl, 3)))
    logging.info("------------------------------------------------------")
    for key in GLOBAL_KEYS:
        global_vars[key] = []
    return dict({"Eval_loss": eval_loss, "Eval_ppl": eval_ppl})
