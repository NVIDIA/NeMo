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


def eval_iter_callback(tensors, global_vars):

    for kv, v in tensors.items():

        if 'smoothedcrossentropyloss' in kv:
            if "dev_mlm_loss" not in global_vars.keys():
                global_vars["dev_mlm_loss"] = []
            for dev_mlm_loss in v:
                global_vars["dev_mlm_loss"].append(dev_mlm_loss.item())
        if 'crossentropylossnm' in kv:
            if "dev_nsp_loss" not in global_vars.keys():
                global_vars["dev_nsp_loss"] = []
            for dev_nsp_loss in v:
                global_vars["dev_nsp_loss"].append(dev_nsp_loss.item())
        if 'lossaggregator' in kv:
            if "dev_loss" not in global_vars.keys():
                global_vars["dev_loss"] = []
            for dev_loss in v:
                global_vars["dev_loss"].append(dev_loss.item())


def eval_epochs_done_callback(global_vars):
    res = {}
    if 'dev_mlm_loss' in global_vars:
        mlm_loss = np.mean(global_vars["dev_mlm_loss"])
        logging.info("Dev MLM perplexity: {0}".format(np.round(np.exp(mlm_loss), 3)))
        global_vars["dev_mlm_loss"] = []
        res["Dev MLM loss"] = mlm_loss
    else:
        mlm_loss = -123.0

    if 'dev_nsp_loss' in global_vars:
        nsp_loss = np.mean(global_vars["dev_nsp_loss"])
        logging.info("Dev NSP perplexity: {0}".format(np.round(np.exp(nsp_loss), 3)))
        global_vars["dev_nsp_loss"] = []
        res["Dev NSP loss"] = nsp_loss
    else:
        nsp_loss = -123.0

    if 'dev_loss' in global_vars:
        total_loss = np.mean(global_vars["dev_loss"])
        logging.info("Dev perplexity: {0}".format(np.round(np.exp(total_loss), 3)))
        global_vars["dev_loss"] = []
        res["Dev loss"] = total_loss
    else:
        nsp_loss = -123.0

    return res
