# Copyright (c) 2019 NVIDIA Corporation
import numpy as np


def eval_iter_callback(tensors, global_vars):
    if "dev_mlm_loss" not in global_vars.keys():
        global_vars["dev_mlm_loss"] = []
    if "dev_nsp_loss" not in global_vars.keys():
        global_vars["dev_nsp_loss"] = []
    keys = list(tensors.keys())
    # TODO: referring to these by name here is error-prone
    for dev_mlm_loss in tensors[keys[1]]:
        global_vars["dev_mlm_loss"].append(dev_mlm_loss.item())

    if len(keys) > 2:
        for dev_nsp_loss in tensors[keys[2]]:
            global_vars["dev_nsp_loss"].append(dev_nsp_loss.item())


def eval_epochs_done_callback(global_vars):
    if 'dev_mlm_loss' in global_vars:
        mlm_loss = np.mean(global_vars["dev_mlm_loss"])
        print("Dev MLM perplexity: {0}".format(np.round(np.exp(mlm_loss), 3)))
        global_vars["dev_mlm_loss"] = []
    else:
        mlm_loss = -123.0

    if 'dev_nsp_loss' in global_vars:
        nsp_loss = np.mean(global_vars["dev_nsp_loss"])
        print("Dev NSP perplexity: {0}".format(np.round(np.exp(nsp_loss), 3)))
        global_vars["dev_nsp_loss"] = []
    else:
        nsp_loss = -123.0

    return dict({"Dev MLM loss": mlm_loss, "Dev NSP loss": nsp_loss})
