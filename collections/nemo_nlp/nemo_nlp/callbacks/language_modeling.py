# Copyright (c) 2019 NVIDIA Corporation
import numpy as np


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

    print("------------------------------------------------------------")
    print("Validation loss: {0}".format(np.round(eval_loss, 3)))
    print("Validation  ppl: {0}".format(np.round(eval_ppl, 3)))
    print("------------------------------------------------------------")
    for key in GLOBAL_KEYS:
        global_vars[key] = []
    return dict({"Eval loss": eval_loss, "Eval ppl": eval_ppl})
