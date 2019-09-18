# Copyright (c) 2019 NVIDIA Corporation
import numpy as np

from nemo.utils.exp_logging import get_logger

GLOBAL_KEYS = ["eval_loss", "sys"]

logger = get_logger('')


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

    logger.info("------------------------------------------------------------")
    logger.info("Eval loss: {0}".format(np.round(eval_loss, 3)))
    logger.info("Eval  ppl: {0}".format(np.round(eval_ppl, 3)))
    logger.info("------------------------------------------------------------")
    for key in GLOBAL_KEYS:
        global_vars[key] = []
    return dict({"Eval_loss": eval_loss, "Eval_ppl": eval_ppl})
