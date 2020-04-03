# Copyright (c) 2019 NVIDIA Corporation
from statistics import mean

import torch

from nemo import logging


def compute_accuracy(tensors):
    logging.info(f"Train Loss: {str(tensors[0].item())}")
    output = tensors[1]
    target = tensors[2]
    res = []
    topk = (1,)
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
    return "Train Top@1 accuracy  {0} ".format(res[0].item())


def eval_iter_callback(tensors, global_vars):
    if "eval_loss" not in global_vars.keys():
        global_vars["eval_loss"] = []
    for kv, v in tensors.items():
        if kv.startswith("loss"):
            global_vars["eval_loss"].append(torch.mean(torch.stack(v)).item())
            # global_vars['eval_loss'].append(v.item())

    if "top1" not in global_vars.keys():
        global_vars["top1"] = []

    output = None
    labels = None
    for kv, v in tensors.items():
        if kv.startswith("output"):
            # output = tensors[kv]
            output = torch.cat(tensors[kv])
        if kv.startswith("label"):
            # labels = tensors[kv]
            labels = torch.cat(tensors[kv])

    if output is None:
        raise Exception("output is None")

    res = []
    topk = (1,)
    with torch.no_grad():
        maxk = max(topk)
        batch_size = labels.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(labels.view(1, -1).expand_as(pred))

        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))

    global_vars["top1"].append(res[0].item())


def eval_epochs_done_callback(global_vars):
    eloss = mean(global_vars["eval_loss"])
    etop1 = mean(global_vars["top1"])
    logging.info("Evaluation Loss: {0}".format(eloss))
    logging.info("Evaluation Top@1: {0}".format(etop1))
    for k in global_vars.keys():
        global_vars[k] = []
    return dict({"Evaluation Loss": eloss, "Evaluation Top@1": etop1})
