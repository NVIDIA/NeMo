# Copyright (c) 2019 NVIDIA Corporation
__all__ = ['eval_iter_callback', 'eval_epochs_done_callback']

import os
import random
import time

import matplotlib
from matplotlib import pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report

from nemo.utils.exp_logging import get_logger


logger = get_logger('')


def eval_iter_callback(tensors,
                       global_vars,
                       eval_data_layer):
    # print(tensors)
    # print(global_vars)
    if 'loss' not in global_vars:
        global_vars['loss'] = []
    if 'point_outputs' not in global_vars:
        global_vars['point_outputs'] = []
    if 'gate_outputs' not in global_vars:
        global_vars['gate_outputs'] = []
    if 'gating_labels' not in global_vars:
        global_vars['gating_labels'] = []
    if 'gate_outputs' not in global_vars:
        global_vars['gate_outputs'] = []

    for kv, v in tensors.items():
        if kv.startswith('loss'):
            global_vars['loss'].append(v[0].cpu().numpy())
        if kv.startswith('point_outputs'):
            global_vars['point_outputs'].append(v[0].cpu().numpy())
        if kv.startswith('gate_outputs'):
            global_vars['gate_outputs'].append(v[0].cpu().numpy())


def list2str(l):
    return ' '.join([str(j) for j in l])


def eval_epochs_done_callback(global_vars, graph_fold):
    print(global_vars['loss'])
    return {}
