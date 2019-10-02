# Copyright (c) 2019 NVIDIA Corporation
"""
This package contains BERT Neural Module
"""
import torch
import torch.nn as nn

from nemo.backends.pytorch.nm import TrainableNM, LossNM
from nemo.core.neural_types import *
from .transformer import ClassificationLogSoftmax

from .transformer import SequenceClassificationLoss
from .transformer.utils import transformer_weights_init








