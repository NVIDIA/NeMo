from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Callable, Optional

import torch
import torch.nn.functional as F
from torch import nn

from nemo.collections.llm.bert.model.base import BertConfig, BertModel
from nemo.collections.llm.utils import Config
from nemo.lightning import OptimizerModule, io, teardown

@dataclass
class GoogleBERTConfig(BertConfig):
    transformer_block_type: str = 'post_ln'
    add_pooler: bool = True
    add_lm_head: bool = True
    init_method_std: float = 0.02
    hidden_dropout: float = 0.1
    normalization: float = 'LayerNorm'
    layernorm_epsilon: float = 1e-5
    

@dataclass
class GoogleBERTBaseConfig(GoogleBERTConfig):
    num_layers: int = 12
    hidden_size: int = 768
    ffn_hidden_size: int = 3072
    num_attention_heads: int = 12
