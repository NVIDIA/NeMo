# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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

from dataclasses import dataclass, is_dataclass
from typing import Any, Optional

import torch
from torch import nn as nn
from hydra.utils import instantiate
from omegaconf import OmegaConf

from nemo.collections.asr.parts.submodules import multi_head_attention as mha
from nemo.collections.common.parts import adapter_modules
from nemo.core.classes.mixins import adapter_mixin_strategies


class MultiHeadAttentionAdapter(mha.MultiHeadAttention, adapter_modules.AbstractAdapterModuleMixin):
    def __init__(
        self,
        n_head: int,
        n_feat: int,
        dropout_rate: float,
        adapter_strategy: adapter_mixin_strategies.ResidualAddAdapterStrategyConfig = None,
    ):
        super().__init__(n_head=n_head, n_feat=n_feat, dropout_rate=dropout_rate, max_cache_len=0)

        # Setup adapter strategy
        self.setup_adapter_strategy(adapter_strategy)

        # reset parameters for Q to be identity operation
        self.reset_parameters()

    # Don't override forward
    # def forward(self, query, key, value, mask, pos_emb=None, cache=None, cache_next=None):
    #     """Compute 'Scaled Dot Product Attention'.
    #     Args:
    #         query (torch.Tensor): (batch, time1, size)
    #         key (torch.Tensor): (batch, time2, size)
    #         value(torch.Tensor): (batch, time2, size)
    #         mask (torch.Tensor): (batch, time1, time2)
    #         cache (torch.Tensor) : (cache_nums, batch, time_cache, size)
    #         cache_next (torch.Tensor) : (cache_nums, batch, time_cache_next, size)
    #
    #     returns:
    #         output (torch.Tensor): transformed `value` (batch, time1, d_model) weighted by the query dot key attention
    #     """

    def reset_parameters(self):
        with torch.no_grad():
            self.linear_out.weight *= 0
            self.linear_out.bias *= 0


@dataclass
class MultiHeadAttentionAdapterConfig:
    n_head: int
    n_feat: int
    dropout_rate: float = 0.0
    adapter_strategy: Optional[Any] = adapter_mixin_strategies.ResidualAddAdapterStrategyConfig()
    _target_: str = "{0}.{1}".format(MultiHeadAttentionAdapter.__module__, MultiHeadAttentionAdapter.__name__)


class RelPositionMultiHeadAttentionAdapter(
    mha.RelPositionMultiHeadAttention, adapter_modules.AbstractAdapterModuleMixin
):
    def __init__(
        self,
        n_head: int,
        n_feat: int,
        dropout_rate: float,
        adapter_strategy: adapter_mixin_strategies.ResidualAddAdapterStrategyConfig = None,
    ):
        super().__init__(
            n_head=n_head, n_feat=n_feat, dropout_rate=dropout_rate, pos_bias_u=None, pos_bias_v=None, max_cache_len=0
        )

        # Setup adapter strategy
        self.setup_adapter_strategy(adapter_strategy)

        # reset parameters for Q to be identity operation
        self.reset_parameters()

    # def forward(self, query, key, value, mask, pos_emb, cache=None, cache_next=None):
    #     """Compute 'Scaled Dot Product Attention' with rel. positional encoding.
    #     Args:
    #         query (torch.Tensor): (batch, time1, size)
    #         key (torch.Tensor): (batch, time2, size)
    #         value(torch.Tensor): (batch, time2, size)
    #         mask (torch.Tensor): (batch, time1, time2)
    #         pos_emb (torch.Tensor) : (batch, time1, size)
    #         cache (torch.Tensor) : (cache_nums, batch, time_cache, size)
    #         cache_next (torch.Tensor) : (cache_nums, batch, time_cache_next, size)
    #     Returns:
    #         output (torch.Tensor): transformed `value` (batch, time1, d_model) weighted by the query dot key attention
    #     """

    def reset_parameters(self):
        with torch.no_grad():
            self.linear_out.weight *= 0.0
            self.linear_out.bias *= 0.0

            self.pos_bias_u *= 0.0
            self.pos_bias_v *= 0.0


@dataclass
class RelPositionMultiHeadAttentionAdapterConfig:
    n_head: int
    n_feat: int
    dropout_rate: float = 0.0
    adapter_strategy: Optional[Any] = adapter_mixin_strategies.ResidualAddAdapterStrategyConfig()
    _target_: str = "{0}.{1}".format(
        RelPositionMultiHeadAttentionAdapter.__module__, RelPositionMultiHeadAttentionAdapter.__name__
    )


class PositionalEncodingAdapter(mha.PositionalEncoding, adapter_modules.AbstractAdapterModuleMixin):
    def __init__(
        self,
        d_model: int,
        max_len: int = 5000,
        xscale=1.0,
        adapter_strategy: adapter_mixin_strategies.ResidualAddAdapterStrategyConfig = None,
    ):
        super().__init__(
            d_model=d_model, dropout_rate=0.0, max_len=max_len, xscale=xscale, dropout_rate_emb=0.0,
        )

        # Setup adapter strategy
        self.setup_adapter_strategy(adapter_strategy)

    # def forward(self, x: torch.Tensor):
    #     """Adds positional encoding.
    #     Args:
    #         x (torch.Tensor): Input. Its shape is (batch, time, feature_size)
    #     Returns:
    #         x+pos_emb (torch.Tensor): Its shape is (batch, time, feature_size)
    #         pos_emb (torch.Tensor): Its shape is (1, time, feature_size)
    #     """


@dataclass
class PositionalEncodingAdapterConfig:
    d_model: int
    max_len: int = 5000
    xscale: float = 1.0
    adapter_strategy: Optional[Any] = adapter_mixin_strategies.ResidualAddAdapterStrategyConfig()
    _target_: str = "{0}.{1}".format(
        PositionalEncodingAdapter.__module__, PositionalEncodingAdapter.__name__
    )


class RelPositionalEncodingAdapter(mha.RelPositionalEncoding, adapter_modules.AbstractAdapterModuleMixin):
    def __init__(
        self,
        d_model: int,
        max_len: int = 5000,
        xscale=1.0,
        adapter_strategy: adapter_mixin_strategies.ResidualAddAdapterStrategyConfig = None,
    ):
        super().__init__(d_model=d_model, dropout_rate=0.0, max_len=max_len, xscale=xscale, dropout_rate_emb=0.0)

        # Setup adapter strategy
        self.setup_adapter_strategy(adapter_strategy)

    # def forward(self, x, cache_len=0):
    #     """Compute positional encoding.
    #     Args:
    #         x (torch.Tensor): Input. Its shape is (batch, time, feature_size)
    #         cache_len (int): the size of the cache which is used to shift positions
    #     Returns:
    #         x (torch.Tensor): Its shape is (batch, time, feature_size)
    #         pos_emb (torch.Tensor): Its shape is (1, time, feature_size)
    #     """


@dataclass
class RelPositionalEncodingAdapterConfig:
    d_model: int
    max_len: int = 5000
    xscale: float = 1.0
    adapter_strategy: Optional[Any] = adapter_mixin_strategies.ResidualAddAdapterStrategyConfig()
    _target_: str = "{0}.{1}".format(
        RelPositionalEncodingAdapter.__module__, RelPositionalEncodingAdapter.__name__
    )
