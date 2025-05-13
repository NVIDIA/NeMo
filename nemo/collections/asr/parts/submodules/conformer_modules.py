# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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
#

import torch
from torch import nn as nn
from torch.nn import LayerNorm
from torch.nn import RMSNorm

from nemo.collections.asr.parts.submodules.adapters.attention_adapter_mixin import AttentionAdapterModuleMixin
from nemo.collections.asr.parts.submodules.batchnorm import FusedBatchNorm1d
from nemo.collections.asr.parts.submodules.causal_convs import CausalConv1D
from nemo.collections.asr.parts.submodules.multi_head_attention import (
    MultiHeadAttention,
    RelPositionMultiHeadAttention,
    RelPositionMultiHeadAttentionLongformer,
)
from nemo.collections.asr.parts.utils.activations import Swish
from nemo.collections.common.parts.utils import activation_registry
from nemo.core.classes.mixins import AccessMixin

__all__ = ['ConformerConvolution', 'ConformerFeedForward', 'ConformerLayer']


class ConformerLayer(torch.nn.Module, AttentionAdapterModuleMixin, AccessMixin):
    """A single block of the Conformer encoder.

    Args:
        d_model (int): input dimension of MultiheadAttentionMechanism and PositionwiseFeedForward
        d_ff (int): hidden dimension of PositionwiseFeedForward
        self_attention_model (str): type of the attention layer and positional encoding
            'rel_pos': relative positional embedding and Transformer-XL
            'rel_pos_local_attn': relative positional embedding and Transformer-XL with local attention using
                overlapping chunks. Attention context is determined by att_context_size parameter.
            'abs_pos': absolute positional embedding and Transformer
            Default is rel_pos.
        global_tokens (int): number of tokens to be used for global attention.
            Only relevant if self_attention_model is 'rel_pos_local_attn'.
            Defaults to 0.
        global_tokens_spacing (int): how far apart the global tokens are
            Defaults to 1.
        global_attn_separate (bool): whether the q, k, v layers used for global tokens should be separate.
            Defaults to False.
        n_heads (int): number of heads for multi-head attention
        conv_norm_type (str): normalization type for convolution module
        conv_kernel_size (int): kernel size for depthwise convolution in convolution module
        dropout (float): dropout probabilities for linear layers
        dropout_att (float): dropout probabilities for attention distributions
        use_bias (bool): Apply bias to all Linear and Conv1d layers from each ConformerLayer to improve activation flow and stabilize training of huge models.
            Defaults to True.
        use_convolution (bool): Whether to use the convolution module. Defaults to True.
        ffn_activation_name (str): activation name for the feed-forward module
    """

    def __init__(
        self,
        d_model,
        d_ff,
        self_attention_model='rel_pos',
        global_tokens=0,
        global_tokens_spacing=1,
        global_attn_separate=False,
        n_heads=4,
        conv_kernel_size=31,
        ff_norm_type='layer_norm',
        conv_norm_type='batch_norm',
        conv_context_size=None,
        dropout=0.1,
        dropout_att=0.1,
        pos_bias_u=None,
        pos_bias_v=None,
        att_context_size=[-1, -1],
        use_bias=True,
        use_pytorch_sdpa=False,
        use_pytorch_sdpa_backends=None,
        use_convolution=True,
        use_pre_mlp=True,
        ffn_activation_name='swish',
    ):
        super(ConformerLayer, self).__init__()

        self.use_pytorch_sdpa = use_pytorch_sdpa
        if use_pytorch_sdpa_backends is None:
            use_pytorch_sdpa_backends = []
        self.use_pytorch_sdpa_backends = use_pytorch_sdpa_backends
        self.self_attention_model = self_attention_model
        self.n_heads = n_heads
        self.fc_factor = 0.5
        self.ff_norm_type = ff_norm_type
        self.conv_norm_type = conv_norm_type
        self.use_convolution = use_convolution
        self.use_pre_mlp = use_pre_mlp
        self.ffn_activation_name = ffn_activation_name

        # first feed forward module
        if self.use_pre_mlp:
            if ff_norm_type == 'rms_norm':
                self.norm_feed_forward1 = RMSNorm(d_model)
            else:
                self.norm_feed_forward1 = LayerNorm(d_model)
            self.feed_forward1 = ConformerFeedForward(
                d_model=d_model, d_ff=d_ff, dropout=dropout, activation_name=self.ffn_activation_name, use_bias=use_bias
            )
        else:
            self.norm_feed_forward1 = None
            self.feed_forward1 = None

        # convolution module (conditional)
        if self.use_convolution:
            if ff_norm_type == 'rms_norm':
                self.norm_conv = RMSNorm(d_model)
            else:
                self.norm_conv = LayerNorm(d_model)
            self.conv = ConformerConvolution(
                d_model=d_model,
                kernel_size=conv_kernel_size,
                norm_type=conv_norm_type,
                conv_context_size=conv_context_size,
                use_bias=use_bias,
            )
        else:
            self.norm_conv = None
            self.conv = None

        # multi-headed self-attention module
        if ff_norm_type == 'rms_norm':
            self.norm_self_att = RMSNorm(d_model)
        else:
            self.norm_self_att = LayerNorm(d_model)
        MHA_max_cache_len = att_context_size[0]

        if self_attention_model == 'rel_pos':
            self.self_attn = RelPositionMultiHeadAttention(
                n_head=n_heads,
                n_feat=d_model,
                dropout_rate=dropout_att,
                pos_bias_u=pos_bias_u,
                pos_bias_v=pos_bias_v,
                max_cache_len=MHA_max_cache_len,
                use_bias=use_bias,
                use_pytorch_sdpa=self.use_pytorch_sdpa,
                use_pytorch_sdpa_backends=self.use_pytorch_sdpa_backends,
            )
        elif self_attention_model == 'rel_pos_local_attn':
            self.self_attn = RelPositionMultiHeadAttentionLongformer(
                n_head=n_heads,
                n_feat=d_model,
                dropout_rate=dropout_att,
                pos_bias_u=pos_bias_u,
                pos_bias_v=pos_bias_v,
                max_cache_len=MHA_max_cache_len,
                att_context_size=att_context_size,
                global_tokens=global_tokens,
                global_tokens_spacing=global_tokens_spacing,
                global_attn_separate=global_attn_separate,
                use_bias=use_bias,
            )
        elif self_attention_model == 'abs_pos':
            self.self_attn = MultiHeadAttention(
                n_head=n_heads,
                n_feat=d_model,
                dropout_rate=dropout_att,
                max_cache_len=MHA_max_cache_len,
                use_bias=use_bias,
                use_pytorch_sdpa=self.use_pytorch_sdpa,
                use_pytorch_sdpa_backends=self.use_pytorch_sdpa_backends,
            )
        else:
            raise ValueError(
                f"'{self_attention_model}' is not not a valid value for 'self_attention_model', "
                f"valid values can be from ['rel_pos', 'rel_pos_local_attn', 'abs_pos']"
            )

        # second feed forward module
        if ff_norm_type == 'rms_norm':
            self.norm_feed_forward2 = RMSNorm(d_model)
        else:
            self.norm_feed_forward2 = LayerNorm(d_model)
        self.feed_forward2 = ConformerFeedForward(
            d_model=d_model, d_ff=d_ff, dropout=dropout, activation_name=self.ffn_activation_name, use_bias=use_bias
        )

        self.dropout = nn.Dropout(dropout)
        if ff_norm_type == 'rms_norm':
            self.norm_out = RMSNorm(d_model)
        else:
            self.norm_out = LayerNorm(d_model)

    def forward(self, x, att_mask=None, pos_emb=None, pad_mask=None, cache_last_channel=None, cache_last_time=None):
        """
        Args:
            x (torch.Tensor): input signals (B, T, d_model)
            att_mask (torch.Tensor): attention masks(B, T, T)
            pos_emb (torch.Tensor): (L, 1, d_model)
            pad_mask (torch.tensor): padding mask
            cache_last_channel (torch.tensor) : cache for MHA layers (B, T_cache, d_model)
            cache_last_time (torch.tensor) : cache for convolutional layers (B, d_model, T_cache)
        Returns:
            x (torch.Tensor): (B, T, d_model)
            cache_last_channel (torch.tensor) : next cache for MHA layers (B, T_cache, d_model)
            cache_last_time (torch.tensor) : next cache for convolutional layers (B, d_model, T_cache)
        """
        current_input = x

        # First FFN module
        if self.use_pre_mlp:
            residual = current_input
            x_ffn1 = self.norm_feed_forward1(current_input)
            x_ffn1 = self.feed_forward1(x_ffn1)
            current_input = residual + self.dropout(x_ffn1) * self.fc_factor
        # If not self.use_pre_mlp, current_input remains the original x

        # Self-attention module
        residual = current_input
        x_attn = self.norm_self_att(current_input)

        if self.self_attention_model == 'rel_pos':
            x_attn = self.self_attn(query=x_attn, key=x_attn, value=x_attn, mask=att_mask, pos_emb=pos_emb, cache=cache_last_channel)
        elif self.self_attention_model == 'rel_pos_local_attn':
            x_attn = self.self_attn(query=x_attn, key=x_attn, value=x_attn, pad_mask=pad_mask, pos_emb=pos_emb, cache=cache_last_channel)
        elif self.self_attention_model == 'abs_pos':
            x_attn = self.self_attn(query=x_attn, key=x_attn, value=x_attn, mask=att_mask, cache=cache_last_channel)
        # self_attention_model is validated in __init__, so no else needed.

        if x_attn is not None and cache_last_channel is not None and isinstance(x_attn, tuple): # MHA returned (output, cache)
             x_attn, cache_last_channel = x_attn
        elif isinstance(x_attn, tuple): # MHA returned a tuple but no cache was expected/returned
             x_attn = x_attn[0]


        current_input = residual + self.dropout(x_attn)

        if self.is_adapter_available():
            pack_input = {
                'x': current_input,
                'loc': 'mha',
                'att_mask': att_mask,
                'pos_emb': pos_emb,
            }
            pack_input = self.forward_enabled_adapters(pack_input)
            current_input = pack_input['x']

        # Convolution module
        if self.use_convolution:
            residual = current_input
            x_conv = self.norm_conv(current_input)
            x_conv = self.conv(x_conv, pad_mask=pad_mask, cache=cache_last_time)
            if cache_last_time is not None and isinstance(x_conv, tuple): # Conv returned (output, cache)
                x_conv, cache_last_time = x_conv
            elif isinstance(x_conv, tuple): # Conv returned a tuple but no cache was expected/returned
                x_conv = x_conv[0]
            current_input = residual + self.dropout(x_conv)
        # If no convolution, current_input passes through.

        # Second FFN module
        residual = current_input
        x_ffn2 = self.norm_feed_forward2(current_input)
        x_ffn2 = self.feed_forward2(x_ffn2)
        current_input = residual + self.dropout(x_ffn2) * self.fc_factor

        x = self.norm_out(current_input)

        if self.is_adapter_available():
            pack_input = {
                'x': x,
                'loc': 'post',
            }
            pack_input = self.forward_enabled_adapters(pack_input)
            x = pack_input['x']

        if self.is_access_enabled(getattr(self, "model_guid", None)) and self.access_cfg.get(
            'save_encoder_tensors', False
        ):
            self.register_accessible_tensor(name='encoder', tensor=x)

        if cache_last_channel is None and cache_last_time is None:
            return x
        else: # Need to return cache(s)
            if cache_last_channel is not None and cache_last_time is not None:
                return x, cache_last_channel, cache_last_time
            elif cache_last_channel is not None:
                return x, cache_last_channel
            else: # cache_last_time is not None
                return x, cache_last_time # This case might need specific handling based on expected return signature


class ConformerConvolution(nn.Module):
    """The convolution module for the Conformer model.
    Args:
        d_model (int): hidden dimension
        kernel_size (int): kernel size for depthwise convolution
        pointwise_activation (str): name of the activation function to be used for the pointwise conv.
            Note that Conformer uses a special key `glu_` which is treated as the original default from
            the paper.
        use_bias (bool): Use bias in all Linear and Conv1d layers improve activation flow and stabilize training of huge models.
            Defaults to True
    """

    def __init__(
        self,
        d_model,
        kernel_size,
        norm_type='batch_norm',
        conv_context_size=None,
        pointwise_activation='glu_',
        use_bias=True,
    ):
        super(ConformerConvolution, self).__init__()
        assert (kernel_size - 1) % 2 == 0
        self.d_model = d_model
        self.kernel_size = kernel_size
        self.norm_type = norm_type
        self.use_bias = use_bias

        if conv_context_size is None:
            conv_context_size = (kernel_size - 1) // 2

        if pointwise_activation in activation_registry:
            self.pointwise_activation = activation_registry[pointwise_activation]()
            dw_conv_input_dim = d_model * 2

            if hasattr(self.pointwise_activation, 'inplace'):
                self.pointwise_activation.inplace = True
        else:
            self.pointwise_activation = pointwise_activation
            dw_conv_input_dim = d_model

        self.pointwise_conv1 = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model * 2,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=self.use_bias,
        )

        self.depthwise_conv = CausalConv1D(
            in_channels=dw_conv_input_dim,
            out_channels=dw_conv_input_dim,
            kernel_size=kernel_size,
            stride=1,
            padding=conv_context_size,
            groups=dw_conv_input_dim,
            bias=self.use_bias,
        )

        if norm_type == 'batch_norm':
            self.batch_norm = nn.BatchNorm1d(dw_conv_input_dim)
        elif norm_type == 'instance_norm':
            self.batch_norm = nn.InstanceNorm1d(dw_conv_input_dim)
        elif norm_type == 'layer_norm':
            self.batch_norm = nn.LayerNorm(dw_conv_input_dim)
        elif norm_type == 'fused_batch_norm':
            self.batch_norm = FusedBatchNorm1d(dw_conv_input_dim)
        elif norm_type == 'rms_norm':
            self.batch_norm = RMSNorm(dw_conv_input_dim)
        elif norm_type.startswith('group_norm'):
            num_groups = int(norm_type.replace("group_norm", ""))
            self.batch_norm = nn.GroupNorm(num_groups=num_groups, num_channels=d_model)
        else:
            raise ValueError(f"conv_norm_type={norm_type} is not valid!")

        self.activation = Swish()
        self.pointwise_conv2 = nn.Conv1d(
            in_channels=dw_conv_input_dim,
            out_channels=d_model,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=self.use_bias,
        )

    def forward(self, x, pad_mask=None, cache=None):
        x = x.transpose(1, 2)
        x = self.pointwise_conv1(x)

        # Compute the activation function or use GLU for original Conformer
        if self.pointwise_activation == 'glu_':
            x = nn.functional.glu(x, dim=1)
        else:
            x = self.pointwise_activation(x)

        if pad_mask is not None:
            x = x.masked_fill(pad_mask.unsqueeze(1), 0.0)

        x = self.depthwise_conv(x, cache=cache)
        if cache is not None:
            x, cache = x

        if self.norm_type == "layer_norm" or self.norm_type == "rms_norm":
            x = x.transpose(1, 2)
            x = self.batch_norm(x)
            x = x.transpose(1, 2)
        else:
            x = self.batch_norm(x)

        x = self.activation(x)
        x = self.pointwise_conv2(x)
        x = x.transpose(1, 2)
        if cache is None:
            return x
        else:
            return x, cache

    def reset_parameters_conv(self):
        pw1_max = pw2_max = self.d_model**-0.5
        dw_max = self.kernel_size**-0.5

        with torch.no_grad():
            nn.init.uniform_(self.pointwise_conv1.weight, -pw1_max, pw1_max)
            nn.init.uniform_(self.pointwise_conv2.weight, -pw2_max, pw2_max)
            nn.init.uniform_(self.depthwise_conv.weight, -dw_max, dw_max)
            if self.use_bias:
                nn.init.uniform_(self.pointwise_conv1.bias, -pw1_max, pw1_max)
                nn.init.uniform_(self.pointwise_conv2.bias, -pw2_max, pw2_max)
                nn.init.uniform_(self.depthwise_conv.bias, -dw_max, dw_max)


class ConformerFeedForward(nn.Module):
    """
    feed-forward module of Conformer model.
    use_bias (bool): Apply bias to all Linear and Conv1d layers improve activation flow and stabilize training of huge models.
    """

    def __init__(self, d_model, d_ff, dropout, activation_name='swish', use_bias=True):
        super(ConformerFeedForward, self).__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.use_bias = use_bias
        self.activation_name = activation_name.lower()

        if self.activation_name == 'swiglu':
            self.linear1 = nn.Linear(d_model, d_ff * 2, bias=self.use_bias)  # Projects to 2x for SwiGLU
            self.activation = nn.SiLU()  # PyTorch's nn.SiLU is Swish with beta=1
            self.dropout = nn.Dropout(p=dropout)
            self.linear2 = nn.Linear(d_ff, d_model, bias=self.use_bias) # Takes d_ff after SwiGLU
        elif self.activation_name == 'swish':
            self.linear1 = nn.Linear(d_model, d_ff, bias=self.use_bias)
            self.activation = Swish()
            self.dropout = nn.Dropout(p=dropout)
            self.linear2 = nn.Linear(d_ff, d_model, bias=self.use_bias)
        else:
            raise ValueError(f"Unsupported activation: {self.activation_name}. Choose 'swish' or 'swiglu'.")

    def forward(self, x):
        if self.activation_name == 'swiglu':
            x = self.linear1(x)
            x_gate, x_val = x.chunk(2, dim=-1)
            x = self.activation(x_gate) * x_val  # This is the SwiGLU operation
            x = self.dropout(x)
            x = self.linear2(x)
        elif self.activation_name == 'swish':
            x = self.linear1(x)
            x = self.activation(x)
            x = self.dropout(x)
            x = self.linear2(x)
        return x

    def reset_parameters_ff(self):
        ffn1_max = self.d_model**-0.5
        ffn2_max = self.d_ff**-0.5
        with torch.no_grad():
            nn.init.uniform_(self.linear1.weight, -ffn1_max, ffn1_max)
            nn.init.uniform_(self.linear2.weight, -ffn2_max, ffn2_max)
            if self.use_bias:
                nn.init.uniform_(self.linear1.bias, -ffn1_max, ffn1_max)
                nn.init.uniform_(self.linear2.bias, -ffn2_max, ffn2_max)
