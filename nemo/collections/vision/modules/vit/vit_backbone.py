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

"""Vision Transformer(VIT) model."""

import math
import einops
import torch
import torch.nn.functional as F
from functools import partial

from nemo.collections.vision.modules.common.megatron.transformer import ParallelVisionTransformer
from nemo.collections.nlp.modules.common.megatron.utils import (
    ApexGuardDefaults,
    get_linear_layer,
    init_method_normal,
    scaled_init_method_normal,
)
from nemo.collections.nlp.modules.common.megatron.module import MegatronModule

try:
    import apex
    from apex.transformer import tensor_parallel
    from apex.transformer.enums import AttnMaskType

    HAVE_APEX = True
except (ImportError, ModuleNotFoundError):
    HAVE_APEX = False

    # fake missing classes with None attributes
    AttnMaskType = ApexGuardDefaults()
    LayerType = ApexGuardDefaults()

CLASS_TOKEN_LENGTH = 8


class VitMlpHead(MegatronModule):
    """Pooler layer.

    Pool hidden states of a specific token (for example start of the
    sequence) and add a linear transformation followed by a tanh.

    Arguments:
        hidden_size: hidden size
        init_method: weight initialization method for the linear layer.
            bias is set to zero.
    """

    def __init__(self, hidden_size, num_classes):
        super(VitMlpHead, self).__init__()
        self.dense_in = torch.nn.Linear(hidden_size, hidden_size)
        self.relu = torch.nn.ReLU()
        self.dense_out = torch.nn.Linear(hidden_size, num_classes)
        torch.nn.init.constant_(self.dense_out.bias, -10)

    def forward(self, hidden_states):
        # hidden_states: [b, 1, h]
        # sequence_index: index of the token to pool.
        dense_in_result = self.dense_in(hidden_states)
        tanh_result = torch.tanh(dense_in_result)
        dense_out_result = self.dense_out(tanh_result)
        return dense_out_result


def isPerfectSquare(x):
    if (x >= 0):
        sr = math.sqrt(x)
        return (int(sr) * int(sr) == x)
    return False


def twod_interpolate_position_embeddings_hook(
        model_cfg,
        class_token_present,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
):
    num_patches_per_dim_h = model_cfg.img_h // model_cfg.patch_dim
    num_patches_per_dim_w = model_cfg.img_w // model_cfg.patch_dim
    num_patches = num_patches_per_dim_h * num_patches_per_dim_w
    hidden_size = model_cfg.hidden_size

    key = prefix + "weight"

    assert key in state_dict
    if key in state_dict:
        input_param = state_dict[key]

        input_seq_len = input_param.shape[0]
        assert (isPerfectSquare(input_seq_len) or isPerfectSquare(input_seq_len - CLASS_TOKEN_LENGTH))
        input_has_class_token = not isPerfectSquare(input_seq_len)
        num_tok_input = input_seq_len - CLASS_TOKEN_LENGTH if input_has_class_token else input_seq_len
        num_tok_output = num_patches
        output_has_class_token = class_token_present

        # update input_param and load it to state_dict[key]
        if input_has_class_token:
            input_param_tok = input_param[:CLASS_TOKEN_LENGTH, :]
            input_param_grid = input_param[CLASS_TOKEN_LENGTH:, :]
        else:
            input_param_tok = torch.zeros(CLASS_TOKEN_LENGTH, hidden_size)
            input_param_grid = input_param

        assert input_param.shape[1] == hidden_size

        if num_tok_input != num_tok_output:
            gs_input = int(math.sqrt(num_tok_input))
            gs_new = (num_patches_per_dim_h, num_patches_per_dim_w)

            input_param_grid = input_param_grid.transpose(0, 1).contiguous()
            input_param_grid = input_param_grid.reshape(
                (1, -1, gs_input, gs_input)
            )
            input_param_grid = input_param_grid.float()
            scale_factor = (gs_new[0] / gs_input, gs_new[1] / gs_input)

            input_param_grid = F.interpolate(
                input_param_grid, scale_factor=scale_factor, mode="bilinear"
            )

            input_param_grid = input_param_grid.half()
            input_param_grid = input_param_grid.reshape((-1, num_tok_output))
            input_param_grid = input_param_grid.transpose(0, 1).contiguous()

            assert input_param_grid.shape[1] == hidden_size

        input_param = input_param_grid
        assert (
                input_param.shape[0] == num_tok_output
                and input_param.shape[1] == hidden_size
        )

        if output_has_class_token:
            input_param = torch.cat((input_param_tok, input_param), dim=0)

        state_dict[key] = input_param


class VitBackbone(MegatronModule):
    """Vision Transformer Model."""

    def __init__(self,
                 model_cfg,
                 init_method=None,
                 scaled_init_method=None,
                 pre_process=True,
                 post_process=True,
                 class_token=True,
                 single_token_output=False):
        super(VitBackbone, self).__init__(share_token_embeddings=False)
        
        self.fp16_lm_cross_entropy = model_cfg.fp16_lm_cross_entropy
        if init_method is None:
            init_method = init_method_normal(init_method_std)
        if scaled_init_method is None:
            scaled_init_method = scaled_init_method_normal(init_method_std, num_layers)

        self.pre_process = pre_process
        self.post_process = post_process
        self.class_token = class_token
        self.hidden_size = model_cfg.hidden_size
        self.patch_dim = model_cfg.patch_dim
        self.img_h = model_cfg.img_h
        self.img_w = model_cfg.img_w
        self.micro_batch_size = model_cfg.micro_batch_size
        self.single_token_output = single_token_output
        self.drop_path_rate = drop_path_rate

        assert self.img_h % self.patch_dim == 0
        assert self.img_w % self.patch_dim == 0
        self.num_patches_per_dim_h = self.img_h // self.patch_dim
        self.num_patches_per_dim_w = self.img_w // self.patch_dim
        self.num_patches = self.num_patches_per_dim_h * self.num_patches_per_dim_w
        self.seq_length = self.num_patches + (CLASS_TOKEN_LENGTH if self.class_token else 0)
        self.flatten_dim = self.patch_dim * self.patch_dim * model_cfg.num_channels
        self.input_tensor = None
        self.position_ids = None

        if self.pre_process:
            # cls_token
            if self.class_token:
                self.cls_token = torch.nn.Parameter(
                    torch.randn(1, CLASS_TOKEN_LENGTH, self.hidden_size)
                )
                torch.nn.init.zeros_(self.cls_token)
            self.position_ids = torch.arange(self.seq_length).expand(1, -1).cuda()

            # Linear encoder
            self.linear_encoder = torch.nn.Linear(
                self.flatten_dim, self.hidden_size
            )

            # embedding
            self.position_embeddings = torch.nn.Embedding(
                self.seq_length, self.hidden_size
            )
            init_method_normal(model_cfg.init_method_std)(
                self.position_embeddings.weight
            )

            class_token_present = self.class_token
            self.position_embeddings._register_load_state_dict_pre_hook(
                partial(
                    twod_interpolate_position_embeddings_hook,
                    model_cfg,
                    class_token_present
                )
            )

            self.embedding_dropout = torch.nn.Dropout(model_cfg.hidden_dropout)

        self.transformer = ParallelVisionTransformer(
            init_method=init_method,
            output_layer_init_method=scaled_init_method,
            num_layers=model_cfg.num_layers,
            hidden_size=model_cfg.hidden_size,
            num_attention_heads=model_cfg.num_attention_heads,
            apply_query_key_layer_scaling=model_cfg.apply_query_key_layer_scaling,
            kv_channels=model_cfg.kv_channels,
            ffn_hidden_size=model_cfg.ffn_hidden_size,
            # self_attn_mask_type=self.encoder_attn_mask_type, # TODO (yuya)
            pre_process=self.pre_process,
            post_process=self.post_process,
            precision=model_cfg.precision,
            fp32_residual_connection=model_cfg.fp32_residual_connection,
            activations_checkpoint_method=model_cfg.activations_checkpoint_method,
            activations_checkpoint_num_layers=model_cfg.activations_checkpoint_num_layers,
            normalization=model_cfg.normalization,
            layernorm_epsilon=model_cfg.layernorm_epsilon,
            hidden_dropout=model_cfg.hidden_dropout,
            drop_path_rate=model_cfg.drop_path_rate,
            use_cpu_initialization=model_cfg.use_cpu_initialization,
            bias_activation_fusion=model_cfg.bias_gelu_fusion,
            persist_layer_norm=model_cfg.persist_layer_norm,
            openai_gelu=model_cfg.openai_gelu,
            onnx_safe=model_cfg.onnx_safe,
            masked_softmax_fusion=model_cfg.masked_softmax_fusion,
            megatron_legacy=model_cfg.megatron_legacy,
            sequence_parallel=model_cfg.sequence_parallel,
            activations_checkpoint_granularity=model_cfg.activations_checkpoint_granularity,
            gradient_accumulation_fusion=model_cfg.gradient_accumulation_fusion,
        )

    def set_input_tensor(self, input_tensor):
        """See megatron.model.transformer.set_input_tensor()"""
        self.transformer.set_input_tensor(input_tensor)

    def forward(self, input):

        if self.pre_process:
            rearranged_input = einops.rearrange(
                input,
                "b c (h p1) (w p2) -> b (h w) (p1 p2 c)",
                p1=self.patch_dim,
                p2=self.patch_dim,
            )

            # assert rearranged_input.dtype == torch.half # TODO (yuya)
            encoder_output = self.linear_encoder(rearranged_input)

            concatenated_tokens = encoder_output
            if self.class_token:
                cls_tokens = self.cls_token.expand(encoder_output.shape[0], -1, -1)
                concatenated_tokens = torch.cat((cls_tokens, encoder_output), dim=1)

            token_embeddings = concatenated_tokens + \
                               self.position_embeddings(self.position_ids[:, :concatenated_tokens.shape[1]])
            hidden_states = self.embedding_dropout(token_embeddings)
        else:
            hidden_states = input

        hidden_states = self.transformer(hidden_states, None)

        if self.single_token_output:
            hidden_states = hidden_states[:, 0, :]

        return hidden_states

