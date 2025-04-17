# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

# pylint: disable=C0115,C0116

from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from megatron.core import tensor_parallel
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer.transformer_config import TransformerConfig
from torch import Tensor

from nemo.collections.diffusion.models.dit.dit_model import DiTCrossAttentionModel
from nemo.collections.diffusion.models.dit_llama.dit_llama_layer_spec import get_dit_llama_spec


class TransposedLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(TransposedLinear, self).__init__()
        # Note the swapped dimensions
        self.weight = nn.Parameter(torch.Tensor(in_features, out_features))

    def forward(self, input):
        # Transpose the weight back in the forward pass
        return F.linear(input, self.weight.t())


class DiTLlamaModel(DiTCrossAttentionModel):
    def __init__(
        self,
        config: TransformerConfig,
        pre_process: bool = True,
        post_process: bool = True,
        fp16_lm_cross_entropy: bool = False,
        parallel_output: bool = True,
        position_embedding_type: Literal["learned_absolute", "rope"] = "rope",
        max_img_h: int = 80,
        max_img_w: int = 80,
        max_frames: int = 34,
        patch_spatial: int = 1,
        patch_temporal: int = 1,
        in_channels: int = 16,
        out_channels: int = 16,
        **kwargs,
    ):
        super().__init__(
            config=config,
            pre_process=pre_process,
            post_process=post_process,
            fp16_lm_cross_entropy=fp16_lm_cross_entropy,
            parallel_output=parallel_output,
            position_embedding_type=position_embedding_type,
            max_img_h=max_img_h,
            max_img_w=max_img_w,
            max_frames=max_frames,
            patch_spatial=patch_spatial,
            patch_temporal=patch_temporal,
            in_channels=in_channels,
            out_channels=out_channels,
            transformer_decoder_layer_spec=get_dit_llama_spec,
            **kwargs,
        )

    def forward(
        self,
        x: Tensor,
        timesteps: Tensor,
        crossattn_emb: Tensor,
        packed_seq_params: PackedSeqParams = None,
        pos_ids: Tensor = None,
        **kwargs,
    ) -> Tensor:
        """Forward pass.

        Args:
            x (Tensor): vae encoded videos (b s c)
            encoder_decoder_attn_mask (Tensor): cross-attention mask between encoder and decoder
            inference_params (InferenceParams): relevant arguments for inferencing

        Returns:
            Tensor: loss tensor
        """
        B = x.shape[0]
        fps = kwargs.get(
            'fps',
            torch.tensor(
                [
                    30,
                ]
                * B,
                dtype=torch.bfloat16,
            ),
        ).view(-1)
        pos_emb = None
        if self.pre_process:
            # transpose to match
            x_B_S_D = self.x_embedder(x)
            pos_emb = self.pos_embedder(pos_ids)
            pos_emb = rearrange(pos_emb, "B S D -> S B D")
            x_S_B_D = rearrange(x_B_S_D, "B S D -> S B D")
        else:
            # intermediate stage of pipeline
            x_S_B_D = None  ### should it take encoder_hidden_states

        timesteps_B_D = self.t_embedder(timesteps.flatten()).to(torch.bfloat16)  # (b d_text_embedding)

        affline_emb_B_D = timesteps_B_D
        fps_B_D = self.fps_embedder(fps)
        fps_B_D = nn.functional.pad(fps_B_D, (0, self.config.hidden_size - fps_B_D.shape[1]))
        affline_emb_B_D += fps_B_D

        crossattn_emb = rearrange(crossattn_emb, 'B S D -> S B D')

        if self.config.sequence_parallel:
            if self.pre_process:
                x_S_B_D = tensor_parallel.scatter_to_sequence_parallel_region(x_S_B_D)
            crossattn_emb = tensor_parallel.scatter_to_sequence_parallel_region(crossattn_emb)
            # `scatter_to_sequence_parallel_region` returns a view, which prevents
            # the original tensor from being garbage collected. Clone to facilitate GC.
            # Has a small runtime cost (~0.5%).
            if self.config.clone_scatter_output_in_embedding:
                if self.pre_process:
                    x_S_B_D = x_S_B_D.clone()
                crossattn_emb = crossattn_emb.clone()

        x_S_B_D = self.decoder(
            hidden_states=x_S_B_D,
            attention_mask=affline_emb_B_D,
            context=crossattn_emb,
            context_mask=None,
            rotary_pos_emb=pos_emb,
            packed_seq_params=packed_seq_params,
        )

        if not self.post_process:
            return x_S_B_D

        if self.config.sequence_parallel:
            x_S_B_D = tensor_parallel.gather_from_sequence_parallel_region(x_S_B_D)

        x_S_B_D = self.final_layer_linear(x_S_B_D)
        return rearrange(x_S_B_D, "S B D -> B S D")

    def set_input_tensor(self, input_tensor: Tensor) -> None:
        """Sets input tensor to the model.

        See megatron.model.transformer.set_input_tensor()

        Args:
            input_tensor (Tensor): Sets the input tensor for the model.
        """
        # This is usually handled in schedules.py but some inference code still
        # gives us non-lists or None
        if not isinstance(input_tensor, list):
            input_tensor = [input_tensor]

        assert len(input_tensor) == 1, 'input_tensor should only be length 1 for gpt/bert'
        self.decoder.set_input_tensor(input_tensor[0])
