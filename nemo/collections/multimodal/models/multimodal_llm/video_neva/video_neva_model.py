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

import os
from functools import partial
from itertools import chain
from typing import Any, Optional

import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from omegaconf.dictconfig import DictConfig
from pytorch_lightning.trainer.trainer import Trainer
from transformers import CLIPVisionModel

from nemo.collections.multimodal.data.neva.conversation import DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN
from nemo.collections.multimodal.data.neva.neva_dataset import (
    DataCollatorForSupervisedDataset,
    make_supervised_data_module,
)
from nemo.collections.multimodal.models.vision_language_foundation.clip.megatron_clip_models import (
    CLIPVisionTransformer,
    MegatronCLIPModel,
)
from nemo.collections.multimodal.parts.utils import extend_instance, load_nemo_model_weights
from nemo.collections.nlp.data.language_modeling.megatron.data_samplers import MegatronPretrainingSampler
from nemo.collections.nlp.models.language_modeling.megatron.gpt_model import GPTModel
from nemo.collections.nlp.models.language_modeling.megatron_gpt_model import MegatronGPTModel, get_specs
from nemo.collections.nlp.models.nlp_model import NLPModel
from nemo.collections.nlp.modules.common.megatron.adapters.parallel_adapters import (
    AdapterName,
    MultimodalProjectorAdapterConfig,
)
from nemo.collections.nlp.modules.common.megatron.utils import average_losses_across_data_parallel_group
from nemo.collections.nlp.modules.common.text_generation_utils import (
    generate,
    get_computeprob_response,
    get_default_length_params,
    get_default_sampling_params,
    megatron_neva_generate,
)
from nemo.collections.nlp.modules.common.transformer.text_generation import LengthParam, OutputType, SamplingParam
from nemo.collections.nlp.parts.mixins.multimodal_adapter_mixins import MultimodalAdapterModelMixin
from nemo.collections.nlp.parts.utils_funcs import get_last_rank
from nemo.collections.vision.data.megatron.data_samplers import MegatronVisionPretrainingRandomSampler
from nemo.core import adapter_mixins
from nemo.core.classes.common import PretrainedModelInfo
from nemo.utils import logging

from nemo.collections.multimodal.models.multimodal_llm.neva.neva_model import FrozenCLIPVisionTransformer

try:
    import apex.transformer.pipeline_parallel.utils

    HAVE_APEX = True

except (ImportError, ModuleNotFoundError):

    HAVE_APEX = False

try:
    from megatron.core import InferenceParams, dist_checkpointing, parallel_state
    from megatron.core.models.gpt import GPTModel as MCoreGPTModel

    HAVE_MEGATRON_CORE = True

except (ImportError, ModuleNotFoundError):

    HAVE_MEGATRON_CORE = False

class NevaWordEmbeddingMixin(torch.nn.Module, adapter_mixins.AdapterModuleMixin):
    """
    A mixin class for integrating vision-based embeddings into language models.

    This class extends the functionality of a language model to include vision-based embeddings
    by integrating a vision encoder. It allows the language model to process media inputs
    alongside text inputs.
    """

    def init_vision(
        self,
        vision_encoder,
        media_start_id,
        media_end_id,
        vision_select_layer=-1,
        class_token_length=1,
        use_vid_start_end=False,
    ):
        self.vision_encoder = vision_encoder
        self.from_hf = isinstance(vision_encoder, CLIPVisionModel)
        self.media_start_id = media_start_id
        self.media_end_id = media_end_id
        self.class_token_length = class_token_length
        self.use_vid_start_end = use_vid_start_end
        self.vision_select_layer = vision_select_layer
        self.media = None
        self.set_accepted_adapter_types([MultimodalProjectorAdapterConfig._target_])

    def set_media(self, media):
        self.media = media

    def forward(self, input_ids, **kwargs):
        media = self.media  # avoid change the signature of embedding forward function
        words_embeddings = super().forward(input_ids, **kwargs)

        return self.replace_media_embeddings(input_ids, words_embeddings, media)

    def encode_vision_x(self, vision_x: torch.Tensor):
        """
        Compute media tokens from vision input by passing it through vision encoder and conditioning language model.
        Args:
            vision_x (torch.Tensor): Vision input
                shape (B, T_img, F, C, H, W)
                Images in the same chunk are collated along T_img, and frames are collated along F
                Currently only F=1 is supported (single-frame videos)

        rearrange code based on https://github.com/dhansmair/flamingo-mini
        """

        assert vision_x.ndim == 6, "vision_x should be of shape (b, T_img, F, C, H, W)"
        b, T, F = vision_x.shape[:3]
        assert F == 1, "Only single frame supported"

        vision_x = rearrange(vision_x, "b T F c h w -> (b T F) c h w")
        vision_x = vision_x.to(self.vision_encoder.dtype)
        with torch.no_grad():
            if self.from_hf:
                vision_x = self.vision_encoder(vision_x, output_hidden_states=True)
                vision_x = vision_x.hidden_states[self.vision_select_layer]
            else:
                self.vision_encoder.backbone.transformer.return_select_layer = self.vision_select_layer
                vision_x = self.vision_encoder(vision_x)
        vision_x = rearrange(vision_x, "(b T F) v d -> b T F v d", b=b, T=T, F=F)
        vision_x = vision_x[:, :, :, self.class_token_length :]
        assert self.is_adapter_available(), "Cannot find multimodal vision adapter!"
        vision_connector = self.get_adapter_module(AdapterName.MULTIMODAL_PROJECTOR_ADAPTER)
        vision_x = vision_connector(vision_x)
        return vision_x

    def replace_media_embeddings(self, input_ids, inputs_embeds, media):
        if media is None:
            return inputs_embeds

        batch_size, sequence_length, hidden_size = inputs_embeds.shape

        # calculate media features without gradients
        media_features = self.encode_vision_x(media)  # b T F S(eq) H(idden)
        num_images_per_sample = media_features.size(1)
        num_patches = media_features.size(3)
        # flatten patches
        media_features = media_features.view(batch_size, -1, hidden_size)

        # create an indices matrix used in torch.scatter
        padded_media_indices = torch.ones(
            (batch_size, num_images_per_sample), dtype=torch.long, device=input_ids.device
        )
        padded_media_indices *= sequence_length
        for idx, input_id in enumerate(input_ids):
            media_end_positions = torch.where(input_id == self.media_end_id)[0]
            if self.use_im_start_end:
                # locate the first media token positions
                padded_media_indices[idx, : len(media_end_positions)] = media_end_positions - num_patches
                assert (
                    input_id[padded_media_indices[idx, : len(media_end_positions)] - 1] == self.media_start_id
                ).all()
            else:
                padded_media_indices[idx, : len(media_end_positions)] = media_end_positions - num_patches + 1
                assert (input_id[padded_media_indices[idx, : len(media_end_positions)]] == self.media_start_id).all()

        # use indices to create a span
        padded_media_indices = padded_media_indices.unsqueeze(-1) + torch.arange(
            num_patches, device=padded_media_indices.device
        ).repeat(*padded_media_indices.shape, 1)
        padded_media_indices = padded_media_indices.reshape(batch_size, -1)
        padded_media_indices = repeat(padded_media_indices, 'b s -> b s h', h=hidden_size)

        # concat placeholder
        updated_input_embeds = torch.cat(
            (inputs_embeds, torch.zeros((batch_size, num_patches, hidden_size), device=inputs_embeds.device)), dim=1
        )
        updated_input_embeds = updated_input_embeds.type(media_features.dtype)
        # scatter media_features
        updated_input_embeds.scatter_(1, padded_media_indices, media_features)

        # chop off placeholder
        updated_input_embeds = updated_input_embeds[:, :sequence_length]

        return updated_input_embeds
