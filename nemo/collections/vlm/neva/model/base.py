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
import re
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Union

import lightning.pytorch as L
import torch
import torch.distributed
import torch.nn.functional as F
from megatron.core import dist_checkpointing
from megatron.core.enums import ModelType
from megatron.core.transformer.enums import AttnMaskType
from megatron.core import InferenceParams, tensor_parallel
from megatron.core.models.multimodal.llava_model import LLaVAModel as MCoreLLaVAModel
from megatron.core.models.vision.clip_vit_model import CLIPViTModel as MCoreCLIPViTModel
from megatron.core.models.vision.multimodal_projector import MultimodalProjector as MCoreMultimodalProjector
from megatron.core.optimizer import OptimizerConfig
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.parallel_state import get_context_parallel_group
from megatron.core.transformer.custom_layers.transformer_engine import (
    TEColumnParallelLinear,
    TENorm,
    TERowParallelLinear,
)
from megatron.core.transformer.mlp import MLP, MLPSubmodules
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_config import TransformerConfig
from torch import nn
from transformer_engine.pytorch.distributed import gather_along_first_dim
from transformers import CLIPVisionConfig, CLIPVisionModel

from nemo.collections.common.tokenizers.tokenizer_spec import TokenizerSpec
from nemo.collections.llm import fn
from nemo.collections.llm.gpt.model import transformer_engine_layer_spec
from nemo.collections.llm.gpt.model.base import get_batch_on_this_context_parallel_rank, get_packed_seq_params
from nemo.collections.vlm.neva.data.multimodal_tokens import IMAGE_TOKEN_INDEX, IGNORE_INDEX
from nemo.lightning import io
from nemo.lightning.io.pl import ckpt_to_weights_subdir
from nemo.lightning.megatron_parallel import MaskedTokenLossReductionWithLossMask
from nemo.lightning.pytorch.optim import MegatronOptimizerModule, OptimizerModule
from nemo.utils import logging

MODEL_CONFIG_ATTR = [
    'num_layers',
    'hidden_size',
    'num_attention_heads',
    'num_query_groups',
    'ffn_hidden_size',
    'kv_channels',
    'hidden_dropout',
    'attention_dropout',
    'fp32_residual_connection',
    'apply_residual_connection_post_layernorm',
    'layernorm_epsilon',
    'layernorm_zero_centered_gamma',
    'add_bias_linear',
    'add_qkv_bias',
    'gated_linear_unit',
    'activation_func',
    'activation_func_fp8_input_store',
    'num_moe_experts',
    'rotary_interleaved',
    'window_size',
    'normalization',
    'qk_layernorm',
    'test_mode',
    'calculate_per_token_loss',
    'seq_length',
]


def get_image_sequence_length(img_h, img_w, patch_dim, add_class_token, class_token_len):
    """Get image sequence length given image size, patch size, and class token."""
    num_patches_per_dim_h = img_h // patch_dim
    num_patches_per_dim_w = img_w // patch_dim
    num_patches = num_patches_per_dim_h * num_patches_per_dim_w
    return num_patches + (class_token_len if add_class_token else 0)

def calculate_model_parallel_padding(config, decoder_seq_len, text_only=False):
    from megatron.core import parallel_state as ps
    cp_size = ps.get_context_parallel_world_size()
    tp_size = ps.get_tensor_model_parallel_world_size()

    mp_padding_needed = 0
    sequence_parallel = getattr(config, "sequence_parallel", False)
    decoder_tp_comm_overlap = getattr(config, "tp_comm_overlap", False)
    decoder_seq_length = config.seq_length
    # TP Comm overlap is performed with combined text+image embeddings.
    # text_only flag skips using the full sequence length to calculate padding and uses
    # the provided decoder_seq_len
    if sequence_parallel and decoder_tp_comm_overlap and not text_only:
        # If TP Comm Overlap is enabled for combined text+image embedding in LM backbone,
        # user needs to provide decoder_seq_length with any potential padding needed for SP+CP
        assert decoder_seq_length is not None, \
            "Please provide --decoder-seq-length when using TP Comm overlap for LM backbone"
        mp_padding_needed = decoder_seq_length - decoder_seq_len
    elif sequence_parallel or cp_size > 1:
        if sequence_parallel and cp_size > 1:
            # Padding to multiple of tp_size * cp_size*2 when using sequence parallel and context parallel
            padding_factor = tp_size * cp_size * 2
        elif cp_size > 1:
            padding_factor = cp_size * 2
        elif sequence_parallel:
            padding_factor = tp_size
        mp_padding_needed = int((decoder_seq_len + padding_factor - 1) // (padding_factor) * (padding_factor)) - decoder_seq_len
        config.seq_length = decoder_seq_len + mp_padding_needed
    else:
        config.seq_length = decoder_seq_len

    return mp_padding_needed


def neva_data_step(dataloader_iter, config) -> Dict[str, torch.Tensor]:
    """neva data step prepare batch"""
    from megatron.core import parallel_state
    cp_size = parallel_state.get_context_parallel_world_size()

    def _get_packed_seq_params(tokens, img_seq_len, mp_padding_needed):
        batch_size = tokens.shape[0]
        # Calculate the valid token seq len that LM backbone should compute on
        combined_valid_seqlen = tokens.shape[1] + img_seq_len - mp_padding_needed
        cu_seqlens = torch.arange(
            0, (batch_size + 1) * (combined_valid_seqlen), step=(combined_valid_seqlen), dtype=torch.int32, device=tokens.device)
        # Calculate the total padded token seq len
        combined_padded_seqlen = tokens.shape[1] + img_seq_len
        cu_seqlens_padded = None
        qkv_format = 'sbhd'
        if cp_size > 1:
            # Provide cu_seqlens_<q/kv>_padded for CP support
            cu_seqlens_padded = torch.arange(
                0, (batch_size + 1) * (combined_padded_seqlen), step=(combined_padded_seqlen), dtype=torch.int32, device=tokens.device)
            # CP with padding mask type requires THD format
            qkv_format = 'thd'
        packed_seq_params = PackedSeqParams(
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_kv=cu_seqlens,
            cu_seqlens_q_padded=cu_seqlens_padded,
            cu_seqlens_kv_padded=cu_seqlens_padded,
            max_seqlen_q=combined_padded_seqlen,
            max_seqlen_kv=combined_padded_seqlen,
            qkv_format=qkv_format,
        )
        return packed_seq_params

    batch = next(dataloader_iter)
    _batch: dict
    if isinstance(batch, tuple) and len(batch) == 3:
        _batch = batch[0]
    else:
        _batch = batch

    packed_seq_params = None
    image_token_mask = None

    required_keys = set()
    required_keys.update(
        (
            "tokens",
            "attention_mask",
            "media",
            "num_media_tiles",
        )
    )
    if parallel_state.is_pipeline_first_stage():
        required_keys.update(("position_ids",))
    if parallel_state.is_pipeline_last_stage():
        required_keys.update(
            (
                "labels",
                "loss_mask",
            )
        )

    _batch = {
        key: val.cuda(non_blocking=True) if key in required_keys and val is not None else None
        for key, val in _batch.items()
    }

    if cp_size > 1:
        # Calculate the number of image embedding tokens will be added to text tokens
        num_image_embeddings_per_tile = config.vision_transformer_config.num_image_embeddings_per_tile
        # Pad to make sure the text sequence can be sharded equally by CP chunks.
        mp_padding_needed_for_text = calculate_model_parallel_padding(config, _batch["tokens"].shape[1], text_only=True)
        if mp_padding_needed_for_text > 0:
            _batch["tokens"], _batch["position_ids"], _batch["labels"], _batch["loss_mask"] = \
                [torch.nn.functional.pad(item, (0, mp_padding_needed_for_text)) for item in
                 (_batch["tokens"], _batch["position_ids"], _batch["labels"], _batch["loss_mask"])]
        # Image token mask must be supplied before distributed sequence to CP ranks.
        image_token_mask = _batch["tokens"] == IMAGE_TOKEN_INDEX
        num_images_per_sample = torch.sum(image_token_mask, dim=-1)
        img_seq_len = (num_image_embeddings_per_tile * num_images_per_sample - num_images_per_sample).max()
        packed_seq_params = _get_packed_seq_params(_batch["tokens"], img_seq_len, mp_padding_needed_for_text)

    # slice batch along sequence dimension for context parallelism
    output = get_batch_on_this_context_parallel_rank(
        {"tokens": _batch["tokens"], "position_ids": _batch["position_ids"]}
    )

    _batch.update(
        {**output, "image_token_mask": image_token_mask, "packed_seq_params": packed_seq_params}
    )

    return _batch


def neva_forward_step(model, batch) -> torch.Tensor:
    forward_args = {
        "images": batch["media"],
        "input_ids": batch["tokens"],
        "position_ids": batch["position_ids"],
        "attention_mask": batch.get("attention_mask", None),
        "loss_mask": batch.get("loss_mask", None),
        "labels": batch.get("labels", None),
        "num_image_tiles": batch.get("num_media_tiles", None),
        "image_token_mask": batch.get("image_token_mask", None),
        "packed_seq_params": batch.get("packed_seq_params", None),
    }

    if 'cu_seqlens' in batch:
        forward_args['packed_seq_params'] = get_packed_seq_params(batch)

    return model(**forward_args)


def set_input_tensor(self, tensor):
    pass


@dataclass
class MultimodalProjectorConfig(TransformerConfig, io.IOMixin):
    """
    For MLP, fc1 in shape of input_size, ffn_hidden_size, fc2 in shape of ffn_hidden_size, hidden_size
    """

    projector_type: str = "mlp2x_gelu"
    layer_spec: Optional[MLPSubmodules] = None
    input_size: Optional[int] = 1024
    hidden_size: int = 1024
    ffn_hidden_size: int = 1024
    activation_func: Callable = F.gelu
    bias: bool = True
    bias_activation_fusion: bool = True
    num_layers: int = 1  # placeholder, NOT used!
    num_attention_heads: int = 8  # placeholder, NOT used!

    def configure_model(self) -> "MCoreMultimodalProjector":
        if self.projector_type.startswith("mcore") and self.layer_spec is None:
            if self.projector_type == "mcore_mlp":
                self.projector_type = "mlp"  # strip "mcore_" for mcore init
                self.layer_spec = ModuleSpec(
                    module=MLP,
                    submodules=MLPSubmodules(
                        linear_fc1=TEColumnParallelLinear,
                        linear_fc2=TERowParallelLinear,
                    ),
                )
                self.layer_spec = self.layer_spec.submodules
            elif self.projector_type == "mcore_affine":
                self.projector_type = "affine"  # strip "mcore_" for mcore init
                self.layer_spec = MLPSubmodules(linear_fc1=TEColumnParallelLinear, linear_fc2=None)
            else:
                raise NotImplementedError(f"Not supported projector type `{self.projector_type}`")

            return MCoreMultimodalProjector(
                self,
                self.layer_spec,
                projector_type=self.projector_type,
                input_size=self.input_size,
            )

        # e.g. "mlp2x_gelu"
        mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu$', self.projector_type)
        if mlp_gelu_match:
            mlp_depth = int(mlp_gelu_match.group(1))
            modules = [torch.nn.Linear(self.input_size, self.hidden_size, bias=True)]
            for _ in range(1, mlp_depth):
                modules.append(torch.nn.GELU())
                modules.append(torch.nn.Linear(self.hidden_size, self.hidden_size, bias=True))
            model = torch.nn.Sequential(*modules)
            from types import MethodType

            model.set_input_tensor = MethodType(set_input_tensor, model)
        else:
            raise NotImplementedError(f"Not supported projector type `{self.projector_type}`")

        return model


@dataclass
class HFCLIPVisionConfig(CLIPVisionConfig, io.IOMixin):
    """
    https://github.com/huggingface/transformers/blob/v4.44.0/src/transformers/models/clip/configuration_clip.py#L261
    """

    hidden_size: int = 1024
    num_image_embeddings_per_tile: Optional[int] = None
    pretrained_model_name_or_path: Optional[Union[str, os.PathLike]] = None

    def __post_init__(self, *args, **kwargs) -> None:
        CLIPVisionConfig.__init__(self, *args, **kwargs, hidden_size=self.hidden_size)
        if self.pretrained_model_name_or_path is not None:
            config = CLIPVisionConfig.from_pretrained(self.pretrained_model_name_or_path)
            for key, value in config.to_dict().items():
                setattr(self, key, value)
        self.num_image_embeddings_per_tile = get_image_sequence_length(
            img_h=self.image_size,
            img_w=self.image_size,
            patch_dim=self.patch_size,
            add_class_token=False,
            class_token_len=1,
        )

    def configure_model(self) -> "CLIPVisionModel":
        # Monkey patch the method to the vision encoder
        CLIPVisionModel.set_input_tensor = set_input_tensor

        if self.pretrained_model_name_or_path is None:
            model = CLIPVisionModel(self)
        else:
            model = CLIPVisionModel.from_pretrained(self.pretrained_model_name_or_path)
        return model


@dataclass
class CLIPViTConfig(TransformerConfig, io.IOMixin):
    ln_pre_impl: Union[ModuleSpec, type] = TENorm
    ln_post_impl: Union[ModuleSpec, type] = TENorm
    add_class_token: bool = True
    class_token_len: int = 1
    patch_dim: int = 14
    img_h: int = 336
    img_w: int = 336
    vision_model_type: str = "clip"  # ["clip", "siglip"]
    num_image_embeddings_per_tile: Optional[int] = None
    transformer_layer_spec: ModuleSpec = transformer_engine_layer_spec

    num_layers: int = 1  # Placeholder, NOT used!
    num_attention_heads: int = 8  # Placeholder, NOT used!

    def __post_init__(self):
        if self.vision_model_type == "siglip":
            self.add_class_token = False
            self.class_token_len = 0
        self.num_image_embeddings_per_tile = get_image_sequence_length(
            img_h=self.img_h,
            img_w=self.img_w,
            patch_dim=self.patch_dim,
            add_class_token=self.add_class_token,
            class_token_len=self.class_token_len,
        )

    def configure_model(self) -> "CLIPViTModel":
        transformer_layer_spec = self.transformer_layer_spec
        if not isinstance(transformer_layer_spec, ModuleSpec):
            from nemo.collections.vlm.layer_specs import get_layer_spec_te

            transformer_layer_spec = get_layer_spec_te(is_vit=True)
        return CLIPViTModel(
            self,
            transformer_layer_spec,
            ln_pre_impl=self.ln_pre_impl,
            ln_post_impl=self.ln_post_impl,
            add_class_token=self.add_class_token,
            class_token_len=self.class_token_len,
            patch_dim=self.patch_dim,
            img_h=self.img_h,
            img_w=self.img_w,
            model_subtype=self.vision_model_type,
        )


@dataclass
class NevaConfig(TransformerConfig, io.IOMixin):
    language_transformer_config: Optional[TransformerConfig] = None
    vision_transformer_config: Optional[TransformerConfig] = None
    vision_projection_config: Optional[TransformerConfig] = None

    drop_vision_class_token: bool = True
    vision_feature_layer: int = -2

    encoder_pipeline_model_parallel_size: int = 0
    encoder_tensor_model_parallel_size: int = 1
    num_layers: int = 1  # Placeholder, NOT used!
    num_attention_heads: int = 8  # Placeholder, NOT used!

    seq_length: int = 1024

    language_model_from_pretrained: Optional[str] = None
    vision_model_from_pretrained: Optional[str] = None  # TODO
    vision_projection_from_pretrained: Optional[str] = None  # TODO

    freeze_language_model: bool = False
    freeze_vision_model: bool = False
    freeze_vision_projection: bool = False

    forward_step_fn: Callable = neva_forward_step
    data_step_fn: Callable = neva_data_step

    def __post_init__(self):
        if self.language_transformer_config is not None:
            for attr in MODEL_CONFIG_ATTR:
                setattr(self, attr, getattr(self.language_transformer_config, attr))

    def configure_model(self, tokenizer) -> "MCoreNevaModel":
        from megatron.core import parallel_state as ps

        self.language_transformer_config.tensor_model_parallel_size = self.tensor_model_parallel_size
        self.language_transformer_config.sequence_parallel = self.sequence_parallel
        self.vision_transformer_config.tensor_model_parallel_size = self.tensor_model_parallel_size
        self.vision_projection_config.tensor_model_parallel_size = self.tensor_model_parallel_size
        self.language_transformer_config.pipeline_model_parallel_size = self.pipeline_model_parallel_size
        self.language_transformer_config.context_parallel_size = self.context_parallel_size

        assert "NEVA `encoder_pipeline_model_parallel_size` has bug for now. Fix will come soon."
        if self.encoder_pipeline_model_parallel_size > 0:
            assert self.encoder_pipeline_model_parallel_size == 1, "ViT can only live on 1 pipeline stage."
            self.vision_transformer_config.pipeline_model_parallel_size = self.encoder_pipeline_model_parallel_size
            self.vision_projection_config.pipeline_model_parallel_size = self.encoder_pipeline_model_parallel_size
            self.language_transformer_config.encoder_pipeline_model_parallel_size = (
                self.encoder_pipeline_model_parallel_size
            )
            if self.encoder_tensor_model_parallel_size > 0:
                self.vision_transformer_config.tensor_model_parallel_size = self.encoder_tensor_model_parallel_size
                self.vision_projection_config.tensor_model_parallel_size = self.encoder_tensor_model_parallel_size

        model = MCoreNevaModel(
            config=self,
            tokenizer=tokenizer,
            pre_process=ps.is_pipeline_first_stage(),
            post_process=ps.is_pipeline_last_stage(),
            add_encoder=ps.is_pipeline_first_stage(),
            add_decoder=ps.is_pipeline_last_stage()
            or ps.get_pipeline_model_parallel_rank() >= self.encoder_pipeline_model_parallel_size,
            drop_vision_class_token=self.drop_vision_class_token,
        )

        return model


class CLIPViTModel(MCoreCLIPViTModel):
    """CLIP ViT vision model."""

    def forward(
        self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, num_unused_layers: int = 0
    ) -> torch.Tensor:
        if num_unused_layers > 0:
            unused_layers = self.decoder.layers[-num_unused_layers:]
            self.decoder.layers = self.decoder.layers[:-num_unused_layers]
            x = super().forward(x, attention_mask)
            self.decoder.layers.append(unused_layers)
            return x

        return super().forward(x, attention_mask)


class MCoreNevaModel(MCoreLLaVAModel):
    def __init__(
        self,
        config: NevaConfig,
        tokenizer: Optional = None,
        pre_process: bool = True,
        post_process: bool = True,
        add_encoder: bool = True,
        add_decoder: bool = True,
        drop_vision_class_token: bool = False,
    ) -> None:
        super(MCoreLLaVAModel, self).__init__(config=config)

        language_transformer_config = config.language_transformer_config
        vision_transformer_config = config.vision_transformer_config
        vision_projection_config = config.vision_projection_config

        self.pre_process = pre_process
        self.post_process = post_process
        self.add_encoder = add_encoder
        self.add_decoder = add_decoder

        self.encoder_hidden_state = None
        self.vision_model = None
        self.vision_projection = None
        self.language_model = None

        self.sequence_parallel_lm = language_transformer_config.sequence_parallel
        self.tp_comm_overlap_lm = language_transformer_config.tp_comm_overlap
        self.context_parallel_lm = language_transformer_config.context_parallel_size
        self.tensor_model_parallel_size_lm = language_transformer_config.tensor_model_parallel_size

        # This attribute is needed to check if an all-reduce is required
        # on the word embeddings inside `finalize_model_grads._allreduce_word_embedding_grads`.
        self.share_embeddings_and_output_weights = False
        if self.add_decoder:
            language_transformer_config.scatter_embedding_sequence_parallel = False
            self.language_model = language_transformer_config.configure_model(
                tokenizer=tokenizer, pre_process=pre_process, post_process=post_process
            )
            self.share_embeddings_and_output_weights = self.language_model.share_embeddings_and_output_weights
            self._language_max_sequence_length = self.language_model.max_sequence_length
            self._language_is_pipeline_parallel = language_transformer_config.pipeline_model_parallel_size > 1
            if config.language_model_from_pretrained is not None:
                sharded_state_dict = dict(state_dict=self.language_model.sharded_state_dict(prefix="module."))
                loaded_state_dict = dist_checkpointing.load(
                    sharded_state_dict=sharded_state_dict,
                    checkpoint_dir=ckpt_to_weights_subdir(config.language_model_from_pretrained, is_saving=False),
                    validate_access_integrity=False,
                )
                loaded_state_dict = {k.removeprefix("module."): v for k, v in loaded_state_dict["state_dict"].items()}
                self.language_model.load_state_dict(loaded_state_dict)
                logging.info(f"Restored language model weights from {config.language_model_from_pretrained}")
        else:
            if config.language_model_from_pretrained is not None:
                dist_checkpointing.load(
                    sharded_state_dict=dict(state_dict={}),
                    checkpoint_dir=config.language_model_from_pretrained,
                    validate_access_integrity=False,
                )

        if self.add_encoder:
            self.vision_model = vision_transformer_config.configure_model()
            self.vision_projection = vision_projection_config.configure_model()
            self._drop_vision_class_token = drop_vision_class_token

        self.freeze(
            freeze_language_model=config.freeze_language_model,
            freeze_vision_model=config.freeze_vision_model,
            freeze_vision_projection=config.freeze_vision_projection,
        )

        self.model_type = ModelType.encoder_or_decoder
        # This attribute is needed to check if an all-reduce is required
        # on the word embeddings inside `finalize_model_grads._allreduce_word_embedding_grads`.

        self.vision_model_from_hf = hasattr(vision_transformer_config, "image_size")
        self._img_seq_len = vision_transformer_config.num_image_embeddings_per_tile

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        loss_mask: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        inference_params: Optional[InferenceParams] = None,
        num_image_tiles: Optional[List[int]] = None,
        image_token_index: Optional[int] = IMAGE_TOKEN_INDEX,
        runtime_gather_output: Optional[bool] = None,
        image_token_mask: Optional[torch.Tensor] = None,
        packed_seq_params: Optional[PackedSeqParams] = None,
    ) -> torch.Tensor:
        """Forward function of the LLaVA model.

        Args:
            images (torch.Tensor): input image of shape [num_tiles, img_h, img_w]. num_tiles means the number of image tiles in this batch.
            input_ids (torch.Tensor): input text ids [batch, text_seq_len].
            position_ids (torch.Tensor): input text position ids [batch, text_seq_len].
            attention_mask (torch.Tensor): Attention mask for the language model [batch, 1, combined_seq_len, combined_seq_len].
            labels (torch.Tensor): Optional target text labels [batch, combined_seq_len].
            loss_mask (torch.Tensor): Text loss mask [batch, text_seq_len].
            inference_params (InferenceParams): Inference-time parameters including KV cache.
            num_image_tiles (list of int): Number of tiles per image. Default 1 tile per image.
            image_token_index (int): ID for input images. Default None means `image_token_index`
                arg in the constructor will be used.
            runtime_gather_output (bool): Gather output at runtime. Default None means
                `parallel_output` arg in the constructor will be used.
            image_token_mask (torch.Tensor): Tensor indicating the location of
                image token index in input_ids.
            packed_seq_params (PackedSeqParams): Dict with padded token information.
                Required for using SP/CP with padding mask type.

        Returns:
            output (torch.Tensor): Loss of shape [b, s] if labels are provided,
                otherwise logits of shape [b, s, vocab_size].
            loss_mask (torch.Tensor): Loss mask expanded to combined sequence length. Shape [b, s].
        """
        use_inference_kv_cache = (
            inference_params is not None
            and "image_tokens_count" in inference_params.key_value_memory_dict
        )
        has_images = images is not None and images.shape[0] > 0

        # If running inference, we can skip images token computation if they were computed already earlier for this sample.
        if use_inference_kv_cache:
            image_embeddings = None
        elif self.add_encoder and not has_images:
            vision_param = next(self.vision_model.parameters())
            # If no images provided, use an empty image embeddings tensor.
            image_embeddings = torch.tensor([], dtype=vision_param.dtype, device=vision_param.device).reshape(0, 0, 0)
        elif self.add_encoder and has_images:
            # images is in shape of (num_images_in_mbs, c, h, w)
            # note num_images_in_mbs is not mbs but total images in this mbs.
            images = images.to(next(self.vision_model.parameters()).dtype)
            if self.vision_model_from_hf:
                self.vision_model = self.vision_model.eval()
                image_embeddings = self.vision_model(images, output_hidden_states=True)
                image_embeddings = image_embeddings[-1][
                    self.config.vision_feature_layer
                ]  # [num_images, img_seq_len, h_vision]
            else:
                # TODO(yuya): MCore Clip path not yet support taking a specific layer hidden states
                image_embeddings = self.vision_model(images, num_unused_layers=-self.config.vision_feature_layer - 1)
            if self._drop_vision_class_token:
                class_token_len = getattr(self.vision_model, "class_token_len", 1)
                image_embeddings = image_embeddings[:, class_token_len:, :]

            # contiguous() required as `permute` can sparsify the tensor and this breaks pipelining
            image_embeddings = image_embeddings.permute(1, 0, 2).contiguous()  # [img_seq_len, num_tiles, h_vision]

            # map vision model output size to language model input size.
            image_embeddings = self.vision_projection(image_embeddings)  # [img_seq_len, num_tiles, h_language]

            # TODO: Support batched inference.
            # In inference, the language model KV cache will be updated for image token positions.
            # Store the image tokens sequence length to be used as an offset to the KV cache later.
            if inference_params is not None:
                inference_params.key_value_memory_dict["image_tokens_count"] = (
                    image_embeddings.shape[0] * image_embeddings.shape[1]
                )
        else:
            image_embeddings = self.encoder_hidden_state

        if not self.add_decoder:
            return image_embeddings

        language_embeddings = None
        if self.pre_process:
            input_ids_text = input_ids.clone()
            # MultiModal Token indices are assumed to be values
            input_ids_text[input_ids_text < 0] = 0
            # Note: This adds absolute position embedding but not RoPE.
            # Each image is counted as one position.
            # RoPE is added in language_model forward. Each image embedding is one position.
            language_embeddings = self.language_model.embedding(
                input_ids=input_ids_text, position_ids=position_ids
            )  # [text_seq_len, b, h_language]
            # Gather the language embeddings back. We need the full embedding to insert
            # image embeddings and then scatter again to avoid load imbalance.
            if self.context_parallel_lm > 1:
                cp_group = get_context_parallel_group()
                language_embeddings, _ = gather_along_first_dim(language_embeddings, cp_group)

            language_embeddings = language_embeddings.transpose(
                1, 0
            ).contiguous()  # [b, text_seq_len, h_language]

        # Assume 1 tile per image if the number of tiles is not provided.
        if num_image_tiles is None:
            num_image_tiles = torch.ones(images.shape[0], dtype=torch.int, device=input_ids.device)
        elif isinstance(num_image_tiles, list):
            num_image_tiles = torch.tensor(num_image_tiles, dtype=torch.int, device=input_ids.device)

        # Preprocess input, labels and loss mask.
        combined_embeddings, new_labels, new_loss_mask = self._preprocess_data(
            image_embeddings,
            language_embeddings,
            input_ids,
            loss_mask,
            labels,
            use_inference_kv_cache,
            inference_params,
            image_token_index,
            num_image_tiles,
            image_token_mask,
        )  # [combined_seq_len, b, h_language], [b, combined_seq_len], [b, combined_seq_len]

        if self.context_parallel_lm > 1 or self.sequence_parallel_lm:
            combined_embeddings, new_labels, new_loss_mask, packed_seq_params = (
                self._process_embedding_token_parallel(
                    combined_embeddings, new_labels, new_loss_mask, packed_seq_params
                )
            )

        output = self.language_model(
            input_ids=None,
            position_ids=None,
            attention_mask=attention_mask,
            decoder_input=combined_embeddings,
            labels=new_labels,
            inference_params=inference_params,
            runtime_gather_output=runtime_gather_output,
            packed_seq_params=packed_seq_params,
        )

        if labels is None or loss_mask is None:
            return output

        return output, new_loss_mask.contiguous()

    def set_input_tensor(self, input_tensor) -> None:
        """Set model chunk input tensor."""
        # This is usually handled in schedules.py but some inference code still
        # gives us non-lists or None
        if not isinstance(input_tensor, list):
            input_tensor = [input_tensor]
        assert len(input_tensor) == 1, 'input_tensor should only be length 1 for llava'

        if self.add_encoder and self.add_decoder:
            self.vision_model.set_input_tensor(input_tensor[0])
        elif self.add_encoder:
            self.vision_model.set_input_tensor(input_tensor[0])
        elif self.pre_process:
            self.encoder_hidden_state = input_tensor[0]
        else:
            self.language_model.set_input_tensor(input_tensor[0])


class NevaModel(L.LightningModule, io.IOMixin, io.ConnectorMixin, fn.FNMixin):
    def __init__(
        self,
        config: NevaConfig,
        optim: Optional[OptimizerModule] = None,
        tokenizer: Optional["TokenizerSpec"] = None,
        model_transform: Optional[Callable[[nn.Module], nn.Module]] = None,
    ):
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer
        self.optim = optim or MegatronOptimizerModule(config=OptimizerConfig(lr=1e-4, use_distributed_optimizer=True))
        self.optim.connect(self)  # This will bind the `configure_optimizers` method
        self.model_transform = model_transform
        self._training_loss_reduction = None
        self._validation_loss_reduction = None

    def configure_model(self) -> None:
        if not hasattr(self, "module"):
            self.module = self.config.configure_model(self.tokenizer)

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        loss_mask: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        inference_params: Optional[InferenceParams] = None,
        num_image_tiles: Optional[List[int]] = None,
        image_token_index: Optional[int] = IMAGE_TOKEN_INDEX,
        runtime_gather_output: Optional[bool] = None,
        image_token_mask: Optional[torch.Tensor] = None,
        packed_seq_params: Optional[PackedSeqParams] = None,
    ) -> torch.Tensor:
        output_tensor = self.module(
            images=images,
            input_ids=input_ids,
            position_ids=position_ids,
            loss_mask=loss_mask,
            attention_mask=attention_mask,
            labels=labels,
            inference_params=inference_params,
            num_image_tiles=num_image_tiles,
            image_token_index=image_token_index,
            runtime_gather_output=runtime_gather_output,
            image_token_mask=image_token_mask,
            packed_seq_params=packed_seq_params,
        )

        return output_tensor

    def data_step(self, dataloader_iter) -> Dict[str, torch.Tensor]:
        return self.config.data_step_fn(dataloader_iter, config=self.config)

    def forward_step(self, batch) -> torch.Tensor:
        return self.config.forward_step_fn(self, batch)

    def training_step(self, batch, batch_idx=None) -> torch.Tensor:
        # In mcore the loss-function is part of the forward-pass (when labels are provided)
        return self.forward_step(batch)

    def validation_step(self, batch, batch_idx=None) -> torch.Tensor:
        # In mcore the loss-function is part of the forward-pass (when labels are provided)

        return self.forward_step(batch)

    @property
    def training_loss_reduction(self) -> MaskedTokenLossReductionWithLossMask:
        if not self._training_loss_reduction:
            self._training_loss_reduction = MaskedTokenLossReductionWithLossMask()

        return self._training_loss_reduction

    @property
    def validation_loss_reduction(self) -> MaskedTokenLossReductionWithLossMask:
        if not self._validation_loss_reduction:
            self._validation_loss_reduction = MaskedTokenLossReductionWithLossMask(validation_step=True)

        return self._validation_loss_reduction


__all__ = [
    "NevaModel",
    "NevaConfig",
    "neva_data_step",
    "neva_forward_step",
]
