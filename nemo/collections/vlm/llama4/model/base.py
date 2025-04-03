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

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import lightning.pytorch as L
import torch
import torch.distributed
from megatron.core import parallel_state as ps
from megatron.core.inference_params import InferenceParams
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer.transformer_config import TransformerConfig
from torch import nn

from nemo.collections.common.tokenizers.tokenizer_spec import TokenizerSpec
from nemo.collections.llm import fn
from nemo.collections.vlm.neva.data.multimodal_tokens import IMAGE_TOKEN_INDEX
from nemo.collections.vlm.neva.model.base import MCoreNevaModel, NevaConfig, NevaModel, MODEL_CONFIG_ATTR
from nemo.lightning import io
from nemo.lightning.pytorch.optim import OptimizerModule


def llama4_data_step(dataloader_iter) -> Dict[str, torch.Tensor]:
    """Llama Omni Data Step"""
    from megatron.core import parallel_state

    # Based on: https://github.com/NVIDIA/Megatron-LM/blob/main/pretrain_gpt.py#L87
    # https://github.com/NVIDIA/NeMo/blob/main/nemo/collections/nlp/models/language_modeling/megatron_gpt_model.py#L828-L842
    batch = next(dataloader_iter)
    _batch: dict
    if isinstance(batch, tuple) and len(batch) == 3:
        _batch = batch[0]
    else:
        _batch = batch

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

    packed_seq_params = _batch.get("packed_seq_params", None)
    _batch = {
        key: val.cuda(non_blocking=True) if key in required_keys and val is not None else None
        for key, val in _batch.items()
    }
    if packed_seq_params is not None:
        for attr in ["cu_seqlens_q", "cu_seqlens_kv", "cu_seqlens_q_padded", "cu_seqlens_kv_padded"]:
            value = getattr(packed_seq_params, attr, None)
            if value is not None:
                setattr(packed_seq_params, attr, value.cuda(non_blocking=True))
    _batch["packed_seq_params"] = packed_seq_params
    if ps.get_context_parallel_world_size() > 1:
        num_valid_tokens_in_ub = None
        if "loss_mask" in _batch and _batch["loss_mask"] is not None:
            num_valid_tokens_in_ub = _batch["loss_mask"].sum()
        _batch["num_valid_tokens_in_ub"] = num_valid_tokens_in_ub

    return _batch


def llama4_forward_step(model, batch) -> torch.Tensor:
    """Llama Omni Forward Step"""
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

    return model(**forward_args)


@dataclass
class Llama4Config(NevaConfig):
    """Llama4 Model Base Config"""

    language_transformer_config: Optional[TransformerConfig] = None
    vision_transformer_config: Optional[TransformerConfig] = None
    vision_projection_config: Optional[TransformerConfig] = None

    drop_vision_class_token: bool = True

    encoder_pipeline_model_parallel_size: int = 0
    encoder_tensor_model_parallel_size: int = 1

    num_layers: int = 1  # Placeholder, NOT used!
    num_attention_heads: int = 8  # Placeholder, NOT used!
    seq_length: int = 8192

    language_model_from_pretrained: Optional[str] = None
    vision_model_from_pretrained: Optional[str] = None
    vision_projection_from_pretrained: Optional[str] = None  # TODO

    freeze_language_model: bool = False
    freeze_vision_model: bool = False
    freeze_vision_projection: bool = False

    forward_step_fn: Callable = llama4_forward_step
    data_step_fn: Callable = llama4_data_step


    def __post_init__(self):
        # pylint: disable=C0115,C0116
        if self.language_transformer_config is not None:
            for attr in MODEL_CONFIG_ATTR:
                setattr(self, attr, getattr(self.language_transformer_config, attr))

    def configure_model(self, tokenizer) -> "MCoreNevaModel":
        # pylint: disable=C0115,C0116
        self.language_transformer_config.scatter_embedding_sequence_parallel = False
        self.language_transformer_config.tensor_model_parallel_size = self.tensor_model_parallel_size
        self.language_transformer_config.sequence_parallel = self.sequence_parallel
        self.vision_transformer_config.tensor_model_parallel_size = self.tensor_model_parallel_size
        self.vision_projection_config.tensor_model_parallel_size = self.tensor_model_parallel_size
        self.language_transformer_config.pipeline_model_parallel_size = self.pipeline_model_parallel_size
        self.language_transformer_config.context_parallel_size = self.context_parallel_size
        self.language_transformer_config.expert_tensor_parallel_size = self.expert_tensor_parallel_size
        self.language_transformer_config.expert_model_parallel_size = self.expert_model_parallel_size

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

        model = Llama4BaseModel(
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


class Llama4BaseModel(MCoreNevaModel):
    """Mllama base model combining vision and text models with cross-attention."""

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
        # pylint: disable=C0301
        """Forward function of the Llama4 model.

        Args:
            images (torch.Tensor): input image of shape [num_tiles, img_h, img_w]. num_tiles means the number of
            image tiles in this batch.
            input_ids (torch.Tensor): input text ids [batch, text_seq_len].
            position_ids (torch.Tensor): input text position ids [batch, text_seq_len].
            attention_mask (torch.Tensor): Attention mask for the language model [batch, 1, combined_seq_len,
            combined_seq_len].
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
                inference_params is not None and "image_tokens_count" in inference_params.key_value_memory_dict
        )
        has_images = images is not None and len(images) > 0

        # If running inference, we can skip images token computation if they were computed already earlier
        # for this sample.
        if use_inference_kv_cache:
            image_embeddings = None
        elif self.add_encoder and not has_images:
            vision_param = next(self.vision_model.parameters())
            # If no images provided, use an empty image embeddings tensor.
            image_embeddings = torch.tensor([], dtype=vision_param.dtype, device=vision_param.device).reshape(0, 0, 0)
        elif self.add_encoder and has_images:
            # images is in shape of (num_images_in_mbs, c, h, w)
            # note num_images_in_mbs is not mbs but total images in this mbs.
            image_mask = (input_ids == self.tokenizer.token_to_id("<|patch|>")).unsqueeze(-1)
            h_ref = torch.zeros_like(input_ids, dtype=next(self.vision_model.parameters()).dtype, device=input_ids.device).unsqueeze(-1)
            image_embeddings = self.vision_model(
                image_batch=images,
                image_mask=image_mask,
                h_ref=h_ref,
            )  # [num_tiles, img_seq_len, h_vision]

            # map vision model output size to language model input size.
            image_embeddings = self.vision_projection(image_embeddings)  # [num_tiles, img_seq_len, h_language]

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

            language_embeddings = language_embeddings.transpose(1, 0).contiguous()  # [b, text_seq_len, h_language]

            # Preprocess input, labels and loss mask.
            if has_images:
                combined_embeddings = language_embeddings * ~image_mask + image_embeddings * image_mask
            else:
                combined_embeddings = language_embeddings
            if not (packed_seq_params is not None and packed_seq_params.qkv_format == "thd"):
                combined_embeddings = combined_embeddings.transpose(0, 1).contiguous() # [combined_seq_len, b, h_language]
        else:
            combined_embeddings = language_embeddings

        final_labels, final_loss_mask, final_attention_mask =(
            labels, loss_mask, attention_mask,
        )

        if self.context_parallel_lm > 1 or self.sequence_parallel_lm:
            combined_embeddings, final_labels, final_loss_mask, packed_seq_params = (
                self._process_embedding_token_parallel(
                    combined_embeddings, final_labels, final_loss_mask, packed_seq_params
                )
            )

        output = self.language_model(
            input_ids=None,
            position_ids=None,
            attention_mask=final_attention_mask,
            decoder_input=combined_embeddings,
            labels=final_labels,
            inference_params=inference_params,
            runtime_gather_output=runtime_gather_output,
            packed_seq_params=packed_seq_params,
        )

        if labels is None or loss_mask is None:
            return output

        return output, final_loss_mask.contiguous()


class Llama4Model(NevaModel):
    """Lightning Module for the MLlama model."""

    def __init__(
            self,
            config: Llama4Config,
            optim: Optional[OptimizerModule] = None,
            tokenizer: Optional["TokenizerSpec"] = None,
            model_transform: Optional[Callable[[nn.Module], nn.Module]] = None,
    ):
        super().__init__(
            config=config,
            optim=optim,
            tokenizer=tokenizer,
            model_transform=model_transform,
        )
