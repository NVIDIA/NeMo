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
from megatron.core import InferenceParams, dist_checkpointing
from megatron.core import parallel_state as ps
from megatron.core import tensor_parallel
from megatron.core.enums import ModelType
from megatron.core.models.multimodal.llava_model import LLaVAModel as MCoreLLaVAModel
from megatron.core.optimizer import OptimizerConfig
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.tensor_parallel import gather_from_sequence_parallel_region
from megatron.core.transformer.transformer_config import TransformerConfig
from torch import nn

from nemo.collections.common.tokenizers.tokenizer_spec import TokenizerSpec
from nemo.collections.llm import fn

from nemo.collections.avlm.data.multimodal_tokens import IGNORE_INDEX, IMAGE_TOKEN_INDEX, AUDIO_TOKEN_INDEX
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

def restore_model_weights(model, checkpoint_path, strict=False):
    """
    Restores model weights from a checkpoint.

    Args:
        model: The model to restore weights for.
        checkpoint_path: Path to the checkpoint.
        strict: Whether to restore weights even if they are not the same.
    """
    if checkpoint_path is not None:
        sharded_state_dict = dict(state_dict=model.sharded_state_dict(prefix="module."))
        loaded_state_dict = dist_checkpointing.load(
            sharded_state_dict=sharded_state_dict,
            checkpoint_dir=ckpt_to_weights_subdir(checkpoint_path, is_saving=False),
            validate_access_integrity=False,
            **({"strict": "log_all"} if not strict else {}),
        )
        loaded_state_dict = {k.removeprefix("module."): v for k, v in loaded_state_dict["state_dict"].items()}
        model.load_state_dict(loaded_state_dict, strict=strict)


def avlm_data_step(dataloader_iter) -> Dict[str, torch.Tensor]:
    """AVLM Data Step"""
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
            "images",
            "num_image_tiles",
            "image_sizes",
            "audios",
            "audio_lengths",
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


def avlm_forward_step(model, batch) -> torch.Tensor:
    """AVLM Forward Step"""
    forward_args = {
        "input_ids": batch["tokens"],
        "position_ids": batch["position_ids"],
        "loss_mask": batch.get("loss_mask", None),
        "attention_mask": batch.get("attention_mask", None),
        "labels": batch.get("labels", None),
        "images": batch["images"],
        "num_image_tiles": batch.get("num_image_tiles", None),
        # "image_sizes": batch.get("image_sizes", None),
        # "image_token_mask": batch.get("image_token_mask", None),
        "audios": batch["audios"],
        "audio_lengths": batch.get("audio_lengths", None),
        "packed_seq_params": batch.get("packed_seq_params", None),
    }

    return model(**forward_args)


@dataclass
class AVLMConfig(TransformerConfig, io.IOMixin):
    """AVLM Model Base Config"""

    language_transformer_config: Optional[TransformerConfig] = None
    vision_transformer_config: Optional[TransformerConfig] = None
    vision_projection_config: Optional[TransformerConfig] = None
    audio_transformer_config: Optional[TransformerConfig] = None
    audio_projection_config: Optional[TransformerConfig] = None

    drop_vision_class_token: bool = True
    vision_feature_layer: int = -2
    audio_feature_layer: int = -2

    encoder_pipeline_model_parallel_size: int = 0
    encoder_tensor_model_parallel_size: int = 1
    num_layers: int = 1  # Placeholder, NOT used!
    num_attention_heads: int = 8  # Placeholder, NOT used!

    seq_length: int = 1024

    language_model_from_pretrained: Optional[str] = None
    vision_model_from_pretrained: Optional[str] = None
    vision_projection_from_pretrained: Optional[str] = None  # TODO
    audio_model_from_pretrained: Optional[str] = None
    audio_projection_from_pretrained: Optional[str] = None  # TODO

    freeze_language_model: bool = False
    freeze_vision_model: bool = False
    freeze_vision_projection: bool = False
    freeze_audio_model: bool = False
    freeze_audio_projection: bool = False

    forward_step_fn: Callable = avlm_forward_step
    data_step_fn: Callable = avlm_data_step

    def __post_init__(self):
        # pylint: disable=C0115,C0116
        if self.language_transformer_config is not None:
            for attr in MODEL_CONFIG_ATTR:
                setattr(self, attr, getattr(self.language_transformer_config, attr))

    def configure_model(self, tokenizer) -> "MCoreAVLMModel":
        # pylint: disable=C0115,C0116
        self.language_transformer_config.tensor_model_parallel_size = self.tensor_model_parallel_size
        self.language_transformer_config.sequence_parallel = self.sequence_parallel
        if self.vision_transformer_config is not None:
            self.vision_transformer_config.tensor_model_parallel_size = self.tensor_model_parallel_size
            self.vision_projection_config.tensor_model_parallel_size = self.tensor_model_parallel_size
        if self.audio_transformer_config is not None:
            self.audio_transformer_config.tensor_model_parallel_size = self.tensor_model_parallel_size
            self.audio_projection_config.tensor_model_parallel_size = self.tensor_model_parallel_size
        self.language_transformer_config.pipeline_model_parallel_size = self.pipeline_model_parallel_size
        self.language_transformer_config.context_parallel_size = self.context_parallel_size

        assert "AVLM `encoder_pipeline_model_parallel_size` has bug for now. Fix will come soon."
        if self.encoder_pipeline_model_parallel_size > 0:
            assert self.encoder_pipeline_model_parallel_size == 1, "ViT can only live on 1 pipeline stage."
            if self.vision_transformer_config is not None:
                self.vision_transformer_config.pipeline_model_parallel_size = self.encoder_pipeline_model_parallel_size
                self.vision_projection_config.pipeline_model_parallel_size = self.encoder_pipeline_model_parallel_size
            if self.audio_transformer_config is not None:
                self.audio_transformer_config.pipeline_model_parallel_size = self.encoder_pipeline_model_parallel_size
                self.audio_projection_config.pipeline_model_parallel_size = self.encoder_pipeline_model_parallel_size
            self.language_transformer_config.encoder_pipeline_model_parallel_size = (
                self.encoder_pipeline_model_parallel_size
            )
            if self.encoder_tensor_model_parallel_size > 0:
                if self.vision_transformer_config is not None:
                    self.vision_transformer_config.tensor_model_parallel_size = self.encoder_tensor_model_parallel_size
                    self.vision_projection_config.tensor_model_parallel_size = self.encoder_tensor_model_parallel_size
                if self.audio_transformer_config is not None:
                    self.audio_transformer_config.tensor_model_parallel_size = self.encoder_tensor_model_parallel_size
                    self.audio_projection_config.tensor_model_parallel_size = self.encoder_tensor_model_parallel_size

        model = MCoreAVLMModel(
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


class _get_data_on_this_cp_rank(torch.autograd.Function):
    """Performs sharding for Context Parallelism in THD format

    In the forward pass, indices are selected for each CP rank and remaining tokens are dropped.
    In the backward pass, this class takes care of managing gradients for dropped tokens on each
    CP rank.
    """

    @staticmethod
    # def forward(ctx, decoder_embeddings, labels, loss_mask, packed_seq_params):
    def forward(ctx, batch, packed_seq_params):
        # pylint: disable=C0115,C0116
        cp_size = ps.get_context_parallel_world_size()
        if cp_size > 1:
            try:
                import transformer_engine_torch as tex
            except ModuleNotFoundError as e:
                logging.error(
                    "Please update Transformer Engine to >= 1.10 to use \
                        Context Parallel with THD format data"
                )
                raise e
            cp_rank = ps.get_context_parallel_rank()
            for key, data in batch.items():
                index = tex.thd_get_partitioned_indices(
                    packed_seq_params.cu_seqlens_q_padded, data.size(1), cp_size, cp_rank
                )
                if key == "combined_embeddings":
                    ctx.decoder_emb_index = index
                    ctx.decoder_emb_seqlen = data.size(1)
                batch[key] = data.index_select(1, index)
                batch[key].requires_grad = data.requires_grad

        return batch

    @staticmethod
    def backward(ctx, grad_out, grad_label, grad_loss):
        # pylint: disable=C0115,C0116
        seqlen = ctx.decoder_emb_seqlen
        index = ctx.decoder_emb_index
        assert grad_out.size(1) == index.size(
            0
        ), f"Shape mismatch in incoming gradient {grad_out.shape} and \
                index from THD CP sharding {index.shape}"
        grad_in = torch.zeros(
            grad_out.size(0),
            seqlen,
            *grad_out.size()[2:],
            dtype=grad_out.dtype,
            device=grad_out.device,
        )
        grad_in[:, ctx.decoder_emb_index, :] = grad_out

        return (grad_in, None, None, None)


class MCoreAVLMModel(MCoreLLaVAModel):
    """AVLM Model Base Model Class"""

    def __init__(
        self,
        config: AVLMConfig,
        tokenizer: Optional = None,
        pre_process: bool = True,
        post_process: bool = True,
        add_encoder: bool = True,
        add_decoder: bool = True,
        drop_vision_class_token: bool = True,
    ) -> None:
        # pylint: disable=C0115,C0116
        super(MCoreLLaVAModel, self).__init__(config=config)

        language_transformer_config = config.language_transformer_config
        vision_transformer_config = config.vision_transformer_config
        vision_projection_config = config.vision_projection_config
        audio_transformer_config = config.audio_transformer_config
        audio_projection_config = config.audio_projection_config

        self.pre_process = pre_process
        self.post_process = post_process
        self.add_encoder = add_encoder
        self.add_decoder = add_decoder

        self.encoder_hidden_state = None
        self.vision_model = None
        self.vision_projection = None
        self.audio_model = None
        self.audio_projection = None
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
            restore_model_weights(self.language_model, config.language_model_from_pretrained)
            logging.info(f"Restored language model weights from {config.language_model_from_pretrained}")

        else:
            if config.language_model_from_pretrained is not None:
                dist_checkpointing.load(
                    sharded_state_dict=dict(state_dict={}),
                    checkpoint_dir=config.language_model_from_pretrained,
                    validate_access_integrity=False,
                )

        if self.add_encoder:
            if vision_transformer_config is not None:
                self.vision_model = vision_transformer_config.configure_model()
                self.vision_projection = vision_projection_config.configure_model()
                self._drop_vision_class_token = drop_vision_class_token
                restore_model_weights(self.vision_model, config.vision_model_from_pretrained)
                logging.info(f"Restored vision model weights from {config.vision_model_from_pretrained}")
            if audio_transformer_config is not None:
                self.audio_model = audio_transformer_config.configure_model()
                self.audio_projection = audio_projection_config.configure_model()
                restore_model_weights(self.audio_model, config.audio_model_from_pretrained)
                logging.info(f"Restored audio model weights from {config.audio_model_from_pretrained}")

        # # DEBUGGING
        # print(stop_here)

        self.freeze(
            freeze_language_model=config.freeze_language_model,
            freeze_vision_model=config.freeze_vision_model,
            freeze_vision_projection=config.freeze_vision_projection,
            freeze_audio_model=config.freeze_audio_model,
            freeze_audio_projection=config.freeze_audio_projection,
        )

        self.model_type = ModelType.encoder_or_decoder
        # This attribute is needed to check if an all-reduce is required
        # on the word embeddings inside `finalize_model_grads._allreduce_word_embedding_grads`.

        # TODO - Modality specific attributes
        self.vision_model_from_hf = hasattr(vision_transformer_config, "image_size")
        self._img_seq_len = vision_transformer_config.num_image_embeddings_per_tile
        if drop_vision_class_token and vision_transformer_config.add_class_token:
            self._img_seq_len -= vision_transformer_config.class_token_len

        # # DEBUGGING
        # if torch.distributed.get_rank() == 0:
        #     def count_parameters(model):
        #         if model is None:
        #             return 0
        #         return sum(p.numel() for p in model.parameters())
        #     print("Number of parameters:")
        #     print(f"Vision model: {count_parameters(self.vision_model):,}")
        #     print(f"Vision projection: {count_parameters(self.vision_projection):,}")
        #     print(f"Audio model: {count_parameters(self.audio_model):,}")
        #     print(f"Audio projection: {count_parameters(self.audio_projection):,}")
        #     print(f"Language model: {count_parameters(self.language_model):,}")
        #     print(stop_here)
    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        loss_mask: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        num_image_tiles: Optional[List[int]] = None,
        image_token_index: Optional[int] = IMAGE_TOKEN_INDEX,
        audios: Optional[torch.Tensor] = None,
        audio_lengths: Optional[List[int]] = None,
        audio_token_index: Optional[int] = AUDIO_TOKEN_INDEX,
        inference_params: Optional[InferenceParams] = None,
        runtime_gather_output: Optional[bool] = None,
        packed_seq_params: Optional[PackedSeqParams] = None,
    ) -> torch.Tensor:
        # pylint: disable=C0301
        """Forward function of the LLaVA model.

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
            packed_seq_params (PackedSeqParams): Dict with padded token information.
                Required for using SP/CP with padding mask type.

        Returns:
            output (torch.Tensor): Loss of shape [b, s] if labels are provided,
                otherwise logits of shape [b, s, vocab_size].
            loss_mask (torch.Tensor): Loss mask expanded to combined sequence length. Shape [b, s].
        """

        # # DEBUGGING
        # print("[Forward step]")
        # print("[Forward step] input_ids.shape: ", input_ids.shape)
        # print("[Forward step] position_ids.shape: ", position_ids.shape)
        # print("[Forward step] loss_mask.shape: ", loss_mask.shape)
        # print("[Forward step] attention_mask.shape: ", attention_mask.shape)
        # print("[Forward step] labels.shape: ", labels.shape)
        # print("[Forward step] images.shape: ", images.shape)
        # print("[Forward step] num_image_tiles.shape: ", num_image_tiles.shape)
        # print("[Forward step] image_token_index: ", image_token_index)
        # print("[Forward step] audios.shape: ", audios.shape)
        # print("[Forward step] audio_lengths.shape: ", audio_lengths.shape)
        # print("[Forward step] audio_token_index: ", audio_token_index)
        # print("----------------------")
        # print(stop_here)


        # TODO: not sure what to do with this?
        use_inference_kv_cache = (
            inference_params is not None and "media_tokens_count" in inference_params.key_value_memory_dict
        )

        # If running inference, we can skip images/audios token computation if they were computed already earlier
        # for this sample.
        if use_inference_kv_cache:
            image_embeddings = None
            audio_embeddings = None
        elif self.add_encoder:
            # Encode images
            if self.vision_model is not None:
                has_images = images is not None and images.shape[0] > 0
                if not has_images:
                    vision_param = next(self.vision_model.parameters())
                    # If no images provided, use an empty image embeddings tensor.
                    image_embeddings = torch.tensor([], dtype=vision_param.dtype, device=vision_param.device).reshape(0, 0, 0)
                else:
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

                    # TODO: not sure what to do with this?
                    # TODO: Support batched inference.
                    # In inference, the language model KV cache will be updated for image token positions.
                    # Store the image tokens sequence length to be used as an offset to the KV cache later.
                    if inference_params is not None:
                        inference_params.key_value_memory_dict["image_tokens_count"] = (
                            image_embeddings.shape[0] * image_embeddings.shape[1]
                        )
                        inference_params.key_value_memory_dict["media_tokens_count"] = inference_params.key_value_memory_dict["image_tokens_count"]
            else:
                image_embeddings = None

            # Encode audios
            if self.audio_model is not None:
                has_audios = audios is not None and audios.shape[0] > 0
                if not has_audios:
                    audio_param = next(self.audio_model.parameters())
                    # If no audios provided, use an empty audio embeddings tensor.
                    audio_embeddings = torch.tensor([], dtype=audio_param.dtype, device=audio_param.device).reshape(0, 0)
                else:
                    # audios is in shape of (num_audios_in_mbs, audio_feature_dim)
                    # note num_audios_in_mbs is not mbs but total audios in this mbs.
                    audios = audios.to(next(self.audio_model.parameters()).dtype)
                    audio_embeddings, audio_embedding_lens = self.audio_model(
                        input_signal = audios,
                        input_signal_length = audio_lengths,
                        processed_signal = None,
                        processed_signal_length = None, # what difference between input_signal and processed_signal?
                    ) 

                    # [num_audios, h_audio, audio_seq_len] -> [audio_seq_len, num_audios, h_audio]
                    # contiguous() required as `permute` can sparsify the tensor and this breaks pipelining
                    audio_embeddings = audio_embeddings.permute(2, 0, 1).contiguous() 

                    # map audio model output size to language model input size.
                    audio_embeddings = self.audio_projection(audio_embeddings)  

                    # TODO: not sure what to do with this?
                    # TODO: Support batched inference.
                    # In inference, the language model KV cache will be updated for audio token positions.
                    # Store the audio tokens sequence length to be used as an offset to the KV cache later.
                    if inference_params is not None:
                        inference_params.key_value_memory_dict["audio_tokens_count"] = (
                            audio_embeddings.shape[0] * audio_embeddings.shape[1]
                        )
                        if "media_tokens_count" in inference_params.key_value_memory_dict:
                            inference_params.key_value_memory_dict["media_tokens_count"] += inference_params.key_value_memory_dict["audio_tokens_count"]
                        else:
                            inference_params.key_value_memory_dict["media_tokens_count"] = inference_params.key_value_memory_dict["audio_tokens_count"]
            else:
                audio_embeddings = None
        else:
            image_embeddings, audio_embeddings = self.encoder_hidden_state

        # # DEBUGGING
        # print("image_embeddings.shape: ", image_embeddings.shape)
        # print("audio_embeddings.shape: ", audio_embeddings.shape)

        if not self.add_decoder:
            return image_embeddings, audio_embeddings


        language_embeddings = None
        if self.pre_process:
            input_ids_text = input_ids.clone()
            # MultiModal Token indices are assumed to be values
            input_ids_text[input_ids_text < 0] = 0

            language_embeddings = self.language_model.embedding(
                input_ids=input_ids_text, position_ids=position_ids
            )  # [text_seq_len, b, h_language]

            language_embeddings = language_embeddings.transpose(1, 0).contiguous()  # [b, text_seq_len, h_language]

        # Assume 1 tile per image if the number of tiles is not provided.
        if num_image_tiles is None:
            num_image_tiles = torch.ones(images.shape[0], dtype=torch.int, device=input_ids.device)
        elif isinstance(num_image_tiles, list):
            num_image_tiles = torch.tensor(num_image_tiles, dtype=torch.int, device=input_ids.device)

        # Preprocess input, labels and loss mask.
        combined_embeddings, final_labels, final_loss_mask, final_attention_mask = self._preprocess_data(
            language_embeddings,
            input_ids,
            attention_mask,
            loss_mask,
            labels,
            image_embeddings,
            num_image_tiles,                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          
            image_token_index,
            audio_embeddings,
            audio_embedding_lens,                                                                                                                      
            audio_token_index,
            use_inference_kv_cache,
            packed_seq_params,
        )  # [combined_seq_len, b, h_language], [b, combined_seq_len], [b, combined_seq_len]

        # # DEBUGGING
        # print(f"combined_embeddings[0]: {combined_embeddings[0]}")
        # print(f"final_labels[0].tolist(): {final_labels[0].tolist()}")
        # print(f"final_loss_mask[0].tolist(): {final_loss_mask[0].tolist()}")
        # print(f"final_attention_mask[0].tolist(): {final_attention_mask[0].tolist()}")
        # print(f"combined_embeddings.shape: {combined_embeddings.shape}")
        # print(f"final_labels.shape: {final_labels.shape}")
        # print(f"final_labels (non_negative_indices)[0].tolist(): {(final_labels[0]>=0).nonzero(as_tuple=True)[0].tolist()}")
        # print(f"final_loss_mask.shape: {final_loss_mask.shape}")
        # print(f"final_loss_mask (non_zero_indices)[0].tolist(): {(final_loss_mask[0]>0).nonzero(as_tuple=True)[0].tolist()}")
        # print(f"final_attention_mask.shape: {final_attention_mask.shape}")
        # print("-----------------")
        # print(stop_here)


        if self.context_parallel_lm > 1 or self.sequence_parallel_lm:
            if self.context_parallel_lm > 1:
                # _process_embedding_token_parallel expects input in shape bshd for cp
                combined_embeddings = combined_embeddings.transpose(1, 0).contiguous()

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

    def freeze(
        self, 
        freeze_language_model: bool, 
        freeze_vision_model: bool, 
        freeze_vision_projection: bool,
        freeze_audio_model: bool, 
        freeze_audio_projection: bool,
    ):
        """Freeze model modules.

        Make specific modules non-trainable by setting requires_grad to False.

        Args:
            freeze_language_model (bool): Freeze the language model module.
            freeze_vision_model (bool): Freeze the vision model module.
            freeze_vision_projection (bool): Freeze the vision projection module.
            freeze_audio_model (bool): Freeze the audio model module.
            freeze_audio_projection (bool): Freeze the audio projection module.
        """
        modules = []
        if freeze_language_model and self.language_model is not None:
            modules.append(self.language_model)
        if freeze_vision_model and self.vision_model is not None:
            modules.append(self.vision_model)
        if freeze_vision_projection and self.vision_projection is not None:
            modules.append(self.vision_projection)
        if freeze_audio_model and self.audio_model is not None:
            modules.append(self.audio_model)
        if freeze_audio_projection and self.audio_projection is not None:
            modules.append(self.audio_projection)

        for module in modules:
            for param in module.parameters():
                param.requires_grad = False

    # TODO: rethink for 2 encoders scenario
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

    def _preprocess_data(
        self,
        language_embeddings,
        input_ids,
        attention_mask,
        loss_mask,
        labels,
        image_embeddings,
        num_image_tiles,
        image_token_index,
        audio_embeddings,
        audio_embedding_lens,
        audio_token_index,
        use_inference_kv_cache,
        packed_seq_params,
    ):
        """
        Preprocesses input data for the model.

        Combining image, audio and text embeddings to final_embedding and creating final_labels and final_loss_mask.

        Returns:
            final_embedding (torch.Tensor): image, audio and text embeddings [combined_seq_len, b, h].
            final_labels (torch.Tensor): labels for image, audio and text positions [b, combined_seq_len].
            final_loss_mask (torch.Tensor): loss mask [b, combined_seq_len].

        """

        assert self.add_decoder, "input text preprocessing is only needed for the language model"

        # No pre- or postprocessing needed.
        # With pipeline parallel > 2, this means a chunk in the middle of the model.
        if not self.pre_process and not self.post_process:
            return language_embeddings, loss_mask, labels, attention_mask

        # # TODO: not sure what to do with this?
        # If using the inference KV cache, the image/audio tokens are already computed.
        if use_inference_kv_cache:
            return language_embeddings, loss_mask, labels, attention_mask

        img_seq_len = self._img_seq_len
        batch_size, text_seq_len = input_ids.shape

        has_labels = labels is not None
        if has_labels:
            assert (
                labels.shape == loss_mask.shape
            ), f"mismatching labels shape {labels.shape} and loss mask shape {loss_mask.shape}"

        packed_sequence = packed_seq_params is not None and packed_seq_params.qkv_format == "thd"

        # Create indices for new text and label positions.
        with torch.no_grad():

            # Compute the final embedding size
            image_token_mask = input_ids == image_token_index
            audio_token_mask = input_ids == audio_token_index
            media_token_mask = image_token_mask | audio_token_mask
            num_images_per_sample = torch.sum(image_token_mask, dim=-1)
            num_image_tiles_batch = num_image_tiles.split(num_images_per_sample.tolist(), dim=0)
            num_image_tiles_batch = torch.tensor([x.sum() for x in num_image_tiles_batch], device=input_ids.device)
            num_audios_per_sample = torch.sum(audio_token_mask, dim=-1)
            audios_embeddings_lengths_batch = audio_embedding_lens.split(num_audios_per_sample.tolist(), dim=0)
            audios_embeddings_lengths_batch = torch.tensor([x.sum() for x in audios_embeddings_lengths_batch], device=input_ids.device)

            # Sequence length for each sample: 
            # (image sequence length multiplied by the number of tiles for that image) + (audio sequence length) 
            # - (number of image tokens + number of audio tokens) 
            # + (text sequence length).
            seq_lens = (
                (num_image_tiles_batch * img_seq_len) + audios_embeddings_lengths_batch 
                - (num_images_per_sample + num_audios_per_sample) 
                + text_seq_len
            )
            max_seq_len = seq_lens.max()
            
            # Pipeline parallel expects fixed input size. Check if we need to pad.
            if self._language_is_pipeline_parallel and max_seq_len < self._language_max_sequence_length:
                max_seq_len = self._language_max_sequence_length
                if packed_sequence:
                    last_seqlen = packed_seq_params.cu_seqlens_q[-1] - packed_seq_params.cu_seqlens_q[-2]
                    last_seqlen_padded = max_seq_len - packed_seq_params.cu_seqlens_q_padded[-2]
                    assert (
                        last_seqlen_padded >= last_seqlen
                    ), "`language_max_sequence_length` needs to increase for sequence packing to work properly."
                    packed_seq_params.cu_seqlens_q_padded[-1] = max_seq_len
                    packed_seq_params.cu_seqlens_kv_padded[-1] = max_seq_len
                    packed_seq_params.max_seqlen_q = max(last_seqlen_padded, packed_seq_params.max_seqlen_q)
                    packed_seq_params.max_seqlen_kv = max(last_seqlen_padded, packed_seq_params.max_seqlen_kv)

            if self.sequence_parallel_lm:
                if self.tp_comm_overlap_lm:
                    # If shorter: Pad to language_max_sequence_length to use TP Comm overlap.
                    # If longer: Gets truncated later.
                    if max_seq_len < self._language_max_sequence_length:
                        padded_seq_len = self._language_max_sequence_length
                else:
                    # Pad to multiple of tp size for sequence parallelism
                    tp_world_size = ps.get_tensor_model_parallel_world_size()
                    padded_seq_len = int((max_seq_len + (tp_world_size - 1)) // tp_world_size * tp_world_size)

                max_seq_len = padded_seq_len
            batch_indices, non_media_indices = torch.where(
                (input_ids != image_token_index) & (input_ids != audio_token_index)
            )

            # New position ids for the text tokens, shifted by the image sequence length.
            # E.g. for input_ids = [-200, 1, 2, 3, -200, 5, 6] and img_seq_len[0] = [576, 300], we get
            # new_position_ids = [576, 577, 578, 579, 879, 880, 881]. text_position_ids are then [577, 578, 579, 880, 881].
            # Build a tensor of contributions: text tokens (non-media) contribute 0 extra,
            # while each media token contributes its media-specific extra length.
            media_token_mask_lens = torch.zeros_like(input_ids, dtype=torch.int32)
            # DEBUGGING
            # print("num_image_tiles type: ", type(num_image_tiles))
            # print("num_image_tiles[0] type: ", type(num_image_tiles[0]))
            # print("num_image_tiles: ", num_image_tiles)
            # print("img_seq_len type: ", type(self._img_seq_len))
            # print("image_token_mask type: ", type(image_token_mask))
            # print("image_token_mask[0] type: ", type(image_token_mask[0]))
            # print("media_token_mask_lens type: ", type(media_token_mask_lens))
            # print("media_token_mask_lens tensor type: ", media_token_mask_lens.dtype)
            # print("----------------------")
            media_token_mask_lens[image_token_mask] = ((num_image_tiles * img_seq_len) - 1).to(torch.int32)
            media_token_mask_lens[audio_token_mask] = (audio_embedding_lens - 1).to(torch.int32)

            # Compute new position ids for every token.
            # We add 1 to every position (for text tokens this gives a step of 1,
            # and for media tokens, it gives a step equal to its extra length + 1),
            # then subtract 1 to adjust for zero-based indexing.
            new_position_ids = torch.cumsum((media_token_mask_lens + 1), dim=-1) - 1

            # # DEBUGGING
            # print("media_token_mask_lens[0]: ", media_token_mask_lens[0].tolist())
            # print("new_position_ids[0]: ", new_position_ids[0].tolist())
            # # print(stop_here)

            # Extract text token positions (non-media tokens).
            text_position_ids = new_position_ids[batch_indices, non_media_indices]

            # Labels are shifted to left by one.
            # So, shift text position ids and non-image indices to left by one.
            if has_labels:
                label_text_position_ids = text_position_ids - 1
                valid_label_text_position_ids = label_text_position_ids >= 0
                label_text_position_ids = label_text_position_ids[valid_label_text_position_ids]

                label_batch_indices = batch_indices[valid_label_text_position_ids]

                label_non_media_indices = non_media_indices - 1
                valid_label_non_media_indices = label_non_media_indices >= 0
                label_non_media_indices = label_non_media_indices[valid_label_non_media_indices]

            # get new position ids for the image/audios tokens (each position is the last position of the image/audio tokens in each sample)
            new_images_position_ids = []
            new_audios_position_ids = []
            for i in range(len(new_position_ids)):
                new_images_position_ids.append(new_position_ids[i][image_token_mask[i]])
                new_audios_position_ids.append(new_position_ids[i][audio_token_mask[i]])

            # # DEBUGGING
            # print("new_images_position_ids: ", new_images_position_ids)
            # print("new_audios_position_ids: ", new_audios_position_ids)
            # # print(stop_here)


        # Create the final input embedding (if this is the first language model stage).
        final_embedding = None
        if self.pre_process:
            embed_dim = language_embeddings.shape[-1]
            final_embedding = torch.zeros(
                batch_size,
                max_seq_len,
                embed_dim,
                dtype=language_embeddings.dtype,
                device=language_embeddings.device,
            )

            # Put text embeddings to the text positions in the result tensor.
            final_embedding[batch_indices, text_position_ids] = language_embeddings[batch_indices, non_media_indices]

            # # DEBUGGING
            # print("images_mask.shape: ", images_mask.shape)
            # print("final_embedding.shape: ", final_embedding.shape)
            # print("image_embeddings.shape: ", image_embeddings.shape)
            # print("image_embeddings.permute(1, 0, 2).reshape(-1, embed_dim).contiguous().shape: ", image_embeddings.permute(1, 0, 2).reshape(-1, embed_dim).contiguous().shape)
            # print("final_embedding[images_mask].shape: ", final_embedding[images_mask].shape)

            # # Put image and audio embeddings to image and audios positions.
            # # NOTE: final_embedding [batch_size, max_seq_len, embed_dim]
            # final_embedding[images_mask] = image_embeddings.permute(1, 0, 2).reshape(-1, embed_dim).contiguous()
            # final_embedding[audios_mask] = audio_embeddings.permute(1, 0, 2).reshape(-1, embed_dim).contiguous()

            # Put image embeddings to the last position of the image tokens
            image_embeddings = image_embeddings.permute(1, 0, 2).reshape(-1, embed_dim).contiguous()
            image_pointer = 0
            tile_pointer = 0
            images_mask = torch.full((batch_size, max_seq_len), False, dtype=torch.bool, device=input_ids.device)
            for i in range(final_embedding.shape[0]):
                for j in range(len(new_images_position_ids[i])):
                    current_image_seq_len = num_image_tiles[image_pointer]*img_seq_len
                    current_image_tokens_end_idx = new_images_position_ids[i][j]
                    current_image_tokens_start_idx = current_image_tokens_end_idx - current_image_seq_len

                    # # DEBUGGING
                    # print("final_embedding.shape: ", final_embedding.shape)
                    # print("image_embeddings.shape: ", image_embeddings.shape)
                    # print("final_embedding[i][current_image_tokens_start_idx:current_image_tokens_end_idx].shape: ", final_embedding[i][current_image_tokens_start_idx:current_image_tokens_end_idx].shape)
                    # print("image_embeddings[tile_pointer : (tile_pointer + current_image_seq_len)].shape: ", image_embeddings[tile_pointer : (tile_pointer + current_image_seq_len)].shape)

                    final_embedding[i][current_image_tokens_start_idx:current_image_tokens_end_idx] = image_embeddings[tile_pointer : (tile_pointer + current_image_seq_len)]
                    images_mask[i][current_image_tokens_start_idx:current_image_tokens_end_idx] = True
                    tile_pointer += num_image_tiles[image_pointer]
                    image_pointer += 1
            # print(stop_here)    

            # Put audio embeddings to the last position of the audio tokens
            audio_embeddings = audio_embeddings.permute(1, 0, 2).contiguous()
            audio_pointer = 0
            audio_length_pointer = 0
            audios_mask = torch.full((batch_size, max_seq_len), False, dtype=torch.bool, device=input_ids.device)
            for i in range(final_embedding.shape[0]):
                for j in range(len(new_audios_position_ids[i])):
                    current_audio_seq_len = audio_embedding_lens[audio_length_pointer]
                    current_audio_tokens_end_idx = new_audios_position_ids[i][j]
                    current_audio_tokens_start_idx = current_audio_tokens_end_idx - current_audio_seq_len

                    # # DEBUGGING
                    # print("final_embedding.shape: ", final_embedding.shape)
                    # print("audio_embeddings.shape: ", audio_embeddings.shape)
                    # print("final_embedding[i][current_audio_tokens_start_idx:current_audio_tokens_end_idx].shape: ", final_embedding[i][current_audio_tokens_start_idx:current_audio_tokens_end_idx].shape)
                    # print("audio_embeddings[audio_pointer][:current_audio_seq_len].shape: ", audio_embeddings[audio_pointer][:current_audio_seq_len].shape)

                    final_embedding[i][current_audio_tokens_start_idx:current_audio_tokens_end_idx] = audio_embeddings[audio_pointer][:current_audio_seq_len]
                    audios_mask[i][current_audio_tokens_start_idx:current_audio_tokens_end_idx] = True
                    audio_pointer += 1
                    audio_length_pointer += 1
            # print(stop_here)

        # Create the final labels and loss mask (if this is the last language model stage).
        final_labels, final_loss_mask = None, None
        if has_labels:
            final_labels = torch.full(
                (batch_size, max_seq_len), IGNORE_INDEX, dtype=labels.dtype, device=labels.device
            )
            final_loss_mask = torch.full((batch_size, max_seq_len), 0, dtype=loss_mask.dtype, device=loss_mask.device)

            # Put text labels and loss mask to the text positions.
            final_labels[label_batch_indices, label_text_position_ids] = labels[
                label_batch_indices, label_non_media_indices
            ]

            final_loss_mask[batch_indices, text_position_ids] = loss_mask[batch_indices, non_media_indices]

            # For labels, pick the last label index that got dropped by the shift to left.
            label_extra_text_position_ids = seq_lens - 1
            batch_range = torch.arange(len(label_extra_text_position_ids))
            final_labels[batch_range, label_extra_text_position_ids] = labels[batch_range, -1]

            # Loss mask the media positions.
            final_loss_mask[images_mask] = 0
            final_loss_mask[audios_mask] = 0

            # Loss mask last text position just before an image/audio
            # so that text token does not need to predict the first image/audio token.
            batch_media_indices, media_indices = torch.where(media_token_mask)
            # Indices just before media tokens. If it's -1, skip it.
            before_media_indices = media_indices - 1
            valid = before_media_indices >= 0
            valid_batch_media_indices = batch_media_indices[valid]
            valid_before_media_indices = before_media_indices[valid]
            # Map those indices those position ids.
            valid_before_media_indices = new_position_ids[valid_batch_media_indices, valid_before_media_indices]

            final_loss_mask[valid_batch_media_indices, valid_before_media_indices] = 0

        if final_embedding is not None and has_labels:
            assert (
                final_embedding.shape[:2] == final_labels.shape == final_loss_mask.shape
            ), "unexpected shapes after data preprocessing"

        truncate_labels = has_labels and final_labels.shape[1] > self._language_max_sequence_length
        if truncate_labels:
            final_labels = final_labels[:, : self._language_max_sequence_length]
            final_loss_mask = final_loss_mask[:, : self._language_max_sequence_length]

        # truncate final embedding
        if final_embedding is not None:
            # transpose final_embeddings to sbhd
            # note this will also transpose thd, which is fine
            final_embedding = final_embedding.transpose(1, 0).contiguous()
            # Truncate if exceeding the language model's max sequence length.
            if final_embedding.shape[0] > self._language_max_sequence_length:
                final_embedding = final_embedding[: self._language_max_sequence_length]

        # packed seq param truncation
        if packed_sequence and packed_seq_params.cu_seqlens_q_padded[-1] > self._language_max_sequence_length:
            truncate_len = packed_seq_params.cu_seqlens_q_padded[-1] - self._language_max_sequence_length
            final_seq_len_padded = (
                packed_seq_params.cu_seqlens_q_padded[-1] - packed_seq_params.cu_seqlens_q_padded[-2]
            )
            final_seq_len_unpadded = packed_seq_params.cu_seqlens_q[-1] - packed_seq_params.cu_seqlens_q[-2]
            final_padding = final_seq_len_padded - final_seq_len_unpadded
            truncate_len -= final_padding
            packed_seq_params.cu_seqlens_q_padded[-1] = self._language_max_sequence_length
            packed_seq_params.cu_seqlens_kv_padded[-1] = self._language_max_sequence_length
            # need to truncate the actual sequence as well
            if truncate_len > 0:
                packed_seq_params.cu_seqlens_q[-1] -= truncate_len
                packed_seq_params.cu_seqlens_kv[-1] -= truncate_len
            assert (
                packed_seq_params.cu_seqlens_q[-1] >= packed_seq_params.cu_seqlens_q[-2]
            ), "with packed sequence, the truncation can only truncate on the last sequence."

        return final_embedding, final_labels, final_loss_mask, attention_mask

    def _process_embedding_token_parallel(self, combined_embeddings, new_labels, new_loss_mask, packed_seq_params):
        """Processes the input data for model parallelism support."""

        # No pre or post processing needed with PP middle chunks.
        if not self.pre_process and not self.post_process:
            return combined_embeddings, new_labels, new_loss_mask, packed_seq_params

        if self.pre_process:
            if self.context_parallel_lm > 1 and self.sequence_parallel_lm:
                shard_factor = self.tensor_model_parallel_size_lm * self.context_parallel_lm * 2
                seq_dim = 1
            elif self.context_parallel_lm > 1:
                shard_factor = self.context_parallel_lm * 2
                seq_dim = 1
            elif self.sequence_parallel_lm:
                shard_factor = self.tensor_model_parallel_size_lm
                seq_dim = 0

            assert (
                combined_embeddings.shape[seq_dim] % shard_factor == 0
            ), f"Sequence length should be divisible by {shard_factor} for \
                Sequence/Context parallelism"
            if self.sequence_parallel_lm and self.tp_comm_overlap_lm:
                assert (
                    combined_embeddings.shape[seq_dim] == self._language_max_sequence_length
                ), "TP Comm overlap either requires Vision+Text token length \
                == language_max_sequence_length"

        if self.context_parallel_lm > 1:
            batch = dict()
            if self.pre_process:
                batch.update(
                    {
                        "combined_embeddings": combined_embeddings,
                    }
                )
            if self.post_process:
                batch.update(
                    {
                        "new_labels": new_labels,
                        "new_loss_mask": new_loss_mask,
                    }
                )
            # Distribute sequence across CP ranks
            if packed_seq_params is None or packed_seq_params.qkv_format == 'sbhd':
                from megatron.training.utils import get_batch_on_this_cp_rank

                batch = get_batch_on_this_cp_rank(batch)
            else:
                batch = _get_data_on_this_cp_rank.apply(batch, packed_seq_params)

            if self.pre_process:
                combined_embeddings = batch["combined_embeddings"]  # [B, S/CP, H]
                combined_embeddings = combined_embeddings.transpose(1, 0).contiguous()  # [B,S/CP,H] -> [S/CP,B,H]
            if self.post_process:
                new_labels = batch["new_labels"]
                new_loss_mask = batch["new_loss_mask"]

        if self.sequence_parallel_lm and self.pre_process:
            combined_embeddings = tensor_parallel.scatter_to_sequence_parallel_region(
                combined_embeddings
            )  # [S/(CP*TP),B,H]

        return combined_embeddings, new_labels, new_loss_mask, packed_seq_params


class AVLMModel(L.LightningModule, io.IOMixin, io.ConnectorMixin, fn.FNMixin):
    """Lightning Wrapper for AVLM Model"""

    def __init__(
        self,
        config: AVLMConfig,
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
        # pylint: disable=C0115,C0116
        if not hasattr(self, "module"):
            self.module = self.config.configure_model(self.tokenizer)

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        loss_mask: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        num_image_tiles: Optional[List[int]] = None,
        image_token_index: Optional[int] = IMAGE_TOKEN_INDEX,
        # image_token_mask: Optional[torch.Tensor] = None,
        audios: Optional[torch.Tensor] = None,
        audio_lengths: Optional[List[int]] = None,
        audio_token_index: Optional[int] = AUDIO_TOKEN_INDEX,
        inference_params: Optional[InferenceParams] = None,
        runtime_gather_output: Optional[bool] = None,
        packed_seq_params: Optional[PackedSeqParams] = None,
    ) -> torch.Tensor:
        # pylint: disable=C0115,C0116
        output_tensor = self.module(
            input_ids=input_ids,
            position_ids=position_ids,
            loss_mask=loss_mask,
            attention_mask=attention_mask,
            labels=labels,
            images=images,
            num_image_tiles=num_image_tiles,
            image_token_index=image_token_index,
            # image_token_mask=image_token_mask,
            audios=audios,
            audio_lengths=audio_lengths,
            audio_token_index=audio_token_index,
            inference_params=inference_params,
            runtime_gather_output=runtime_gather_output,
            packed_seq_params=packed_seq_params,
        )

        return output_tensor

    def data_step(self, dataloader_iter) -> Dict[str, torch.Tensor]:
        # pylint: disable=C0115,C0116
        return self.config.data_step_fn(dataloader_iter)

    def forward_step(self, batch) -> torch.Tensor:
        # pylint: disable=C0115,C0116
        return self.config.forward_step_fn(self, batch)

    def training_step(self, batch, batch_idx=None) -> torch.Tensor:
        # pylint: disable=C0115,C0116
        # In mcore the loss-function is part of the forward-pass (when labels are provided)
        return self.forward_step(batch)

    def validation_step(self, batch, batch_idx=None) -> torch.Tensor:
        # pylint: disable=C0115,C0116
        # In mcore the loss-function is part of the forward-pass (when labels are provided)

        return self.forward_step(batch)

    @property
    def training_loss_reduction(self) -> MaskedTokenLossReductionWithLossMask:
        # pylint: disable=C0115,C0116
        if not self._training_loss_reduction:
            self._training_loss_reduction = MaskedTokenLossReductionWithLossMask()

        return self._training_loss_reduction

    @property
    def validation_loss_reduction(self) -> MaskedTokenLossReductionWithLossMask:
        # pylint: disable=C0115,C0116
        if not self._validation_loss_reduction:
            self._validation_loss_reduction = MaskedTokenLossReductionWithLossMask(validation_step=True)

        return self._validation_loss_reduction


__all__ = [
    "AVLMModel",
    "AVLMConfig",
    "AVLM_data_step",
    "AVLM_forward_step",
]
