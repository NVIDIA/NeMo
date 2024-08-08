from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Dict, Literal, Optional, Union
from einops import rearrange

import pytorch_lightning as L
import torch
import torch.distributed
from megatron.core.optimizer import OptimizerConfig
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_config import TransformerConfig
from torch import nn

from nemo.collections.llm import fn
from nemo.lightning import get_vocab_size, io
from nemo.lightning.megatron_parallel import MaskedTokenLossReduction
from nemo.lightning.pytorch.optim import MegatronOptimizerModule, OptimizerModule
from nemo.collections.llm.gpt.model import transformer_engine_layer_spec, local_layer_spec
from nemo.collections.common.tokenizers.tokenizer_spec import TokenizerSpec
from nemo.utils import logging

from megatron.core.models.multimodal.llava_model import LLaVAModel as MCoreLLaVAModel


def neva_data_step(dataloader_iter) -> Dict[str, torch.Tensor]:
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
    required_keys.add("attention_mask")
    if parallel_state.is_pipeline_first_stage():
        required_keys.update(("media", "tokens", "position_ids"))
    if parallel_state.is_pipeline_last_stage():
        required_keys.update(("labels", "loss_mask"))

    _batch = {
        key: val.cuda(non_blocking=True)
        if key in required_keys and val is not None else None
        for key, val in _batch.items()
    }
    # slice batch along sequence dimension for context parallelism
    output = get_batch_on_this_context_parallel_rank(_batch)

    return output


def neva_forward_step(model, batch) -> torch.Tensor:
    forward_args = {
        "media": batch["media"],
        "input_ids": batch["tokens"],
        "position_ids": batch["position_ids"],
        "attention_mask": batch.get("attention_mask", None),
        "loss_mask": batch.get("loss_mask", None),
        "labels": batch.get("labels", None),
    }

    if 'cu_seqlens' in batch:
        forward_args['packed_seq_params'] = get_packed_seq_params(batch)

    return model(**forward_args)

from nemo.collections.llm.gpt.model.base import get_batch_on_this_context_parallel_rank, get_packed_seq_params
from megatron.core.models.vision.clip_vit_model import CLIPViTModel as MCoreCLIPViTModel
from megatron.core.transformer.custom_layers.transformer_engine import (
    TEColumnParallelLinear,
    TEDotProductAttention,
    TELayerNormColumnParallelLinear,
    TENorm,
    TERowParallelLinear,
)

from megatron.core.models.gpt.gpt_layer_specs import _get_mlp_module_spec
from megatron.core.models.vision.multimodal_projector import MultimodalProjector as MCoreMultimodalProjector
from megatron.core.inference_params import InferenceParams
from megatron.core.transformer.mlp import MLP, MLPSubmodules
from nemo.collections.nlp.modules.common.megatron.module import MegatronModule
from megatron.core.transformer.enums import AttnMaskType, ModelType

@dataclass
class MultimodalProjectorConfig(TransformerConfig, io.IOMixin):
    projector_type: str = "mlp"
    input_size: Optional[int] = None
    layer_spec: Optional[MLPSubmodules] = None

    def configure_model(self) -> "MCoreMultimodalProjector":
        if self.layer_spec is None:
            if self.projector_type == "mlp":
                self.layer_spec = _get_mlp_module_spec().submodules

        return MCoreMultimodalProjector(
            self,
            self.layer_spec,
            projector_type=self.projector_type,
            input_size=self.input_size,
        )

@dataclass
class CLIPViTConfig(TransformerConfig, io.IOMixin):
    ln_pre_impl: Union[ModuleSpec, type] = TENorm
    add_class_token: bool = True
    class_token_len: int = 1
    patch_dim: int = 14
    img_h: int = 336
    img_w: int = 336
    transformer_layer_spec: ModuleSpec = transformer_engine_layer_spec

    def configure_model(self) -> "MCoreCLIPViTModel":
        transformer_layer_spec = self.transformer_layer_spec
        if not isinstance(transformer_layer_spec, ModuleSpec):
            transformer_layer_spec = transformer_layer_spec(self)
        return MCoreCLIPViTModel(
            self,
            transformer_layer_spec,
            ln_pre_impl=self.ln_pre_impl,
            add_class_token=self.add_class_token,
            class_token_len=self.class_token_len,
            patch_dim=self.patch_dim,
            img_h=self.img_h,
            img_w=self.img_w,
        )


@dataclass
class NevaConfig(TransformerConfig, io.IOMixin):
    language_transformer_config: Optional[TransformerConfig] = None
    vision_transformer_config: Optional[TransformerConfig] = None
    vision_projection_config: Optional[TransformerConfig] = None
    drop_vision_class_token: bool = True
    allow_missing_vision_projection_checkpoint: bool = False
    img_embedding_idx: int = 0

    forward_step_fn: Callable = neva_forward_step
    data_step_fn: Callable = neva_data_step

    def configure_model(self, tokenizer) -> "MCoreLLaVAModel":
        language_model = self.language_transformer_config.configure_model(tokenizer)
        vision_model = self.vision_transformer_config.configure_model()
        vision_projection = self.vision_projection_config.configure_model()
        return MCoreNevaModel(
            transformer_config=self,
            language_model=language_model,
            vision_model=vision_model,
            vision_projection=vision_projection,
            drop_vision_class_token=self.drop_vision_class_token,
            img_embedding_idx=self.img_embedding_idx,
        )


class MCoreNevaModel(MCoreLLaVAModel):
    def __init__(
        self,
        transformer_config: TransformerConfig,
        language_model: MegatronModule,
        vision_model: MegatronModule,
        vision_projection: MegatronModule,
        pre_process: bool = True,
        post_process: bool = True,
        drop_vision_class_token: bool = False,
        img_embedding_idx: int = 0,
    ) -> None:
        super(MCoreLLaVAModel, self).__init__(config=transformer_config)

        logging.warning(
            "LLaVA model is under development and may be missing features."
        )

        self.pre_process = pre_process
        self.post_process = post_process
        self.img_embedding_idx = img_embedding_idx

        self.encoder_hidden_state = None
        self.vision_model = vision_model
        self.vision_projection = vision_projection
        self.language_model = language_model
        self.model_type = ModelType.encoder_or_decoder
        # This attribute is needed to check if an all-reduce is required
        # on the word embeddings inside `finalize_model_grads._allreduce_word_embedding_grads`.
        self.share_embeddings_and_output_weights = False
        if self.language_model is not None:
            self.share_embeddings_and_output_weights = (
                self.language_model.share_embeddings_and_output_weights
            )

        if self.vision_model is not None:
            self._drop_vision_class_token = drop_vision_class_token

        self.add_encoder = self.vision_model is not None
        self.add_decoder = self.language_model is not None

    def _merge_input_ids_with_media_features(self, media_features, inputs_embeds, input_ids, loss_mask, labels):
        """
        modified from llava next _merge_input_ids_with_image_features
        https://github.com/huggingface/transformers/blob/main/src/transformers/models/llava_next/modeling_llava_next.py#L409
        """
        ignore_index = -100
        media_token_index = -200  #TODO(yuya): update

        num_medias, num_media_patches, embed_dim = media_features.shape
        batch_size, sequence_length = input_ids.shape
        # 1. Create a mask to know where special media tokens are
        special_media_token_mask = input_ids == media_token_index
        num_special_media_tokens = torch.sum(special_media_token_mask, dim=-1)
        # Compute the maximum embed dimension
        max_embed_dim = (num_special_media_tokens.max() * (num_media_patches - 1)) + sequence_length
        batch_indices, non_media_indices = torch.where(input_ids != media_token_index)

        # 2. Compute the positions where text should be written
        # Calculate new positions for text tokens in merged media-text sequence.
        # `special_media_token_mask` identifies media tokens. Each media token will be replaced by `nb_text_tokens_per_medias - 1` text tokens.
        # `torch.cumsum` computes how each media token shifts subsequent text token positions.
        # - 1 to adjust for zero-based indexing, as `cumsum` inherently increases indices by one.
        new_token_positions = torch.cumsum((special_media_token_mask * (num_media_patches - 1) + 1), -1) - 1
        nb_media_pad = max_embed_dim - 1 - new_token_positions[:, -1]
        text_to_overwrite = new_token_positions[batch_indices, non_media_indices]

        # 3. Create the full embedding, already padded to the maximum position
        final_embedding = torch.zeros(
            batch_size, max_embed_dim, embed_dim, dtype=inputs_embeds.dtype, device=inputs_embeds.device
        )
        final_loss_mask = torch.zeros(
            (batch_size, max_embed_dim), dtype=loss_mask.dtype, device=loss_mask.device
        )
        if labels is not None:
            final_labels = torch.full(
                (batch_size, max_embed_dim), ignore_index, dtype=input_ids.dtype, device=input_ids.device
            )
        # In case the Vision model or the Language model has been offloaded to CPU, we need to manually
        # set the corresponding tensors into their correct target device.
        target_device = inputs_embeds.device
        batch_indices, non_media_indices, text_to_overwrite = (
            batch_indices.to(target_device),
            non_media_indices.to(target_device),
            text_to_overwrite.to(target_device),
        )

        # 4. Fill the embeddings based on the mask. If we have ["hey" "<media>", "how", "are"]
        # we need to index copy on [0, 577, 578, 579] for the text and [1:576] for the media features
        final_embedding[batch_indices, text_to_overwrite] = inputs_embeds[batch_indices, non_media_indices]
        final_loss_mask[batch_indices, text_to_overwrite] = loss_mask[batch_indices, non_media_indices]
        if labels is not None:
            final_labels[batch_indices, text_to_overwrite] = labels[batch_indices, non_media_indices]

        # 5. Fill the embeddings corresponding to the medias. Anything that is not `text_positions` needs filling (#29835)
        media_to_overwrite = torch.full(
            (batch_size, max_embed_dim), True, dtype=torch.bool, device=inputs_embeds.device
        )
        media_to_overwrite[batch_indices, text_to_overwrite] = False
        media_to_overwrite &= media_to_overwrite.cumsum(-1) - 1 >= nb_media_pad[:, None].to(target_device)

        if media_to_overwrite.sum() != media_features.shape[:-1].numel():
            raise ValueError(
                f"The input provided to the model are wrong. The number of media tokens is {torch.sum(special_media_token_mask)} while"
                f" the number of media given to the model is {num_medias}. This prevents correct indexing and breaks batch generation."
            )

        final_embedding[media_to_overwrite] = media_features.contiguous().reshape(-1, embed_dim).to(target_device)

        if labels is None:
            final_labels = None
        return final_embedding, final_loss_mask, final_labels

    def _preprocess_data(
        self,
        media_embeddings,
        input_ids,
        position_ids,
        loss_mask,
        labels,
        use_inference_kv_cache,
    ):
        """Preprocess input data before input to language model.

        image_token_index = -200 signifies the image position in the input_ids = [1, 2, -200, 3, 4] and labels = [2, -200, 3, 4, 5], for example.
        We want to replace the image position (-200) with image_embeddings and return the following:
        - new_embeddings = [language_embeddings1, image_embeddings, language_embeddings2],
            where language_embeddings1/2 are embeddings for [1, 2] and [3, 4], respectively.
        - new_loss_mask = [1, 0, 1, 1, 1]
        - labels = [2, -100, 3, 4, 5]

        This function also handles the case where the input does not contain an image (text-only sample).

        If pipeline parallelism is not used, then self.pre_process and self.post_process are both True and we update both
        input embeddings, labels and loss masks (if available).

        If pipeline parallelism is used, then we do the following
        - the first language model chunk has self.pre_process = True and self.post_process = False. We update input embeddings.
        - the middle language model chunk(s) has self.pre_process = False and self.post_process = False. We don't need to update anything.
        - the last language model chunk has self.pre_process = False and self.post_process = True. We update labels and loss mask.
        """
        assert self.language_model is not None, "data preprocessing should only run for the language model"

        # No pre- or postprocessing needed. With pipeline parallel > 2, this means a chunk in the middle of the model.
        if not self.pre_process and not self.post_process:
            return None, loss_mask, labels

        # If using the inference KV cache, the image tokens are already computed.
        if use_inference_kv_cache:
            language_embeddings = None
            if self.pre_process:
                language_embeddings = self.language_model.embedding(
                    input_ids=input_ids, position_ids=position_ids
                )  # [text_seq_len, b, h_language]

            return language_embeddings, loss_mask, labels

        assert input_ids.shape == position_ids.shape, "mismatching input shapes"
        if labels is not None:
            assert labels.shape == loss_mask.shape, "mismatching labels and loss mask shapes"

        if self.pre_process:
            input_language_embeddings_ids = input_ids.clone()
            # MultiModal Token indices are assumed to be values
            input_language_embeddings_ids[input_ids < 0] = 0
            language_embeddings = self.language_model.embedding(
                input_ids=input_language_embeddings_ids, position_ids=position_ids
            )  # [text_seq_len, b, h_language]

            language_embeddings = language_embeddings.transpose(0, 1)  # [b, text_seq_len, h_language]
            combined_embeddings, loss_mask, labels = self._merge_input_ids_with_media_features(
                media_embeddings, language_embeddings, input_ids, loss_mask, labels
            )
            combined_embeddings = combined_embeddings.transpose(0, 1).contiguous()  # [text_seq_len, b, h_language]

        else:
            combined_embeddings = None

        return combined_embeddings, loss_mask, labels


    def forward(
        self,
        media: torch.Tensor,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        loss_mask: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor = None,
        inference_params: InferenceParams = None,
    ) -> torch.Tensor:
        use_inference_kv_cache = (
            inference_params is not None
            and "image_tokens_count" in inference_params.key_value_memory_dict
        )
        # If running inference, we can skip media token computation if they were computed already earlier for this sample.
        if use_inference_kv_cache:
            media_embeddings = None
        elif self.vision_model is not None:
            # mbs, medias_per_micro_batch, frames
            b, T, F = media.shape[:3] 
            media = rearrange(media, "b T F c h w -> (b T F) c h w")
            media_embeddings = self.vision_model(media)  # [b, img_seq_len, h_vision]
            if self._drop_vision_class_token:
                media_embeddings = media_embeddings[:, self.vision_model.class_token_len :, :]
            # map vision model output size to language model input size.
            media_embeddings = self.vision_projection(
                media_embeddings
            )  # [img_seq_len, b, h_vision]

            # If running inference, the language model KV cache will be updated for media token positions.
            # Here we store the media tokens sequence length, which can be used as an offset to the KV cache later.
            if inference_params is not None:
                inference_params.key_value_memory_dict["media_tokens_count"] = (
                    media_embeddings.shape[0]
                )
        else:
            media_embeddings = self.encoder_hidden_state

        if not self.add_decoder:
            return media_embeddings, loss_mask

        # Preprocess input, labels and loss mask.
        combined_embeddings, loss_mask, labels = self._preprocess_data(
            media_embeddings,
            input_ids,
            position_ids,
            loss_mask,
            labels,
            use_inference_kv_cache,
        )

        output = self.language_model(
            input_ids=None,
            position_ids=None,
            attention_mask=attention_mask,
            decoder_input=combined_embeddings,
            labels=labels,
            inference_params=inference_params,
        )

        return output, loss_mask


class NevaModel(L.LightningModule, io.IOMixin, io.ConnectorMixin, fn.FNMixin):
    def __init__(
            self,
            config: NevaConfig,
            # TODO: Add transformer_layer_spec when we update mcore
            optim: Optional[OptimizerModule] = None,
            tokenizer: Optional["TokenizerSpec"] = None,
            media_processor: Optional = None,  # TODO(yuya): add class type
            model_transform: Optional[Callable[[nn.Module], nn.Module]] = None,
    ):
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer
        self.media_processor = media_processor
        self.optim = optim or MegatronOptimizerModule(config=OptimizerConfig(lr=1e-4, use_distributed_optimizer=True))
        self.optim.connect(self)  # This will bind the `configure_optimizers` method
        self.model_transform = model_transform

    def configure_model(self) -> None:
        if not hasattr(self, "module"):
            self.module = self.config.configure_model(self.tokenizer)

    def forward(
            self,
            media: torch.Tensor,
            input_ids: torch.Tensor,
            position_ids: torch.Tensor,
            loss_mask: torch.Tensor,
            attention_mask: torch.Tensor,
            labels: torch.Tensor = None,
            inference_params: InferenceParams = None,
    ) -> torch.Tensor:
        output_tensor = self.module(
            media,
            input_ids,
            position_ids,
            loss_mask,
            attention_mask,
            labels=labels,
            inference_params=inference_params,
        )

        return output_tensor

    def data_step(self, dataloader_iter) -> Dict[str, torch.Tensor]:
        return self.config.data_step_fn(dataloader_iter)

    def forward_step(self, batch) -> torch.Tensor:
        return self.config.forward_step_fn(self, batch)

    def training_step(self, batch, batch_idx=None) -> torch.Tensor:
        # In mcore the loss-function is part of the forward-pass (when labels are provided)
        return self.forward_step(batch)

    def validation_step(self, batch, batch_idx=None) -> torch.Tensor:
        # In mcore the loss-function is part of the forward-pass (when labels are provided)

        return self.forward_step(batch)

    def training_loss_reduction(self) -> MaskedTokenLossReduction:
        return MaskedTokenLossReduction()

    def validation_loss_reduction(self) -> MaskedTokenLossReduction:
        return MaskedTokenLossReduction(validation_step=True)


# __all__ = [
#     "GPTModel",
#     "GPTConfig",
#     "gpt_data_step",
#     "gpt_forward_step",
#     "transformer_engine_layer_spec",
#     "local_layer_spec",
# ]
