from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Dict, Literal, Optional, Union

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

from nemo.utils import logging

if TYPE_CHECKING:
    from megatron.core.models.multimodal.llava_model import MCoreLLaVAModel
    from nemo.collections.common.tokenizers.tokenizer_spec import TokenizerSpec


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

    _batch = {key: val.cuda(non_blocking=True) if key in required_keys else None for key, val in _batch.items()}
    # slice batch along sequence dimension for context parallelism
    output = get_batch_on_this_context_parallel_rank(_batch)

    return output


def neva_forward_step(model, batch) -> torch.Tensor:
    forward_args = {
        "media": batch["media"],
        "input_ids": batch["tokens"],
        "position_ids": batch["position_ids"],
        "attention_mask": batch["attention_mask"],
        "labels": batch["labels"],
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

from megatron.core.models.vision.multimodal_projector import MultimodalProjector as MCoreMultimodalProjector
from megatron.core.inference_params import InferenceParams
from megatron.core.transformer.mlp import MLP, MLPSubmodules
from nemo.collections.nlp.modules.common.megatron.module import MegatronModule


@dataclass
class MultimodalProjectorConfig(TransformerConfig, io.IOMixin):
    projector_type: str
    input_size: int
    layer_spec: ModuleSpec = MLPSubmodules

    def configure_model(self) -> "MCoreMultimodalProjector":
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
        return MCoreCLIPViTModel(
            self,
            self.transformer_layer_spec,
            ln_pre_impl=self.ln_pre_impl,
            add_class_token=self.add_class_token,
            class_token_len=self.class_token_len,
            patch_dim=self.patch_dim,
            img_h=self.img_h,
            img_w=self.img_w,
        )


@dataclass
class NevaConfig(TransformerConfig, io.IOMixin):
    language_transformer_config: TransformerConfig
    vision_transformer_config: TransformerConfig
    vision_projection_config: TransformerConfig
    drop_vision_class_token: bool
    vision_projection_config: TransformerConfig
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
        MegatronModule.__init__(self, config=transformer_config)

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

        # This attribute is needed to check if an all-reduce is required
        # on the word embeddings inside `finalize_model_grads._allreduce_word_embedding_grads`.
        self.share_embeddings_and_output_weights = False
        if self.language_model is not None:
            self.share_embeddings_and_output_weights = (
                self.language_model.share_embeddings_and_output_weights
            )

        if self.vision_model is not None:
            self._drop_vision_class_token = drop_vision_class_token
            # Map (intermediate) vision model outputs to the language model input dimension.
            # self.vision_projection = MultimodalProjector(
            #     vision_projection_config,
            #     vision_projection_layer_spec,
            #     vision_projection_type,
            #     vision_transformer_config.hidden_size,  # input size to the projection.
            # )
            # # This allows ignoring missing weights for the vision projection during checkpoint loading.
            # # This should be disabled by default but can be enabled if your checkpoint contains pretrained
            # # vision and language models but not the projection from vision model outputs to language model inputs.
            # if allow_missing_vision_projection_checkpoint:
            #     vision_projection_param_names = [
            #         f"vision_projection.{name}"
            #         for name in self.vision_projection.state_dict().keys()
            #     ]
            #     self.vision_projection.register_load_state_dict_post_hook(
            #         partial(_load_state_dict_hook_ignore_param_names, vision_projection_param_names)
            #     )

    def _merge_input_ids_with_image_features(self, image_features, inputs_embeds, input_ids, attention_mask, labels):
        num_images, num_image_patches, embed_dim = image_features.shape
        batch_size, sequence_length = input_ids.shape
        left_padding = not torch.sum(input_ids[:, -1] == torch.tensor(self.pad_token_id))
        # 1. Create a mask to know where special image tokens are
        special_image_token_mask = input_ids == self.config.image_token_index
        num_special_image_tokens = torch.sum(special_image_token_mask, dim=-1)
        # Compute the maximum embed dimension
        max_embed_dim = (num_special_image_tokens.max() * (num_image_patches - 1)) + sequence_length
        batch_indices, non_image_indices = torch.where(input_ids != self.config.image_token_index)

        # 2. Compute the positions where text should be written
        # Calculate new positions for text tokens in merged image-text sequence.
        # `special_image_token_mask` identifies image tokens. Each image token will be replaced by `nb_text_tokens_per_images - 1` text tokens.
        # `torch.cumsum` computes how each image token shifts subsequent text token positions.
        # - 1 to adjust for zero-based indexing, as `cumsum` inherently increases indices by one.
        new_token_positions = torch.cumsum((special_image_token_mask * (num_image_patches - 1) + 1), -1) - 1
        nb_image_pad = max_embed_dim - 1 - new_token_positions[:, -1]
        if left_padding:
            new_token_positions += nb_image_pad[:, None]  # offset for left padding
        text_to_overwrite = new_token_positions[batch_indices, non_image_indices]

        # 3. Create the full embedding, already padded to the maximum position
        final_embedding = torch.zeros(
            batch_size, max_embed_dim, embed_dim, dtype=inputs_embeds.dtype, device=inputs_embeds.device
        )
        final_attention_mask = torch.zeros(
            batch_size, max_embed_dim, dtype=attention_mask.dtype, device=inputs_embeds.device
        )
        if labels is not None:
            final_labels = torch.full(
                (batch_size, max_embed_dim), self.config.ignore_index, dtype=input_ids.dtype, device=input_ids.device
            )
        # In case the Vision model or the Language model has been offloaded to CPU, we need to manually
        # set the corresponding tensors into their correct target device.
        target_device = inputs_embeds.device
        batch_indices, non_image_indices, text_to_overwrite = (
            batch_indices.to(target_device),
            non_image_indices.to(target_device),
            text_to_overwrite.to(target_device),
        )
        attention_mask = attention_mask.to(target_device)

        # 4. Fill the embeddings based on the mask. If we have ["hey" "<image>", "how", "are"]
        # we need to index copy on [0, 577, 578, 579] for the text and [1:576] for the image features
        final_embedding[batch_indices, text_to_overwrite] = inputs_embeds[batch_indices, non_image_indices]
        final_attention_mask[batch_indices, text_to_overwrite] = attention_mask[batch_indices, non_image_indices]
        if labels is not None:
            final_labels[batch_indices, text_to_overwrite] = labels[batch_indices, non_image_indices]

        # 5. Fill the embeddings corresponding to the images. Anything that is not `text_positions` needs filling (#29835)
        image_to_overwrite = torch.full(
            (batch_size, max_embed_dim), True, dtype=torch.bool, device=inputs_embeds.device
        )
        image_to_overwrite[batch_indices, text_to_overwrite] = False
        image_to_overwrite &= image_to_overwrite.cumsum(-1) - 1 >= nb_image_pad[:, None].to(target_device)

        if image_to_overwrite.sum() != image_features.shape[:-1].numel():
            raise ValueError(
                f"The input provided to the model are wrong. The number of image tokens is {torch.sum(special_image_token_mask)} while"
                f" the number of image given to the model is {num_images}. This prevents correct indexing and breaks batch generation."
            )

        final_embedding[image_to_overwrite] = image_features.contiguous().reshape(-1, embed_dim).to(target_device)
        final_attention_mask |= image_to_overwrite
        position_ids = (final_attention_mask.cumsum(-1) - 1).masked_fill_((final_attention_mask == 0), 1)

        # 6. Mask out the embedding at padding positions, as we later use the past_key_value value to determine the non-attended tokens.
        batch_indices, pad_indices = torch.where(input_ids == self.pad_token_id)
        indices_to_mask = new_token_positions[batch_indices, pad_indices]

        final_embedding[batch_indices, indices_to_mask] = 0

        if labels is None:
            final_labels = None

        return final_embedding, final_attention_mask, final_labels, position_ids

    def forward(
        self,
        media: torch.Tensor,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor = None,
        inference_params: InferenceParams = None,
    ) -> torch.Tensor:
        use_inference_kv_cache = (
            inference_params is not None
            and "image_tokens_count" in inference_params.key_value_memory_dict
        )
        # If running inference, we can skip image token computation if they were computed already earlier for this sample.
        if use_inference_kv_cache:
            image_embeddings = None
        elif self.vision_model is not None:
            image_embeddings = self.vision_model(media)  # [b, img_seq_len, h_vision]
            if self._drop_vision_class_token:
                image_embeddings = image_embeddings[:, self.vision_model.class_token_len :, :]
            # contiguous() call required as `permute` can sparsify the tensor and this breaks pipelining
            image_embeddings = image_embeddings.permute(
                1, 0, 2
            ).contiguous()  # [img_seq_len, b, h_vision]
            # map vision model output size to language model input size.
            image_embeddings = self.vision_projection(
                image_embeddings
            )  # [img_seq_len, b, h_vision]

            # If running inference, the language model KV cache will be updated for image token positions.
            # Here we store the image tokens sequence length, which can be used as an offset to the KV cache later.
            if inference_params is not None:
                inference_params.key_value_memory_dict["image_tokens_count"] = (
                    image_embeddings.shape[0]
                )
        else:
            image_embeddings = self.encoder_hidden_state

        if not self.add_decoder:
            return image_embeddings

        if self.pre_process:
            language_embeddings = self.language_model.embedding(
                input_ids=input_ids, position_ids=position_ids
            )  # [text_seq_len, b, h_language]

            # If running inference, we can skip image token computation if they were computed already earlier for this sample.
            if use_inference_kv_cache:
                combined_embeddings = language_embeddings
            else:
                combined_embeddings = torch.cat(
                    [
                        language_embeddings[: self.img_embedding_idx],
                        image_embeddings,
                        language_embeddings[self.img_embedding_idx :],
                    ],
                    dim=0,
                )  # [combined_seq_len, b, h_language]
                image_embeddings, language_embeddings, input_ids, attention_mask, labels = \
                self._merge_input_ids_with_image_features(
                    image_embeddings, language_embeddings, input_ids, attention_mask, labels
                )
        else:
            combined_embeddings = None

        output = self.language_model(
            input_ids=None,
            position_ids=None,
            attention_mask=attention_mask,
            decoder_input=combined_embeddings,
            labels=labels,
            inference_params=inference_params,
        )

        return output


class NevaModel(L.LightningModule, io.IOMixin, io.ConnectorMixin, fn.FNMixin):
    def __init__(
            self,
            config: NevaConfig,
            # TODO: Add transformer_layer_spec when we update mcore
            optim: Optional[OptimizerModule] = None,
            tokenizer: Optional["TokenizerSpec"] = None,
            image_processor: Optional = None,  # TODO(yuya): add class type
            model_transform: Optional[Callable[[nn.Module], nn.Module]] = None,
    ):
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer
        self.image_processor = image_processor
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
            attention_mask: torch.Tensor,
            labels: torch.Tensor = None,
            inference_params: InferenceParams = None,
    ) -> torch.Tensor:
        output_tensor = self.module(
            media,
            input_ids,
            position_ids,
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
