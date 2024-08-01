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
        required_keys.update(("images", "tokens", "position_ids"))
    if parallel_state.is_pipeline_last_stage():
        required_keys.update(("labels", "loss_mask"))

    _batch = {key: val.cuda(non_blocking=True) if key in required_keys else None for key, val in _batch.items()}
    # slice batch along sequence dimension for context parallelism
    output = get_batch_on_this_context_parallel_rank(_batch)

    return output


def neva_forward_step(model, batch) -> torch.Tensor:
    forward_args = {
        "images": batch["images"],
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

    def configure_model(self) -> "MCoreMultimodalProjector":
        layer_spec: ModuleSpec = MLPSubmodules
        projector_type: str
        input_size: int
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
        pass

    def forward(
        self,
        images: torch.Tensor,
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
            image_embeddings = self.vision_model(images)  # [b, img_seq_len, h_vision]
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
            images: torch.Tensor,
            input_ids: torch.Tensor,
            position_ids: torch.Tensor,
            attention_mask: torch.Tensor,
            labels: torch.Tensor = None,
            inference_params: InferenceParams = None,
    ) -> torch.Tensor:
        output_tensor = self.module(
            images,
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
