from dataclasses import dataclass
from typing import Callable, Dict, Optional, Union

import lightning.pytorch as L
import numpy as np
import torch
import torch.distributed
import torch.nn.functional as F
from megatron.core.enums import ModelType
from megatron.core.extensions.transformer_engine import TENorm
from megatron.core.models.gpt import GPTModel as MCoreGPTModel
from megatron.core.models.vision.clip_vit_model import CLIPViTModel as MCoreCLIPViTModel
from megatron.core.optimizer import OptimizerConfig
from megatron.core.transformer import MegatronModule
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.transformer_config import TransformerConfig
from torch import nn

from nemo.collections.llm import fn
from nemo.collections.llm.gpt.model import transformer_engine_layer_spec
from nemo.collections.llm.gpt.model.base import default_layer_spec
from nemo.collections.vlm.clip.loss.clip_loss import ClipMegatronLoss
from nemo.lightning import MegatronOptimizerModule, OptimizerModule, get_vocab_size, io
from nemo.utils import logging


def clip_forward_step(model, batch) -> torch.Tensor:
    forward_args = {"images": batch["images"], "captions": batch["captions"]}
    return model(**forward_args)


def set_input_tensor(self, tensor):
    pass


@dataclass
class CLIPViTConfig(TransformerConfig, io.IOMixin):
    output_dim: int = 1024  # Getting this default from megatron_clip_VIT-H-14.yaml
    ln_final_impl: Union[ModuleSpec, type] = TENorm
    ln_pre_impl: Union[ModuleSpec, type] = TENorm
    ln_post_impl: Union[ModuleSpec, type] = TENorm
    add_class_token: bool = True
    class_token_len: int = 8
    patch_dim: int = 16
    img_h: int = 224
    img_w: int = 224
    vision_model_type: str = "clip"  # ["clip", "siglip"]
    transformer_layer_spec: ModuleSpec = transformer_engine_layer_spec
    gated_linear_unit = False

    # Without these the init for transformer will give error
    num_layers: int = 1  # Placeholder, NOT used!
    num_attention_heads: int = 8  # Placeholder, NOT used!

    def configure_model(self) -> "CLIPViTModel":
        transformer_layer_spec = self.transformer_layer_spec
        if not isinstance(transformer_layer_spec, ModuleSpec):
            from nemo.collections.vlm.layer_specs import get_layer_spec_te

            transformer_layer_spec = get_layer_spec_te(is_vit=True)

        return CLIPViTModel(
            self,
            transformer_layer_spec=transformer_layer_spec,
            ln_pre_impl=self.ln_pre_impl,
            ln_post_impl=self.ln_post_impl,
            add_class_token=self.add_class_token,
            class_token_len=self.class_token_len,
            patch_dim=self.patch_dim,
            img_h=self.img_h,
            img_w=self.img_w,
            model_subtype=self.vision_model_type,
            output_dim=self.output_dim,
            ln_final_impl=self.ln_final_impl,
        )


class CLIPViTModel(MCoreCLIPViTModel):
    def __init__(
        self,
        transformer_config: TransformerConfig,
        transformer_layer_spec: ModuleSpec,
        ln_pre_impl: Union[ModuleSpec, type] = TENorm,
        ln_post_impl: Union[ModuleSpec, type] = TENorm,
        add_class_token: bool = True,
        class_token_len: int = 8,
        patch_dim: int = 16,
        img_h: int = 224,
        img_w: int = 224,
        model_subtype: str = "clip",
        output_dim: int = 1024,
        ln_final_impl: Union[ModuleSpec, type] = TENorm,
    ):
        # TODO (yuya): need to handle post_process correctly in order to enable PP
        self.output_dim = output_dim
        super().__init__(
            transformer_config=transformer_config,
            transformer_layer_spec=transformer_layer_spec,
            ln_pre_impl=ln_pre_impl,
            ln_post_impl=ln_post_impl,
            add_class_token=add_class_token,
            class_token_len=class_token_len,
            patch_dim=patch_dim,
            img_h=img_h,
            img_w=img_w,
            model_subtype=model_subtype,
        )

        self.final_layernorm = build_module(
            ln_final_impl,
            config=self.config,
            hidden_size=self.config.hidden_size,
            eps=self.config.layernorm_epsilon,
        )

        self.head = torch.nn.Linear(
            self.config.hidden_size,
            self.output_dim,
            bias=False,
        )

    def set_input_tensor(self, tensor):
        pass

    def forward(self, x):
        x = super().forward(
            x,
        )
        x = self.final_layernorm(x)
        x = x[:, 0]
        x = self.head(x)
        return x


@dataclass
class CLIPTextModelConfig(TransformerConfig, io.IOMixin):
    output_dim: int = 1024
    ln_final_impl: Union[ModuleSpec, type] = TENorm
    make_vocab_size_divisible_by: int = 128
    max_seq_length: int = 1024
    # TODO: ask Yao if he knows defaults for this. Actually this might not matter?
    share_embeddings_and_output_weights: bool = False
    # Copied from gpt/base model
    gated_linear_unit = False
    use_transformer_engine_full_layer_spec: bool = False
    transformer_layer_spec: Union[ModuleSpec, Callable[["GPTConfig"], ModuleSpec]] = default_layer_spec

    # Without these the init for transformer will give error

    def configure_model(self, tokenizer, pre_process=None, post_process=None) -> "CLIPTextModel":

        transformer_layer_spec = self.transformer_layer_spec
        if not isinstance(transformer_layer_spec, ModuleSpec):
            transformer_layer_spec = transformer_layer_spec(self)

        if hasattr(self, 'vocab_size'):
            vocab_size = self.vocab_size
            logging.info(
                f"Use preset vocab_size: {vocab_size}, original vocab_size: {tokenizer.vocab_size}, dummy tokens:"
                f" {vocab_size - tokenizer.vocab_size}."
            )
        else:
            vocab_size = get_vocab_size(self, tokenizer.vocab_size, self.make_vocab_size_divisible_by)

        return CLIPTextModel(
            transformer_config=self,
            transformer_layer_spec=transformer_layer_spec,
            vocab_size=vocab_size,
            max_sequence_length=self.max_seq_length,
            pre_process=pre_process,
            post_process=post_process,
            output_dim=self.output_dim,
            ln_final_impl=self.ln_final_impl,
            share_embeddings_and_output_weights=self.share_embeddings_and_output_weights,
        )


class CLIPTextModel(MCoreGPTModel):
    def __init__(
        self,
        transformer_config: TransformerConfig,
        transformer_layer_spec: ModuleSpec,
        vocab_size: int,
        max_sequence_length: int,
        pre_process: bool = True,
        post_process: bool = True,
        output_dim: int = 1024,
        ln_final_impl: Union[ModuleSpec, type] = TENorm,
        share_embeddings_and_output_weights: bool = False,
    ):
        # TODO (yuya): need to handle post_process correctly in order to enable PP
        # TODO: I give post_process as false to get hidden states instead of logits
        # What's the right way to do this?
        self.output_dim = output_dim

        super().__init__(
            transformer_config,
            transformer_layer_spec,
            vocab_size,
            max_sequence_length,
            True,
            False,
            share_embeddings_and_output_weights,
        )
        self.final_layernorm = build_module(
            ln_final_impl,
            config=self.config,
            hidden_size=self.config.hidden_size,
            eps=self.config.layernorm_epsilon,
        )

        self.head = torch.nn.Linear(
            self.config.hidden_size,
            self.output_dim,
            bias=False,
        )
        self.position_ids = None
        if self.pre_process:
            self.position_ids = torch.arange(self.max_sequence_length).expand(1, -1).cuda()

    def forward(self, input_ids):
        x = super().forward(input_ids, position_ids=self.position_ids, attention_mask=None)
        x = self.final_layernorm(x)
        x = x[input_ids.argmax(dim=-1), torch.arange(x.shape[1])]
        x = self.head(x)
        return x

    def set_input_tensor(self, tensor):
        pass


@dataclass
class ClipConfig(TransformerConfig, io.IOMixin):
    text_transformer_config: Optional[CLIPTextModelConfig] = None
    vision_transformer_config: Optional[CLIPViTConfig] = None
    get_attention_mask_from_fusion: bool = True
    forward_step_fn: Callable = clip_forward_step

    # Without these the init for transformer will give error
    num_layers: int = 1  # Placeholder, NOT used!
    num_attention_heads: int = 8  # Placeholder, NOT used!
    hidden_size: int = 768  # Placeholder, NOT used!

    def configure_model(self, tokenizer, pre_process=True, post_process=True):
        print(self.kv_channels)
        return MCoreClipModel(
            self,
            tokenizer=tokenizer,
            pre_process=pre_process,
            post_process=post_process,
        )


class MCoreClipModel(MegatronModule):

    def __init__(self, config: ClipConfig, tokenizer, pre_process=True, post_process=True) -> None:
        super().__init__(config=config)
        self.pre_process = pre_process
        self.post_process = post_process
        vision_transformer_config = config.vision_transformer_config
        text_transformer_config = config.text_transformer_config
        self.output_dim = config.vision_transformer_config.output_dim
        self.vision_model = vision_transformer_config.configure_model()
        self.text_model = text_transformer_config.configure_model(
            tokenizer=tokenizer, pre_process=pre_process, post_process=post_process
        )
        self.logit_scale = torch.nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.model_type = ModelType.encoder_or_decoder

    def forward(self, images: torch.Tensor, captions: torch.Tensor):
        image_features = self.vision_model(images)
        text_features = self.text_model(captions)
        if self.post_process:
            return F.normalize(image_features, dim=-1), F.normalize(text_features, dim=-1), self.logit_scale.exp()

        return image_features, text_features

    def set_input_tensor(self, tensor):
        pass


class CLIPModel(L.LightningModule, io.IOMixin, io.ConnectorMixin, fn.FNMixin):
    def __init__(
        self,
        config: ClipConfig,
        optim: Optional[OptimizerModule] = None,
        tokenizer: Optional["TokenizerSpec"] = None,
        model_transform: Optional[Callable[[nn.Module], nn.Module]] = None,
    ):
        # I feel like only use of tokenizer we are doing here is to get the vocab size which can be removed

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

    def forward(self, images: torch.Tensor, captions: torch.Tensor):
        return self.module(images, captions)

    def data_step(self, dataloader_iter) -> Dict[str, torch.Tensor]:
        batch = next(dataloader_iter)
        # TODO: I have no odea why we have 3 inputs? Are the preprocess and post ?
        if isinstance(batch, tuple) and len(batch) == 3:
            _batch = batch[0]
        else:
            _batch = batch
        _batch = {key: val.cuda(non_blocking=True) if val is not None else None for key, val in _batch.items()}
        return _batch
        #     _batch: dict
        # TODO (make data_step_fn)
        # return self.config.data_step_fn(dataloader_iter)

    def forward_step(self, batch) -> torch.Tensor:
        return self.config.forward_step_fn(self, batch)

    def training_step(self, batch, batch_idx=None) -> torch.Tensor:
        # In mcore the loss-function is part of the forward-pass (when labels are provided)
        return self.forward_step(batch)

    def validation_step(self, batch, batch_idx=None) -> torch.Tensor:
        # In mcore the loss-function is part of the forward-pass (when labels are provided)

        return self.forward_step(batch)

    @property
    def training_loss_reduction(self) -> ClipMegatronLoss:
        if not self._training_loss_reduction:
            self._training_loss_reduction = ClipMegatronLoss()

        return self._training_loss_reduction

    @property
    def validation_loss_reduction(self) -> ClipMegatronLoss:
        if not self._validation_loss_reduction:
            self._validation_loss_reduction = ClipMegatronLoss()

        return self._validation_loss_reduction
