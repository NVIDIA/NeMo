import logging
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import torch
import torch.nn.functional as F
from megatron.core import dist_checkpointing
from megatron.core.models.vision.clip_vit_model import CLIPViTModel as MCoreCLIPViTModel
from megatron.core.models.vision.multimodal_projector import MultimodalProjector as MCoreMultimodalProjector
from megatron.core.transformer.custom_layers.transformer_engine import (
    TEColumnParallelLinear,
    TENorm,
    TERowParallelLinear,
)
from megatron.core.transformer.mlp import MLP, MLPSubmodules
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_config import TransformerConfig

from nemo.collections.llm import Llama2Config7B, LlamaConfig
from nemo.collections.llm.gpt.model import transformer_engine_layer_spec
from nemo.collections.llm.gpt.model.base import get_batch_on_this_context_parallel_rank
from nemo.collections.multimodal.mimo.model.model import MimoModel
from nemo.collections.multimodal.mimo.model.projection import ImageOutputProjectionPoolingHead
from nemo.lightning import get_vocab_size, io


def mimo_forward_step(model, batch) -> torch.Tensor:
    forward_args = {
        "images": batch["images"],
        "output_images": batch["output_images"],
        "input_ids": batch["tokens"],
        "position_ids": batch["position_ids"],
        "attention_mask": batch.get("attention_mask", None),
        "loss_mask": batch.get("loss_mask", None),
        "labels": batch.get("labels", None),
        "input_text": batch.get("input_text", None),
    }
    # loss_mask = batch.get("loss_mask", None)
    output_dict = model(**forward_args)
    return output_dict
    # return model(**forward_args), loss_mask


def mimo_data_step(dataloader_iter) -> Dict[str, torch.Tensor]:
    from megatron.core import parallel_state

    batch = next(dataloader_iter)

    _batch: dict
    if isinstance(batch, tuple) and len(batch) == 3:
        _batch = batch[0]
    else:
        _batch = batch

    required_keys = set()
    required_keys.add("attention_mask")
    if parallel_state.is_pipeline_first_stage():
        required_keys.update(("images", "tokens", "position_ids", "input_text", "output_images"))
    if parallel_state.is_pipeline_last_stage():
        required_keys.update(("labels", "loss_mask", "input_text"))

    _batch = {
        key: (
            (val.cuda(non_blocking=True) if hasattr(val, "cuda") else val)
            if key in required_keys and val is not None
            else None
        )
        for key, val in _batch.items()
    }
    # slice batch along sequence dimension for context parallelism
    output = get_batch_on_this_context_parallel_rank(_batch)

    return output


def get_output_projection_layer_spec() -> ModuleSpec:
    output_projection_submodules = TransformerLayerSubmodules(
        cross_attention=ModuleSpec(
            module=CrossAttention,
            params={"attn_mask_type": MCoreAttnMaskType.no_mask},
            submodules=CrossAttentionSubmodules(
                linear_q=TEColumnParallelLinear,
                linear_kv=TEColumnParallelLinear,
                core_attention=TEDotProductAttention,
                linear_proj=TERowParallelLinear,
            ),
        ),
        cross_attn_bda=get_bias_dropout_add,
        mlp=ModuleSpec(
            module=MLP,
            submodules=MLPSubmodules(
                linear_fc1=TELayerNormColumnParallelLinear,
                linear_fc2=TERowParallelLinear,
            ),
        ),
        mlp_bda=get_bias_dropout_add,
    )
    output_projection_submodules.output_linear_layer = ColumnParallelLinear  # TEColumnParallelLinear

    return ModuleSpec(module=ImageOutputProjectionPoolingHead, submodules=output_projection_submodules)


@dataclass
class ImageInputProjectionConfig(TransformerConfig, io.IOMixin):
    projector_type: str = "mlp"
    input_size: Optional[int] = 1024
    hidden_size: int = 4096
    ffn_hidden_size: int = 4096
    activation_func: Callable = F.gelu
    bias: bool = True
    bias_activation_fusion: bool = True
    add_bias_linear: bool = True
    layer_spec: ModuleSpec = ModuleSpec(
        module=MLP,
        submodules=MLPSubmodules(
            linear_fc1=TEColumnParallelLinear,
            linear_fc2=TERowParallelLinear,
        ),
    ).submodules
    num_layers: int = 1  # placeholder, NOT used!
    num_attention_heads: int = 8  # placeholder, NOT used!'

    def configure_model(self) -> "MCoreMultimodalProjector":
        return MCoreMultimodalProjector(
            self,
            self.layer_spec,
            self.projector_type,
            input_size=self.input_size,  # input size to the projection.
        )


@dataclass
class ImageOutputProjectionConfig(TransformerConfig, io.IOMixin):
    num_layers: int = 2
    num_attention_heads: int = 16  # was 32?
    num_query_token: int = 77
    add_bias_linear: bool = True
    add_qkv_bias: bool = True
    hidden_size: int = 4096
    hidden_dropout: int = 0.0
    attention_dropout: float = 0.0
    ffn_hidden_size: int = 4096
    gated_linear_unit: bool = False
    layernorm_zero_centered_gamma: bool = False
    apply_query_key_layer_scaling: bool = False  # TODO: Yash Check this
    bias_activation_fusion: bool = False
    bias_dropout_fusion: bool = False
    attention_softmax_in_fp32: bool = True
    normalization = 'LayerNorm'
    layer_spec: ModuleSpec = get_output_projection_layer_spec()

    def configure_model(self):
        return ImageOutputProjectionPoolingHead(
            config=self,
            submodules=self.layer_spec.submodules,
            num_query_token=self.num_query_token,
        )


@dataclass
class ImageEncoderTransformerConfig(TransformerConfig, io.IOMixin):
    vision_model_type = 'clip'
    drop_vision_class_token: bool = True
    add_bias_linear: bool = True
    add_qkv_bias: bool = True
    hidden_dropout: int = 0.0
    attention_dropout: float = 0.0
    gated_linear_unit: bool = False
    layernorm_zero_centered_gamma: bool = False
    apply_query_key_layer_scaling: bool = False  # TODO: Yash Check this
    bias_activation_fusion: bool = False
    bias_dropout_fusion: bool = False
    attention_softmax_in_fp32: bool = True
    normalization = 'LayerNorm'
    apply_rope_fusion = False
    layer_spec: ModuleSpec = (
        transformer_engine_layer_spec  # TODO: Yash change this with colletions.vlm.layer_spec after rebase
    )
    img_h: int = 336
    img_w: int = 336
    patch_dim: int = 14

    ln_pre_impl: Union[ModuleSpec, type] = TENorm
    ln_post_impl: Union[ModuleSpec, type] = TENorm

    def __post_init__(self):
        if self.vision_model_type == "siglip":
            self.num_layers = 27
            self.num_attention_heads = 16
            self.hidden_size = 1152
            self.ffn_hidden_size = 4304
            self.activation_func = torch.nn.functional.gelu  # Mcore uses fast_gelu
            self.kv_channels = 72
            self.num_query_groups = 16
            self.qk_layernorm = False
            self.layernorm_epsilon = 1e-6
            self.add_class_token: bool = False
            self.class_token_len: int = 0
        elif self.vision_model_type == "clip":
            self.num_layers: int = 24
            self.num_attention_heads: int = 16
            self.hidden_size: int = 1024
            self.ffn_hidden_size: int = 4096
            self.kv_channels: int = 64
            self.num_query_groups: int = 16
            self.activation_func = torch.nn.functional.gelu  # Mcore uses fast_gelu
            self.add_class_token: bool = True
            self.class_token_len: int = 1
        else:
            raise NotImplementedError(f"Vision model type {self.vision_model_type} not implemented")
        # TODO: Yash add config for internvit

    def configure_model(self) -> "MCoreCLIPViTModel":

        return MCoreCLIPViTModel(
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


class ImageDecoderTransformerConfig(io.IOMixin):
    image_decoder_name = "stabilityai/stable-diffusion-2"

    def configure_model(self):
        from diffusers import EulerDiscreteScheduler, StableDiffusionPipeline

        scheduler = EulerDiscreteScheduler.from_pretrained(image_decoder_name, subfolder="scheduler")
        image_decoder = StableDiffusionPipeline.from_pretrained(image_decoder_name, scheduler=scheduler)

        image_decoder.vae.requires_grad_(False)
        image_decoder.unet.requires_grad_(False)
        image_decoder.text_encoder.requires_grad_(False)
        return image_decoder


@dataclass
class MimoConfig(TransformerConfig, io.IOMixin):

    language_transformer_config: Optional[TransformerConfig] = field(default_factory=lambda: Llama2Config7B())

    image_encoder_transformer_config: Optional[TransformerConfig] = field(
        default_factory=lambda: ImageEncoderTransformerConfig()
    )
    image_decoder_transformer_config = field(default_factory=lambda: ImageDecoderTransformerConfig())

    image_input_projection_config: Optional[TransformerConfig] = field(
        default_factory=lambda: ImageInputProjectionConfig()
    )
    image_output_projection_config: Optional[TransformerConfig] = field(
        default_factory=lambda: ImageOutputProjectionConfig()
    )

    freeze_language_model: bool = True
    freeze_image_encoder: bool = True
    freeze_image_input_projection: bool = False
    freeze_image_output_projection: bool = False

    forward_step_fn: Callable = mimo_forward_step
    data_step_fn: Callable = mimo_data_step

    vocab_size: Optional[int] = None
    num_layers: int = 1  # placeholder, NOT used!
    num_attention_heads: int = 8  # placeholder, NOT used!
    image_special_tokens: Optional[List[str]] = None
    image_special_token_indices: Optional[List[int]] = None
    make_vocab_size_divisible_by: int = 128
    parallel_output: bool = True
    rotary_percent: float = 1.0
    rotary_base: int = 1000000
    rope_scaling: bool = False

    load_vision_mlp_language_model_path: Optional[str] = None
    load_language_model_path: Optional[str] = None
    load_vision_model_path: Optional[str] = None
    load_mlp_projector_path: Optional[str] = None

    stage: str = "encoder_alignment"  # [encoder_alignment, decoder_alignment, 'interleaved_pretrain']

    def configure_model(self, tokenizer) -> "MimoModel":

        self.vocab_size = get_vocab_size(self, tokenizer.vocab_size, self.make_vocab_size_divisible_by)
        logging.info(f"padded vocab size to {self.vocab_size}")

        model = MiMoModel(config=self)
        # from megatron.core.dist_checkpointing.validation import StrictHandling

        # from nemo.lightning.io.pl import ckpt_to_weights_subdir

        # if self.load_language_model_path:
        #     sharded_state_dict = dict(state_dict=model.language_model.sharded_state_dict(prefix="module."))

        #     strict = StrictHandling.LOG_UNEXPECTED
        #     loaded_state_dict = dist_checkpointing.load(
        #         ckpt_to_weights_subdir(self.load_language_model_path),
        #         sharded_state_dict=sharded_state_dict,
        #         checkpoint_dir='load_language_model_path',
        #         strict=strict,
        #     )
        #     loaded_state_dict = {k.removeprefix("module."): v for k, v in loaded_state_dict["state_dict"].items()}

        #     model.language_model.load_state_dict(loaded_state_dict)

        # model.freeze(
        #     freeze_language_model=self.freeze_language_model,
        #     freeze_vision_model=self.freeze_vision_model,
        #     freeze_vision_projection=self.freeze_vision_model,
        # )
        return model
