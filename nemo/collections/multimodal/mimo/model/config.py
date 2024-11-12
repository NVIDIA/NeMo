import logging
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import torch
import torch.nn.functional as F
from megatron.core import dist_checkpointing
from megatron.core.transformer.custom_layers.transformer_engine import TEColumnParallelLinear, TERowParallelLinear
from megatron.core.transformer.mlp import MLP, MLPSubmodules
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_config import TransformerConfig

from nemo.collections.llm import Llama2Config7B, LlamaConfig
from nemo.collections.llm.gpt.model import transformer_engine_layer_spec
from nemo.collections.llm.gpt.model.base import get_batch_on_this_context_parallel_rank
from nemo.collections.multimodal.mimo.model import CustomMimoModel
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


@dataclass
class BaseInputProjectorConfig(TransformerConfig, io.IOMixin):
    projector_type: str = "mlp2x_gelu"
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
    num_attention_heads: int = 8  # placeholder, NOT used!


@dataclass
class BaseOutputProjectorConfig(TransformerConfig, io.IOMixin):
    # projector_type: str = "mlp2x_gelu" # not needed
    input_size: Optional[int] = 4096  # verify to hidden dimension of language model
    hidden_size: int = 1024
    ffn_hidden_size: int = 1024
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
    num_attention_heads: int = 8  # placeholder, NOT used!


@dataclass
class BaseVisionTransformerConfig(TransformerConfig, io.IOMixin):
    num_layers: int = 24
    num_attention_heads: int = 16  # was 32?
    add_bias_linear: bool = True
    add_qkv_bias: bool = True
    hidden_size: int = 1024
    hidden_dropout: int = 0.0
    attention_dropout: float = 0.0
    ffn_hidden_size: int = 4096
    gated_linear_unit: bool = False
    # activation_func = quick_gelu
    # kv_channels: int = 64
    # num_query_groups: int = 16
    layernorm_zero_centered_gamma: bool = False
    apply_query_key_layer_scaling: bool = False  # TODO: Yash Check this
    bias_activation_fusion: bool = False
    bias_dropout_fusion: bool = False
    attention_softmax_in_fp32: bool = True
    normalization = 'LayerNorm'
    layer_spec: ModuleSpec = transformer_engine_layer_spec
    img_h: int = 336
    img_w: int = 336
    patch_dim: int = 14
    vision_model_type = 'clip'


@dataclass
class Llama2Config1B(LlamaConfig):
    num_layers: int = 1
    hidden_size: int = 1024
    num_attention_heads: int = 1
    num_query_groups: int = 1
    ffn_hidden_size: int = 1024


@dataclass
class CustomMimoConfig(TransformerConfig, io.IOMixin):
    language_transformer_config: Optional[TransformerConfig] = field(default_factory=lambda: Llama2Config7B())
    vision_transformer_config: Optional[TransformerConfig] = field(
        default_factory=lambda: BaseVisionTransformerConfig()
    )
    vision_projection_config: Optional[TransformerConfig] = field(default_factory=lambda: BaseInputProjectorConfig())

    vision_output_projection_config: Optional[TransformerConfig] = field(
        default_factory=lambda: BaseOutputProjectorConfig()
    )
    freeze_language_model: bool = True
    freeze_vision_model: bool = True
    freeze_vision_projection: bool = False

    forward_step_fn: Callable = mimo_forward_step
    data_step_fn: Callable = mimo_data_step

    vocab_size: Optional[int] = None
    num_layers: int = 1  # placeholder, NOT used!
    num_attention_heads: int = 8  # placeholder, NOT used!
    image_special_tokens: Optional[List[str]] = None
    image_special_token_indices: Optional[List[int]] = None
    make_vocab_size_divisible_by: int = 128
    freeze_language_model: bool = True
    freeze_vision_model: bool = True
    freeze_vision_projection: bool = True

    def configure_model(self, tokenizer) -> "CustomMimoModel":

        self.vocab_size = get_vocab_size(self, tokenizer.vocab_size, self.make_vocab_size_divisible_by)
        logging.info(f"padded vocab size to {self.vocab_size}")

        model = CustomMimoModel(
            model_config=self,
            language_transformer_config=self.language_transformer_config,
            language_transformer_layer_spec=transformer_engine_layer_spec(self.language_transformer_config),
            language_vocab_size=self.vocab_size,
            language_max_sequence_length=self.language_transformer_config.seq_length,
            vision_transformer_config=self.vision_transformer_config,
            vision_transformer_layer_spec=transformer_engine_layer_spec(self.vision_transformer_config),
            drop_vision_class_token=True,
            vision_projection_config=self.vision_projection_config,
            vision_projection_layer_spec=self.vision_projection_config.layer_spec,
            vision_output_projection_config=self.vision_output_projection_config,
            vision_output_projection_spec=self.vision_output_projection_config.layer_spec,
            vision_projection_type="mlp",
            allow_missing_vision_projection_checkpoint=True,
            parallel_output=True,
            pre_process=True,
            post_process=True,
            add_encoder=True,
            add_decoder=True,
            img_h=self.vision_transformer_config.img_h,
            img_w=self.vision_transformer_config.img_w,
            patch_dim=self.vision_transformer_config.patch_dim,
        )
        from megatron.core.dist_checkpointing.validation import StrictHandling

        sharded_state_dict = dict(state_dict=model.language_model.sharded_state_dict(prefix="module."))
        if torch.distributed.get_rank() == 0:  # or other ranks
            breakpoint()
        torch.distributed.barrier()
        strict = StrictHandling.LOG_UNEXPECTED
        loaded_state_dict = dist_checkpointing.load(
            sharded_state_dict=sharded_state_dict,
            checkpoint_dir='/root/.cache/nemo/models/lmsys/vicuna-7b-v1.5/weights',
            strict=strict,
        )
        loaded_state_dict = {k.removeprefix("module."): v for k, v in loaded_state_dict["state_dict"].items()}
        if torch.distributed.get_rank() == 0:  # or other ranks
            breakpoint()
        torch.distributed.barrier()
        model.language_model.load_state_dict(loaded_state_dict)

        model.freeze(
            freeze_language_model=self.freeze_language_model,
            freeze_vision_model=self.freeze_vision_model,
            freeze_vision_projection=self.freeze_vision_model,
        )
        return model
