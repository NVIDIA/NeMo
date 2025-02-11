import os
import re
import time
from dataclasses import dataclass
from functools import partial
from typing import Callable, Dict, List, Optional, Union, Any, Tuple

from lightning.fabric.utilities import measure_flops
from lightning.pytorch.utilities.types import STEP_OUTPUT
from megatron.core import InferenceParams
import lightning.pytorch as L
import torch
import timm
import torch.distributed
import torch.nn.functional as F
from megatron.core import dist_checkpointing
from megatron.core import parallel_state as ps
from megatron.core import tensor_parallel
from megatron.core.enums import ModelType
from megatron.core.models.multimodal.llava_model import LLaVAModel as MCoreLLaVAModel
from megatron.core.models.vision.clip_vit_model import CLIPViTModel as MCoreCLIPViTModel
from megatron.core.models.vision.multimodal_projector import MultimodalProjector as MCoreMultimodalProjector
from megatron.core.optimizer import OptimizerConfig
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.tensor_parallel import gather_from_sequence_parallel_region
from megatron.core.transformer.custom_layers.transformer_engine import (
    TEColumnParallelLinear,
    TENorm,
    TERowParallelLinear,
)
from megatron.core.transformer.mlp import MLP, MLPSubmodules
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_config import TransformerConfig
from timm.models.vision_transformer import LayerScale
from torch import nn
from torch.utils.flop_counter import FlopCounterMode
from transformers import CLIPVisionConfig, CLIPVisionModel

from nemo.collections.common.tokenizers.tokenizer_spec import TokenizerSpec
from nemo.collections.llm import fn
from nemo.collections.llm.gpt.model import transformer_engine_layer_spec
from nemo.collections.llm.gpt.model.base import get_batch_on_this_context_parallel_rank, get_packed_seq_params
from nemo.collections.nlp.modules.common.megatron.layer_type import LayerType
from nemo.collections.vlm.neva.data.multimodal_tokens import IGNORE_INDEX, IMAGE_TOKEN_INDEX
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


def openvla_data_step(dataloader_iter) -> Dict[str, torch.Tensor]:
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

    return _batch


def openvla_forward_step(model, batch) -> torch.Tensor:
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


def set_input_tensor(self, tensor):
    pass


@dataclass
class MultimodalProjectorConfig(TransformerConfig, io.IOMixin):
    """
    For MLP, fc1 in shape of input_size, ffn_hidden_size, fc2 in shape of ffn_hidden_size, hidden_size
    """

    num_vision_encoders: int = 2
    projector_type: Optional[str] = "openvla_mlp"
    layer_spec: Optional[MLPSubmodules] = None
    input_size: Optional[int] = 1024
    output_size: int = 1024
    ffn_hidden_size: Optional[int] = 0 # 4 * input_size
    activation_func: Callable = F.gelu
    bias: bool = True
    bf16 : bool = True
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

        assert self.projector_type == "openvla_mlp", "Only mcore projections or openvla_mlp are supported"
        # e.g. "mlp2x_gelu"
        assert self.num_vision_encoders <= 2, "Only 1 or 2 vision encoders supported"
        if self.num_vision_encoders == 1:
            modules = [torch.nn.Linear(self.input_size, self.output_size, bias=True)]
            modules.append(torch.nn.GELU())
            modules.append(torch.nn.Linear(self.output_size, self.output_size, bias=True))
        elif self.num_vision_encoders == 2:
            if self.ffn_hidden_size == 0:
                self.ffn_hidden_size = 4 * self.input_size

            # self.ffn_hidden_size = 0
            modules = [torch.nn.Linear(self.input_size, self.ffn_hidden_size, bias=True)]
            modules.append(torch.nn.GELU())
            modules.append(torch.nn.Linear(self.ffn_hidden_size, self.output_size, bias=True))
            modules.append(torch.nn.GELU())
            modules.append(torch.nn.Linear(self.output_size, self.output_size, bias=True))
        else:
            raise NotImplementedError

        model = torch.nn.Sequential(*modules)

        from types import MethodType
        model.set_input_tensor = MethodType(set_input_tensor, model)
        return model


def unpack_tuple(fn: Callable[[Any], Tuple[Any]]) -> Callable[[Any], Any]:
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        result = fn(*args, **kwargs)
        return result[0] if isinstance(result, tuple) else result

    return wrapper

def _ls_new_forward(self, x: torch.Tensor) -> torch.Tensor:
    return x.mul_(self.scale_factor) if self.inplace else x * self.scale_factor


def ls_apply_patch(ls_module: LayerScale):
    ls_module.scale_factor = nn.Parameter(ls_module.gamma.clone())
    ls_module.forward = _ls_new_forward.__get__(ls_module, LayerScale)
    del ls_module.gamma


@dataclass
class TimmCLIPVisionConfig(io.IOMixin):
    """
    https://github.com/huggingface/transformers/blob/v4.44.0/src/transformers/models/clip/configuration_clip.py#L261
    """

    hidden_size: int = 1024 # It's 1024 and 1152 for us
    pretrained_model_name_or_path: Union[str, os.PathLike] = "vit_large_patch14_reg4_dinov2.lvd142m"
    # This is either     "vit_large_patch14_reg4_dinov2.lvd142m",
    #     "vit_so400m_patch14_siglip_224"
    num_image_embeddings_per_tile: Optional[int] = None # For us there is just 1 tile and this number is 256
    patch_size: int = 14
    image_size: int = 224
    load_weights: bool = True
    embed_dim: Optional[int] = None
    override_act_layer: Optional[LayerType] = None
    dtype: Optional[torch.dtype] = torch.bfloat16

    def __post_init__(self, *args, **kwargs) -> None:
        # import pdb; pdb.set_trace()
        self.num_image_embeddings_per_tile = get_image_sequence_length(
            img_h=self.image_size,
            img_w=self.image_size,
            patch_dim=self.patch_size,
            add_class_token=False,
            class_token_len=1,
        )

        self.featurizer = timm.create_model(
            self.pretrained_model_name_or_path,
            pretrained=False,
            num_classes=0,
            img_size=self.image_size,
            act_layer=self.override_act_layer,
        )


    def configure_model(self) -> "CLIPVisionModel":
        # Monkey patch the method to the vision encoder
        # CLIPVisionModel.set_input_tensor = set_input_tensor

        # import pdb; pdb.set_trace()


        self.featurizer.forward = unpack_tuple(
            partial(self.featurizer.get_intermediate_layers, n={len(self.featurizer.blocks) - 2})
        )

        # Patch `vision_backbone.featurizer` and `vision_backbone.fused_featurizer` with HF-Compatible LayerScale
        for module in self.featurizer.modules():
            if isinstance(module, LayerScale):
                ls_apply_patch(module)

        from types import MethodType
        self.featurizer.set_input_tensor = MethodType(set_input_tensor, self.featurizer)

        self.featurizer = self.featurizer.to(dtype=self.dtype)
        self.featurizer.requires_grad_(False)
        self.featurizer = self.featurizer.to("cuda")
        # import pdb; pdb.set_trace()
        return self.featurizer


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
    bf16: bool = True

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


@dataclass
class OpenVLAConfig(TransformerConfig, io.IOMixin):
    language_transformer_config: Optional[TransformerConfig] = None
    vision_transformer_config: Optional[TransformerConfig] = None
    secondary_vision_transformer_config: Optional[TransformerConfig] = None
    vision_projection_config: Optional[MultimodalProjectorConfig ] = None
    vision_feature_layer: int = -2
    secondary_vision_feature_layer: int = -2

    # TODO(abhi): Not sure what these are
    drop_vision_class_token: bool = True

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
    freeze_secondary_vision_model: bool = False
    freeze_vision_projection: bool = False
    bf16 : bool = True

    forward_step_fn: Callable = openvla_forward_step
    data_step_fn: Callable = openvla_data_step

    def __post_init__(self):
        if self.language_transformer_config is not None:
            for attr in MODEL_CONFIG_ATTR:
                setattr(self, attr, getattr(self.language_transformer_config, attr))

    def configure_model(self, tokenizer) -> "MCoreNevaModel":
        self.language_transformer_config.tensor_model_parallel_size = self.tensor_model_parallel_size
        self.language_transformer_config.sequence_parallel = self.sequence_parallel
        self.vision_transformer_config.tensor_model_parallel_size = self.tensor_model_parallel_size
        self.secondary_vision_transformer_config.tensor_model_parallel_size = self.tensor_model_parallel_size
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

        model = MCoreOpenVLAModel(
            config=self,
            tokenizer=tokenizer,
            pre_process=ps.is_pipeline_first_stage(),
            post_process=ps.is_pipeline_last_stage(),
            add_encoder=ps.is_pipeline_first_stage(),
            add_secondary_encoder=ps.is_pipeline_first_stage() and self.secondary_vision_transformer_config is not None,
            add_decoder=ps.is_pipeline_last_stage()
            or ps.get_pipeline_model_parallel_rank() >= self.encoder_pipeline_model_parallel_size,
            drop_vision_class_token=self.drop_vision_class_token,
        )

        return model

# TODO(abhi): Not sure if inheriting from MCoreNevaModel is correct. We can just make a Megatron Module too.
class MCoreOpenVLAModel(MCoreLLaVAModel):
    def __init__(
        self,
        config: OpenVLAConfig,
        tokenizer: Optional = None,
        pre_process: bool = True,
        post_process: bool = True,
        add_encoder: bool = True,
        add_secondary_encoder: bool = True,
        add_decoder: bool = True,
        drop_vision_class_token: bool = False,
    ) -> None:
        # Init just the Megatron Module
        super(MCoreLLaVAModel, self).__init__(config=config)

        self.config = config
        language_transformer_config = config.language_transformer_config
        vision_transformer_config = config.vision_transformer_config
        secondary_vision_transformer_config = config.secondary_vision_transformer_config
        vision_projection_config = config.vision_projection_config

        self.pre_process = pre_process
        self.post_process = post_process
        self.add_encoder = add_encoder
        self.add_secondary_encoder = add_secondary_encoder
        self.add_decoder = add_decoder

        self.encoder_hidden_state = None
        self.vision_model = None
        self.secondary_vision_model = None
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

        self._drop_vision_class_token = drop_vision_class_token # Vision class token argument is considered
        if self.add_encoder:
            self.vision_model = vision_transformer_config.configure_model()
            self.vision_projection = vision_projection_config.configure_model()

        if self.add_secondary_encoder:
            self.secondary_vision_model = secondary_vision_transformer_config.configure_model()

        # self._freeze(freeze_language_model=config.freeze_language_model, freeze_vision_model=config.freeze_vision_model,
        #             freeze_secondary_vision_model=config.freeze_secondary_vision_model,
        #             freeze_vision_projection=config.freeze_vision_projection)

        self.model_type = ModelType.encoder_or_decoder
        # This attribute is needed to check if an all-reduce is required
        # on the word embeddings inside `finalize_model_grads._allreduce_word_embedding_grads`.

        self.vision_model_from_hf = hasattr(vision_transformer_config, "image_size")
        self.secondary_vision_model_from_hf = hasattr(secondary_vision_transformer_config, "image_size")

        if self.secondary_vision_model_from_hf:
            assert self.vision_model_from_hf == self.secondary_vision_model_from_hf, \
                "Both vision model and secondary vision model should be from HF"

        if self.vision_model_from_hf:
            self._drop_vision_class_token = False

        # TODO(abhi): This is basically the length of image seqeunce per tile. Later this is used to calculate the seq_length for language decoder as following:
        # # Sequence length for each sample is the image sequence length multiplied by
        #             # the number of tiles for that image, minus image token indices,
        #             # plus text sequence length.
        #             seq_lens = num_image_tiles_batch * img_seq_len - num_images_per_sample + text_seq_len
        # I think we can set number of tiles as 1 and img_Seq_len as DInoV2 + SigLip models' output seq lengths or just SigLip models' output seq lengths
        # Depending on which model we are using.
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
                inference_params is not None and "image_tokens_count" in inference_params.key_value_memory_dict
        )
        has_images = images is not None and images.shape[0] > 0
        # images : bsz * 3 * 336 * 336
        # If running inference, we can skip images token computation if they were computed already earlier for this sample.
        if use_inference_kv_cache:
            image_embeddings = None
        elif self.add_encoder and not has_images:
            vision_param = next(self.vision_model.parameters())
            # If no images provided, use an empty image embeddings tensor.
            image_embeddings = torch.tensor([], dtype=vision_param.dtype, device=vision_param.device).reshape(0, 0, 0)
        elif self.add_encoder and (not self.add_secondary_encoder) and has_images:
            assert False, "TODO(Abhi): Need to implement the case where we just have one vision encoder."
            images = images.to(next(self.vision_model.parameters()).dtype)
            img = images # img is in shape of (bsz, 3, h, w) h and 2 are 224 and 224 respectively

            if self.vision_model_from_hf:
                self.vision_model = self.vision_model.eval()
                image_embeddings = self.vision_model(images)  # [1, 256, 1024]
            else:
                # TODO(yuya): MCore Clip path not yet support taking a specific layer hidden states
                image_embeddings = self.vision_model(images, num_unused_layers=-self.config.vision_feature_layer - 1)

        elif self.add_encoder and self.add_secondary_encoder and has_images:
            # Code from OpenVLA HF(Maybe Huy's dataset will be in the same format?)
            #        # Split `pixel_values :: [bsz, 2 * 3, resolution, resolution]` =>> featurize =>> channel stack
            # img, img_fused = torch.split(pixel_values, [3, 3], dim=1)
            # patches, patches_fused = self.featurizer(img), self.fused_featurizer(img_fused)

            # images is in shape of (num_images_in_mbs, c, h, w)
            # note num_images_in_mbs is not mbs but total images in this mbs which is approximately equal to num_tiles*mbs.
            images = images.to(next(self.vision_model.parameters()).dtype)

            # import pdb; pdb.set_trace()
            # Assuming we are getting images in the shape of (mbs, 2*3, h, w)
            img, img_secondary = torch.split(images, [3, 3], dim=1)
            # Img and img_secondary is in shape of (mbs, c, h, w) where c=3
            if self.vision_model_from_hf:
                # self.vision_model = self.vision_model.eval()
                torch.cuda.synchronize()
                torch.distributed.barrier()
                start = time.time()
                image_embeddings = self.vision_model(img)  # [1, 256, 1024]
                torch.cuda.synchronize()
                print(f"Time for primary image encoder: {time.time() - start}")
            else:
                # TODO(yuya): MCore Clip path not yet support taking a specific layer hidden states
                image_embeddings = self.vision_model(img, num_unused_layers=-self.config.vision_feature_layer - 1)

            if self.secondary_vision_model_from_hf:
                # self.secondary_vision_model = self.secondary_vision_model.eval()
                torch.cuda.synchronize()
                torch.distributed.barrier()
                start = time.time()
                image_embeddings_secondary = self.secondary_vision_model(img_secondary) # [1, 256, 1152]
                torch.cuda.synchronize()
                print(f"Time for secondary image encoder: {time.time() - start}")
            else:
                # TOdo(yuya): MCore Clip path not yet support taking a specific layer hidden states
                image_embeddings_secondary = self.secondary_vision_model(
                    img_secondary, num_unused_layers=-self.config.secondary_vision_feature_layer - 1
                )

            # TODO(Abhi): Check if the dim is correct. HF concats along dim 2
            image_embeddings = torch.cat([image_embeddings, image_embeddings_secondary], dim=2)
            # import pdb; pdb.set_trace()
            #print(f"image embedding shape: {image_embeddings.shape}")
            if self._drop_vision_class_token:
                class_token_len = getattr(self.vision_model, "class_token_len", 1)
                image_embeddings = image_embeddings[:, class_token_len:, :]

            # contiguous() required as `permute` can sparsify the tensor and this breaks pipelining
            # Todo(Abhi): Ask Yu why we are doing this
            image_embeddings = image_embeddings.permute(1, 0, 2).contiguous()  # [img_seq_len, num_tiles, h_vision]

            torch.cuda.synchronize()
            torch.distributed.barrier()
            start = time.time()
            # map vision model output size to language model input size.
            image_embeddings = self.vision_projection(image_embeddings)  # [img_seq_len, num_tiles, h_language]
            torch.cuda.synchronize()
            print(f"Time for Vision projection: {time.time() - start}")

            # TODO: Support batched inference.
            # In inference, the language model KV cache will be updated for image token positions.
            # Store the image tokens sequence length to be used as an offset to the KV cache later.
            if inference_params is not None:
                inference_params.key_value_memory_dict["image_tokens_count"] = (
                        image_embeddings.shape[0] * image_embeddings.shape[1]
                )
        else:
            image_embeddings = self.encoder_hidden_state
            # raise ValueError(f"Invalid combination of add_encoder and add_secondary_encoder. "
            #                  f"{self.add_encoder = }, {self.add_secondary_encoder = }")

        if not self.add_decoder:
            return image_embeddings

        language_embeddings = None
        if self.pre_process:
            torch.cuda.synchronize()
            torch.distributed.barrier()
            start = time.time()
            input_ids_text = input_ids.clone()
            #print(f"input_ids_text shape: {input_ids_text.shape}")
            # MultiModal Token indices are assumed to be values
            input_ids_text[input_ids_text < 0] = 0
            # Note: This adds absolute position embedding but not RoPE.
            # Each image is counted as one position.
            # RoPE is added in language_model forward. Each image embedding is one position.
            if self.sequence_parallel_lm:
                # Pad to nearest multiple of TP world size for embedding.
                tp_world_size = ps.get_tensor_model_parallel_world_size()
                padded_seq_len = (
                        int((input_ids_text.shape[1] + tp_world_size - 1) // tp_world_size * tp_world_size)
                        - input_ids_text.shape[1]
                )
                if padded_seq_len != 0:
                    input_ids_text = torch.nn.functional.pad(input_ids_text, (0, padded_seq_len))
                    if position_ids is not None:
                        position_ids = torch.nn.functional.pad(position_ids, (0, padded_seq_len))


            # We shift the position ids for VLA
            # Shift elements starting from index 2 by 255
            shifted_position_ids = position_ids.clone()
            shifted_position_ids[:, 2:] += 255
            language_embeddings = self.language_model.embedding(
                input_ids=input_ids_text, position_ids=shifted_position_ids
            )  # [text_seq_len, b, h_language]
            if self.sequence_parallel_lm:
                # Gather the language embeddings back.
                # We use the full embedding to insert image embeddings
                # and then scatter to avoid load imbalance.
                language_embeddings = gather_from_sequence_parallel_region(
                    language_embeddings, tensor_parallel_output_grad=False
                )
                # Remove the padding done for SP as we'll need new padding calculation
                # after image embeddings are inserted.
                if padded_seq_len != 0:
                    language_embeddings = language_embeddings[:-padded_seq_len]
            language_embeddings = language_embeddings.transpose(1, 0).contiguous()  # [b, text_seq_len, h_language]
            torch.cuda.synchronize()
            print(f"Time for language model: {time.time() - start}")


        #print(f"image embedding shape: {language_embeddings.shape}")
        # Assume 1 tile per image if the number of tiles is not provided.
        if num_image_tiles is None:
            num_image_tiles = torch.ones(images.shape[0], dtype=torch.int, device=input_ids.device)
        elif isinstance(num_image_tiles, list):
            num_image_tiles = torch.tensor(num_image_tiles, dtype=torch.int, device=input_ids.device)

        # import pdb; pdb.set_trace()
        #print(f"image embedding shape Right before: {language_embeddings.shape}")
        # Preprocess input, labels and loss mask.
        torch.cuda.synchronize()
        torch.distributed.barrier()
        start = time.time()
        combined_embeddings, final_labels, final_loss_mask, final_attention_mask = self._preprocess_data(
            image_embeddings=image_embeddings,
            language_embeddings=language_embeddings,
            input_ids=input_ids,
            loss_mask=loss_mask,
            labels=labels,
            use_inference_kv_cache=use_inference_kv_cache,
            image_token_index=image_token_index,
            num_image_tiles=num_image_tiles,
            attention_mask=attention_mask,
            packed_seq_params=packed_seq_params,
        )  # [combined_seq_len, b, h_language], [b, combined_seq_len], [b, combined_seq_len]
        torch.cuda.synchronize()
        # import pdb;
        # pdb.set_trace()
        print(f"Time to preprocess data: {time.time() - start}")
        #print(f"combined_embeddings shape: {combined_embeddings.shape}\n"
              # f"final_labels: {final_labels}\n"
              # f"final_loss_mask: {final_loss_mask}\n"
              # f"final_attention_mask: {final_attention_mask}")
        torch.cuda.synchronize()
        torch.distributed.barrier()
        start = time.time()
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
        torch.cuda.synchronize()
        print(f"Time to run language model: {time.time() - start}")

        #print(f"output shape: {output.shape}")
        if labels is None or loss_mask is None:
            return output

        #print(f"Final loss mask shape: {final_loss_mask.shape}")
        return output, final_loss_mask.contiguous()

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


    def print_with_line_number(self):
        import inspect
        line_number = inspect.currentframe().f_lineno
        #print(f"Line {line_number}: This is the message.")

    def _preprocess_data(
        self,
        image_embeddings,
        language_embeddings,
        input_ids,
        loss_mask,
        labels,
        use_inference_kv_cache,
        image_token_index,
        num_image_tiles,
        attention_mask,
        packed_seq_params,
    ):
        """Preprocess input data before input to language model.

        This function is adopted from
        https://github.com/huggingface/transformers/blob/85817d98fb60977c97e3014196a462b732d2ed1a/src/transformers/models/llava_next/modeling_llava_next.py#L409
        for our input data conventions.

        image_token_index = -200 indicates the image position in the input_ids = [0, 1, -200, 2, 3]
        and labels = [1, -200, 2, 3, 4], for example.
        We want to replace the image position (-200) with image_embeddings and return the following:
        - final_embeddings = [0, 1, image_embeddings, 2, 3],
        - final_labels = [1, -100, 2, 3, 4]
        - final_loss_mask = [1, 0, 0, 1, 1]

        This function handles samples without images (text-only sample). It also handles samples
        with images that are split into multiples tiles.

        If pipeline parallelism is not used, then self.pre_process and self.post_process
        are both True and we update both input embeddings, labels and loss masks (if available).

        If pipeline parallelism is used, then we do the following
        - the first language model chunk has self.pre_process = True and
          self.post_process = False. We update input embeddings.
        - the middle language model chunk(s) has self.pre_process = False and
          self.post_process = False. We don't need to update anything.
        - the last language model chunk has self.pre_process = False and
          self.post_process = True. We update labels and loss mask.

        TODO: This function should adjust the attention mask too.
        Currently, we assume the language model uses a causal mask.

        Returns:
            final_embedding (torch.Tensor): image and text embeddings [combined_seq_len, b, h].
            final_labels (torch.Tensor): labels for image and text positions [b, combined_seq_len].
            final_loss_mask (torch.Tensor): loss mask [b, combined_seq_len].
        """
        assert self.add_decoder, "input text preprocessing is only needed for the language model"
        from inspect import currentframe, getframeinfo
        # No pre- or postprocessing needed.
        # With pipeline parallel > 2, this means a chunk in the middle of the model.
        frameinfo = getframeinfo(currentframe()); #print(frameinfo.filename, frameinfo.lineno)
        if not self.pre_process and not self.post_process:
            return language_embeddings, loss_mask, labels, attention_mask

        frameinfo = getframeinfo(currentframe()); #print(frameinfo.filename, frameinfo.lineno)
        # If using the inference KV cache, the image tokens are already computed.
        if use_inference_kv_cache:
            return language_embeddings, loss_mask, labels, attention_mask

        img_seq_len = self._img_seq_len
        batch_size, text_seq_len = input_ids.shape
        frameinfo = getframeinfo(currentframe()); #print(frameinfo.filename, frameinfo.lineno)
        has_labels = labels is not None
        if has_labels:
            assert (
                labels.shape == loss_mask.shape
            ), f"mismatching labels shape {labels.shape} and loss mask shape {loss_mask.shape}"
        frameinfo = getframeinfo(currentframe()); #print(frameinfo.filename, frameinfo.lineno)
        packed_sequence = packed_seq_params is not None and packed_seq_params.qkv_format == "thd"
        frameinfo = getframeinfo(currentframe()); #print(frameinfo.filename, frameinfo.lineno)
        # Create indices for new text and label positions.
        with torch.no_grad():
            frameinfo = getframeinfo(currentframe());
            #print(frameinfo.filename, frameinfo.lineno)

            image_token_mask = input_ids == image_token_index
            num_images_per_sample = torch.sum(image_token_mask, dim=-1)
            frameinfo = getframeinfo(currentframe()); #print(frameinfo.filename, frameinfo.lineno)
            # Number of tiles per sample.
            #print(f"{num_images_per_sample =}")
            #print(f"{num_image_tiles = }")
            num_image_tiles_batch = num_image_tiles.split(num_images_per_sample.tolist(), dim=0)
            frameinfo = getframeinfo(currentframe());
            #print(frameinfo.filename, frameinfo.lineno)
            num_image_tiles_batch = torch.tensor([x.sum() for x in num_image_tiles_batch], device=input_ids.device)
            frameinfo = getframeinfo(currentframe()); #print(frameinfo.filename, frameinfo.lineno)
            # import pdb; pdb.set_trace()
            # Sequence length for each sample is the image sequence length multiplied by
            # the number of tiles for that image, minus image token indices,
            # plus text sequence length.
            seq_lens = num_image_tiles_batch * img_seq_len - num_images_per_sample + text_seq_len
            max_seq_len = seq_lens.max()
            frameinfo = getframeinfo(currentframe()); #print(frameinfo.filename, frameinfo.lineno)
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
                sp_padding_needed = padded_seq_len - max_seq_len
                max_seq_len = padded_seq_len
            frameinfo = getframeinfo(currentframe()); #print(frameinfo.filename, frameinfo.lineno)
            batch_indices, non_image_indices = torch.where(input_ids != image_token_index)
            frameinfo = getframeinfo(currentframe()); #print(frameinfo.filename, frameinfo.lineno)
            # New position ids for the text tokens, shifted by the image sequence length.
            # E.g. for input_ids = [-200, 1, 2, 3] and img_seq_len = 576, we get
            # new_position_ids = [576, 577, 578, 579]. text_position_ids are then [577, 578, 579].
            image_token_mask_lens = image_token_mask.int().clone()
            # -1 is for the removed image token index.
            image_token_mask_lens[image_token_mask] = num_image_tiles * img_seq_len - 1
            frameinfo = getframeinfo(currentframe()); #print(frameinfo.filename, frameinfo.lineno)
            # +1 is needed here for the cumulative sum. -1 is adjusting for zero-based indexing.
            new_position_ids = torch.cumsum((image_token_mask_lens + 1), dim=-1) - 1
            frameinfo = getframeinfo(currentframe()); #print(frameinfo.filename, frameinfo.lineno)
            text_position_ids = new_position_ids[batch_indices, non_image_indices]
            frameinfo = getframeinfo(currentframe()); #print(frameinfo.filename, frameinfo.lineno)
            # Labels are shifted to left by one.
            # So, shift text position ids and non-image indices to left by one.
            if has_labels:
                frameinfo = getframeinfo(currentframe()); #print(frameinfo.filename, frameinfo.lineno)
                label_text_position_ids = text_position_ids - 1
                valid_label_text_position_ids = label_text_position_ids >= 0
                label_text_position_ids = label_text_position_ids[valid_label_text_position_ids]
                frameinfo = getframeinfo(currentframe()); #print(frameinfo.filename, frameinfo.lineno)
                label_batch_indices = batch_indices[valid_label_text_position_ids]
                frameinfo = getframeinfo(currentframe()); #print(frameinfo.filename, frameinfo.lineno)
                label_non_image_indices = non_image_indices - 1
                valid_label_non_image_indices = label_non_image_indices >= 0
                label_non_image_indices = label_non_image_indices[valid_label_non_image_indices]

            frameinfo = getframeinfo(currentframe()); #print(frameinfo.filename, frameinfo.lineno)
            # Create a mask for the image embedding positions.
            images_mask = torch.full((batch_size, max_seq_len), True, dtype=torch.bool, device=input_ids.device)
            # No images in the text positions.
            images_mask[batch_indices, text_position_ids] = False
            frameinfo = getframeinfo(currentframe()); #print(frameinfo.filename, frameinfo.lineno)
            # Samples can have different amount of images tokens.
            # new_position_ids[:, -1] gives the last text position id for each sample.
            # Padding is needed when the number of image tokens differs.
            first_padding_idx = new_position_ids[:, -1] + 1
            frameinfo = getframeinfo(currentframe()); #print(frameinfo.filename, frameinfo.lineno)
            images_mask[
                torch.arange(max_seq_len, device=first_padding_idx.device).repeat(batch_size, 1)
                >= first_padding_idx.unsqueeze(1)
            ] = False

        # Create the final input embedding (if this is the first language model stage).
        frameinfo = getframeinfo(currentframe()); #print(frameinfo.filename, frameinfo.lineno)
        final_embedding = None
        if self.pre_process:
            frameinfo = getframeinfo(currentframe()); #print(frameinfo.filename, frameinfo.lineno)
            embed_dim = language_embeddings.shape[-1]
            final_embedding = torch.zeros(
                batch_size,
                max_seq_len,
                embed_dim,
                dtype=language_embeddings.dtype,
                device=language_embeddings.device,
            )
            frameinfo = getframeinfo(currentframe()); #print(frameinfo.filename, frameinfo.lineno)

            # Put text embeddings to the text positions in the result tensor.
            final_embedding[batch_indices, text_position_ids] = language_embeddings[batch_indices, non_image_indices]
            frameinfo = getframeinfo(currentframe()); #print(frameinfo.filename, frameinfo.lineno)
            # Put image embeddings to image positions.
            final_embedding[images_mask] = image_embeddings.permute(1, 0, 2).reshape(-1, embed_dim).contiguous()
        frameinfo = getframeinfo(currentframe()); #print(frameinfo.filename, frameinfo.lineno)
        # Create the final labels and loss mask (if this is the last language model stage).
        final_labels, final_loss_mask = None, None
        if has_labels:
            final_labels = torch.full(
                (batch_size, max_seq_len), IGNORE_INDEX, dtype=labels.dtype, device=labels.device
            )
            final_loss_mask = torch.full((batch_size, max_seq_len), 0, dtype=loss_mask.dtype, device=loss_mask.device)

            # Put text labels and loss mask to the text positions.
            final_labels[label_batch_indices, label_text_position_ids] = labels[
                label_batch_indices, label_non_image_indices
            ]

            final_loss_mask[batch_indices, text_position_ids] = loss_mask[batch_indices, non_image_indices]

            # For labels, pick the last label index that got dropped by the shift to left.
            label_extra_text_position_ids = seq_lens - 1
            batch_range = torch.arange(len(label_extra_text_position_ids))
            final_labels[batch_range, label_extra_text_position_ids] = labels[batch_range, -1]

            # Loss mask the image positions.
            final_loss_mask[images_mask] = 0

            # Loss mask last text position just before an image
            # so that text token does not need to predict the first image token.
            batch_image_indices, image_indices = torch.where(image_token_mask)
            # Indices just before image tokens. If it's -1, skip it.
            before_image_indices = image_indices - 1
            valid = before_image_indices >= 0
            valid_batch_image_indices = batch_image_indices[valid]
            valid_before_image_indices = before_image_indices[valid]
            # Map those indices those position ids.
            valid_before_image_indices = new_position_ids[valid_batch_image_indices, valid_before_image_indices]

            final_loss_mask[valid_batch_image_indices, valid_before_image_indices] = 0

        if final_embedding is not None and has_labels:
            assert (
                final_embedding.shape[:2] == final_labels.shape == final_loss_mask.shape
            ), "unexpected shapes after data preprocessing"

        truncate_labels = has_labels and final_labels.shape[1] > self._language_max_sequence_length
        if truncate_labels:
            final_labels = final_labels[:, : self._language_max_sequence_length]
            final_loss_mask = final_loss_mask[:, : self._language_max_sequence_length]

        if final_embedding is not None:
            final_embedding = final_embedding.transpose(1, 0).contiguous()
            # Truncate if exceeding the language model's max sequence length.
            if final_embedding.shape[0] > self._language_max_sequence_length:
                final_embedding = final_embedding[: self._language_max_sequence_length]
                if packed_sequence:
                    truncate_len = packed_seq_params.cu_seqlens_q_padded[-1] - self._language_max_sequence_length
                    packed_seq_params.cu_seqlens_q_padded[-1] = self._language_max_sequence_length
                    packed_seq_params.cu_seqlens_kv_padded[-1] = self._language_max_sequence_length
                    packed_seq_params.cu_seqlens_q[-1] -= truncate_len
                    packed_seq_params.cu_seqlens_kv[-1] -= truncate_len
                    assert (
                        packed_seq_params.cu_seqlens_q[-1] >= packed_seq_params.cu_seqlens_q[-2]
                    ), "with packed sequence, the truncation can only truncate on the last sequence."

            if self.sequence_parallel_lm and not packed_sequence:
                # Create an attention mask. This ensures correct computation.
                # This is done even when no padding was done as we set mask_type to
                # 'padding' or 'padding_causal' when using SP.
                if attention_mask is None:
                    # Create base attention mask with original seq len to indicate valid tokens
                    attention_mask = (
                        torch.ones(
                            (
                                final_embedding.shape[1],
                                final_embedding.shape[0] - sp_padding_needed,
                            ),
                            device=final_embedding.device,
                        )
                        .unsqueeze(1)
                        .unsqueeze(1)
                    )  # [b, 1, 1, final seq len - sp_padding_needed]
                if sp_padding_needed > 0:
                    # Add the padding portion of the mask
                    attention_mask = torch.nn.functional.pad(attention_mask, (0, sp_padding_needed))

                # Attention mask True/False meaning flipped in 1.7.0
                attention_mask = attention_mask < 0.5
            if self.sequence_parallel_lm:
                final_embedding = tensor_parallel.scatter_to_sequence_parallel_region(final_embedding)

        return final_embedding, final_labels, final_loss_mask, attention_mask

    def _freeze(
        self, freeze_language_model: bool, freeze_vision_model: bool,
            freeze_secondary_vision_model: bool,
            freeze_vision_projection: bool
    ):
        """Freeze model modules.

        Make specific modules non-trainable by setting requires_grad to False.

        Args:
            freeze_language_model (bool): Freeze the language model module.
            freeze_vision_model (bool): Freeze the vision model module.
            freeze_secondary_vision_model (bool): Freeze the secondary vision model module.
            freeze_vision_projection (bool): Freeze the vision projection module.
        """
        modules = []
        if freeze_language_model and self.language_model is not None:
            modules.append(self.language_model)
        if freeze_vision_model and self.vision_model is not None:
            modules.append(self.vision_model)
        if freeze_secondary_vision_model and self.secondary_vision_model is not None:
            modules.append(self.secondary_vision_model)
        if freeze_vision_projection and self.vision_projection is not None:
            modules.append(self.vision_projection)

        for module in modules:
            for param in module.parameters():
                param.requires_grad = False

class OpenVLAModel(L.LightningModule, io.IOMixin, io.ConnectorMixin, fn.FNMixin):
    def __init__(
        self,
        config: OpenVLAConfig,
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

        self.input_ids = input_ids
        self.position_ids = position_ids
        self.loss_mask = loss_mask
        self.attention_mask = attention_mask
        self.labels = labels
        self.inference_params = inference_params
        self.num_image_tiles = num_image_tiles
        self.image_token_index = image_token_index
        self.runtime_gather_output = runtime_gather_output
        self.image_token_mask = image_token_mask
        self.packed_seq_params = packed_seq_params
        self.images = images

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

    def run_forward(self) -> torch.Tensor:
        # Store the inputs and use them
        output_tensor = self.module(
            images=self.images,
            input_ids=self.input_ids,
            position_ids=self.position_ids,
            loss_mask=self.loss_mask,
            attention_mask=self.attention_mask,
            labels=self.labels,
            inference_params=self.inference_params,
            num_image_tiles=self.num_image_tiles,
            image_token_index=self.image_token_index,
            runtime_gather_output=self.runtime_gather_output,
            image_token_mask=self.image_token_mask,
            packed_seq_params=self.packed_seq_params,
        )
        return output_tensor

    def data_step(self, dataloader_iter) -> Dict[str, torch.Tensor]:
        return self.config.data_step_fn(dataloader_iter)

    def forward_step(self, batch) -> torch.Tensor:
        return self.config.forward_step_fn(self, batch)

    def training_step(self, batch, batch_idx=None) -> torch.Tensor:
        # In mcore the loss-function is part of the forward-pass (when labels are provided)
        return self.forward_step(batch)

    def on_train_batch_end(self, outputs: STEP_OUTPUT, batch: Any, batch_idx: int) -> None:
        # return
        torch.cuda.synchronize()
        torch.distributed.barrier()
        start = time.time()
        model_loss = lambda y: y[0].sum()
        fwd_and_bwd_flops = measure_flops(self, self.run_forward, model_loss)
        print(f"FLOPS: {fwd_and_bwd_flops}")
        print(f"Time: {time.time() - start}")

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


