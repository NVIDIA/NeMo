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
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, Dict, List, Literal, Optional, Tuple, Union

import pytorch_lightning as L
import torch
import torch.nn.functional as F
import wandb
from megatron.core import dist_checkpointing
from megatron.core.inference_params import InferenceParams
from megatron.core.models.multimodal.llava_model import LLaVAModel as MCoreLLaVAModel
from megatron.core.models.vision.multimodal_projector import MultimodalProjector as MCoreMultimodalProjector
from megatron.core.optimizer import OptimizerConfig
from megatron.core.transformer.custom_layers.transformer_engine import (
    TEColumnParallelLinear,
    TENorm,
    TERowParallelLinear,
)
from megatron.core.transformer.enums import ModelType
from megatron.core.transformer.mlp import MLP, MLPSubmodules
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_config import TransformerConfig
from torch import Tensor, nn

from nemo.collections.llm import Llama2Config7B, Llama2Config13B, LlamaConfig, fn
from nemo.collections.llm.gpt.model import local_layer_spec, transformer_engine_layer_spec
from nemo.lightning import get_vocab_size, io
from nemo.lightning.megatron_parallel import MaskedTokenLossReduction, MaskedTokenLossReductionWithLossMask
from nemo.lightning.pytorch.optim import MegatronOptimizerModule, OptimizerModule
from nemo.utils import logging

if TYPE_CHECKING:
    from megatron.core.models.gpt.gpt_model import GPTModel as MCoreGPTModel

    from nemo.collections.common.tokenizers.tokenizer_spec import TokenizerSpec

from nemo.collections.llm.gpt.model.base import get_batch_on_this_context_parallel_rank, get_packed_seq_params

# from nemo.collections.multimodal.mimo.model.gpt import MimoGPTModel


def compute_snr(timesteps, noise_scheduler):
    """
    Computes SNR as per https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L847-L849
    """
    alphas_cumprod = noise_scheduler.alphas_cumprod
    sqrt_alphas_cumprod = alphas_cumprod**0.5
    sqrt_one_minus_alphas_cumprod = (1.0 - alphas_cumprod) ** 0.5

    # Expand the tensors.
    # Adapted from https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L1026
    sqrt_alphas_cumprod = sqrt_alphas_cumprod.to(device=timesteps.device)[timesteps].float()
    while len(sqrt_alphas_cumprod.shape) < len(timesteps.shape):
        sqrt_alphas_cumprod = sqrt_alphas_cumprod[..., None]
    alpha = sqrt_alphas_cumprod.expand(timesteps.shape)

    sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod.to(device=timesteps.device)[timesteps].float()
    while len(sqrt_one_minus_alphas_cumprod.shape) < len(timesteps.shape):
        sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod[..., None]
    sigma = sqrt_one_minus_alphas_cumprod.expand(timesteps.shape)

    # Compute SNR.
    snr = (alpha / sigma) ** 2
    return snr


class MimoLossReduction(MaskedTokenLossReduction):
    def __init__(self, validation_step: bool = False, val_drop_last: bool = True, l2_weight: float = 1.0) -> None:
        super().__init__(validation_step, val_drop_last)
        self.l2_weight = l2_weight

    def forward(
        self,
        batch: Dict[str, torch.Tensor],
        forward_out: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Calculate masked token loss using superclass logic and add L2 loss.
        """
        # output_triple,new_loss_mask = forward_out
        # output, output_projection_embeddings, image_caption_embeddings = output_triple
        output_dict = forward_out

        output = output_dict['output']
        new_loss_mask = output_dict['new_loss_mask']
        output_projection_embeddings = output_dict['output_projection_embeddings']
        image_caption_embeddings = output_dict['image_caption_embeddings']
        # Use the superclass's forward method to calculate token loss
        token_loss, token_loss_info = super().forward(batch={"loss_mask": new_loss_mask}, forward_out=output)

        l2_loss = self._calculate_l2_loss(output_projection_embeddings, image_caption_embeddings)
        l2_loss = self.l2_weight * l2_loss
        from nemo.collections.nlp.modules.common.megatron.utils import average_losses_across_data_parallel_group

        reduced_l2_loss = average_losses_across_data_parallel_group([l2_loss])

        total_loss = token_loss + l2_loss
        logging.info(f"Yash loss debug total_loss {total_loss}")
        token_loss_info['avg'] = token_loss_info['avg'] + reduced_l2_loss
        token_loss_info.update({"l2_loss": reduced_l2_loss})

        # denoise l2 loss

        # mse_loss_weights = output_dict['denoise_mse_loss_weights']
        model_pred = output_dict['denoise_model_pred']
        target = output_dict['denoise_target']

        # gen_loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
        # gen_loss = gen_loss.mean(dim=[]) * mse_loss_weights
        # gen_loss = gen_loss.mean()
        gen_loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
        reduced_gen_l2_loss = average_losses_across_data_parallel_group([gen_loss])

        total_loss = total_loss + gen_loss
        token_loss_info['avg'] = token_loss_info['avg'] + reduced_gen_l2_loss

        logging.info(
            f"Yash loss debug full loss {token_loss_info['avg'] } token_loss {token_loss} embedding l2 loss {reduced_l2_loss} denoise l2 loss {reduced_gen_l2_loss}"
        )
        # if torch.distributed.get_rank() == 0:
        #     wandb.log(
        #         {
        #             "full_loss": token_loss_info['avg'],
        #             "token_loss": token_loss,
        #             "embedding_l2_loss": reduced_l2_loss,
        #             "denoise_l2_loss": reduced_gen_l2_loss,
        #         }
        #     )
        return total_loss, token_loss_info

    def _calculate_l2_loss(self, embeddings1: torch.Tensor, embeddings2: torch.Tensor) -> torch.Tensor:
        """Calculate L2 loss (mean squared error) between two sets of embeddings."""
        return torch.nn.functional.mse_loss(embeddings1, embeddings2)


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


class TransformersProjector(nn.Module):
    def __init__(self, in_features, out_features, num_query_token, **kwargs):
        super().__init__()
        hidden_dim = 512
        self.in_fc = nn.Linear(in_features, hidden_dim)
        self.tfm = nn.Transformer(
            batch_first=True,
            norm_first=True,
            d_model=hidden_dim,
            num_encoder_layers=4,
            num_decoder_layers=4,
            dim_feedforward=hidden_dim * 4,
            dropout=0.0,
            nhead=4,
        )
        self.out_fc = nn.Linear(hidden_dim, out_features)

        self.query_embs = nn.Parameter(torch.randn(1, num_query_token, hidden_dim))
        self.query_embs.data.normal_(mean=0.0, std=0.0)

    def forward(self, x):
        # x = x + input_embs # Yash TODO: pass in input embeddings
        x = self.in_fc(x)
        x = self.tfm(x, self.query_embs.repeat(x.shape[0], 1, 1))
        outputs = self.out_fc(x)
        return outputs


from megatron.core.models.gpt.gpt_model import GPTModel as MCoreGPTModel


class MimoGPTModel(MCoreGPTModel):
    from megatron.core.packed_seq_params import PackedSeqParams

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(
        self,
        input_ids: Tensor,
        position_ids: Tensor,
        attention_mask: Tensor,
        decoder_input: Tensor = None,
        labels: Tensor = None,
        inference_params: InferenceParams = None,
        packed_seq_params: PackedSeqParams = None,
        extra_block_kwargs: dict = None,
        runtime_gather_output: Optional[bool] = None,
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:

        original_post_process = self.post_process
        self.post_process = False

        try:
            hidden_states = super().forward(
                input_ids=input_ids,
                position_ids=position_ids,
                attention_mask=attention_mask,
                decoder_input=decoder_input,
                inference_params=inference_params,
                packed_seq_params=packed_seq_params,
                extra_block_kwargs=extra_block_kwargs,
                runtime_gather_output=runtime_gather_output,
            )
        finally:
            self.post_process = original_post_process

        output_weight = None
        if self.share_embeddings_and_output_weights:
            output_weight = self.shared_embedding_or_output_weight()

        logits, _ = self.output_layer(hidden_states, weight=output_weight, runtime_gather_output=runtime_gather_output)

        if labels is None:
            return logits.transpose(0, 1).contiguous(), hidden_states
        # if torch.distributed.get_rank() == 0:  # or other ranks
        #     breakpoint()
        # torch.distributed.barrier()
        loss = self.compute_language_model_loss(labels, logits)
        return loss, hidden_states


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
class BaseMimoConfig(TransformerConfig, io.IOMixin):
    language_transformer_config: Optional[TransformerConfig] = field(default_factory=lambda: Llama2Config7B())
    vision_transformer_config: Optional[TransformerConfig] = field(
        default_factory=lambda: BaseVisionTransformerConfig()
    )
    vision_projection_config: Optional[TransformerConfig] = field(default_factory=lambda: BaseInputProjectorConfig())

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

    def configure_model(self, tokenizer) -> "MCoreLLaVAModel":

        self.vocab_size = get_vocab_size(self, tokenizer.vocab_size, self.make_vocab_size_divisible_by)
        logging.info(f"padded vocab size to {self.vocab_size}")

        model = MCoreLLaVAModel(
            language_transformer_config=self.language_transformer_config,
            language_transformer_layer_spec=transformer_engine_layer_spec(self.language_transformer_config),
            language_vocab_size=self.vocab_size,
            language_max_sequence_length=self.language_transformer_config.seq_length,
            vision_transformer_config=self.vision_transformer_config,
            vision_transformer_layer_spec=transformer_engine_layer_spec(self.vision_transformer_config),
            drop_vision_class_token=True,
            vision_projection_config=self.vision_projection_config,
            vision_projection_layer_spec=self.vision_projection_config.layer_spec,
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
        return model


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


class CustomMimoModel(MCoreLLaVAModel):
    def __init__(
        self,
        model_config: TransformerConfig,
        language_transformer_config: TransformerConfig,
        language_transformer_layer_spec: ModuleSpec,
        language_vocab_size: int,
        language_max_sequence_length: int,
        vision_transformer_config: TransformerConfig,
        vision_transformer_layer_spec: ModuleSpec,
        drop_vision_class_token: bool,
        vision_projection_config: TransformerConfig,
        vision_projection_layer_spec: ModuleSpec,
        vision_output_projection_config: TransformerConfig,
        vision_output_projection_spec: ModuleSpec,
        vision_projection_type: str = "mlp",
        allow_missing_vision_projection_checkpoint: bool = False,
        parallel_output: bool = True,
        language_position_embedding_type: str = 'learned_absolute',
        language_rotary_percent: float = 1.0,
        pre_process: bool = True,
        post_process: bool = True,
        add_encoder: bool = True,
        add_decoder: bool = True,
        img_h: int = 336,
        img_w: int = 336,
        patch_dim: int = 14,
        language_rotary_base: int = 10000,
        language_rope_scaling: bool = False,
    ) -> None:
        # Temporarily disable add_decoder to prevent MCoreGPTModel initialization
        self.add_decoder = False
        super().__init__(
            language_transformer_config=language_transformer_config,
            language_transformer_layer_spec=language_transformer_layer_spec,
            language_vocab_size=language_vocab_size,
            language_max_sequence_length=language_max_sequence_length,
            vision_transformer_config=vision_transformer_config,
            vision_transformer_layer_spec=vision_transformer_layer_spec,
            drop_vision_class_token=drop_vision_class_token,
            vision_projection_config=vision_projection_config,
            vision_projection_layer_spec=vision_projection_layer_spec,
            vision_projection_type=vision_projection_type,
            allow_missing_vision_projection_checkpoint=allow_missing_vision_projection_checkpoint,
            parallel_output=parallel_output,
            language_position_embedding_type=language_position_embedding_type,
            language_rotary_percent=language_rotary_percent,
            pre_process=pre_process,
            post_process=post_process,
            add_encoder=add_encoder,
            add_decoder=False,  # Ensure GPTModel isn't initialized
            img_h=img_h,
            img_w=img_w,
            patch_dim=patch_dim,
            language_rotary_base=language_rotary_base,
            language_rope_scaling=language_rope_scaling,
        )
        self.model_config = model_config
        # Now re-enable add_decoder after parent constructor is done
        self.add_decoder = True
        self.model_type = ModelType.encoder_or_decoder

        # Initialize MimoGPTModel
        self.language_model = MimoGPTModel(
            config=language_transformer_config,
            transformer_layer_spec=language_transformer_layer_spec,
            vocab_size=language_vocab_size,
            max_sequence_length=language_max_sequence_length,
            parallel_output=parallel_output,
            position_embedding_type=language_position_embedding_type,
            rotary_percent=language_rotary_percent,
            pre_process=pre_process,
            post_process=post_process,
            rotary_base=language_rotary_base,
            rope_scaling=language_rope_scaling,
        )

        self.share_embeddings_and_output_weights = self.language_model.share_embeddings_and_output_weights
        self._language_max_sequence_length = language_max_sequence_length
        self._language_is_pipeline_parallel = language_transformer_config.pipeline_model_parallel_size > 1
        from diffusers import EulerDiscreteScheduler, StableDiffusionPipeline

        self.image_decoder_name = "stabilityai/stable-diffusion-2"
        self.scheduler = EulerDiscreteScheduler.from_pretrained(self.image_decoder_name, subfolder="scheduler")
        self.image_decoder = StableDiffusionPipeline.from_pretrained(self.image_decoder_name, scheduler=self.scheduler)

        self.image_decoder.vae.requires_grad_(False)
        self.image_decoder.unet.requires_grad_(False)
        self.image_decoder.text_encoder.requires_grad_(False)

        # output projection Megatron Module

        # self.vision_output_projection_module = MCoreMultimodalProjector(
        #         vision_output_projection_config,
        #         vision_output_projection_spec,
        #         projector_type="mlp" ,
        #         input_size=vision_output_projection_config.input_size,
        #     )
        self.vision_output_projection_module = TransformersProjector(
            in_features=self.config.hidden_size, out_features=1024, num_query_token=77
        )  # Yash : TODO Fix hard coding

    def get_image_caption_embeddings(self, text_input):
        with torch.no_grad():
            text_inputs = self.image_decoder.tokenizer(
                text_input, padding="max_length", truncation=True, return_tensors="pt", add_special_tokens=True
            )
            # if torch.distributed.get_rank() == 0:  # or other ranks
            #     breakpoint()
            # torch.distributed.barrier()
            text_inputs = text_inputs.to(self.image_decoder.device)
            image_caption_embeddings = self.image_decoder.text_encoder(**text_inputs)[0]  # b,77,1024

            return image_caption_embeddings

    def forward(
        self,
        images: torch.Tensor,
        output_images: torch.Tensor,
        input_ids: torch.Tensor,
        input_text: str,
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        loss_mask: Optional[torch.Tensor] = None,
        inference_params: Optional[InferenceParams] = None,
        num_image_tiles: Optional[List[int]] = None,
        image_token_index: Optional[int] = -200,
        runtime_gather_output: Optional[bool] = None,
    ) -> torch.Tensor:
        """Forward function of the LLaVA model.

        Args:
            images (torch.Tensor): input images of shape [num_tiles, img_h, img_w].
                num_tiles means the number of image tiles in this batch.
                num_tiles = 0 if the batch doesn't contain images.
            input_ids (torch.Tensor): input text ids [batch, text_seq_len].
            position_ids (torch.Tensor): input text position ids [batch, text_seq_len].
            attention_mask (torch.Tensor): Language model attention mask
                [batch, 1, combined_seq_len, combined_seq_len].
            labels (torch.Tensor): Optional target text labels [batch, combined_seq_len].
            loss_mask (torch.Tensor): Text loss mask [batch, text_seq_len].
            inference_params (InferenceParams): Inference-time parameters including KV cache.
            num_image_tiles (list of int): Number of tiles per image. Default 1 tile per image.
            image_token_index (int): ID for input images.
            runtime_gather_output (bool): Gather output at runtime. Default None means
                `parallel_output` arg in the constructor will be used.

        Returns:
            output (torch.Tensor): Loss of shape [b, s] if labels are provided,
                otherwise logits of shape [b, s, vocab_size].
            loss_mask (torch.Tensor): Loss mask expanded to combined sequence length. Shape [b, s].
        """
        use_inference_kv_cache = (
            inference_params is not None and "image_tokens_count" in inference_params.key_value_memory_dict
        )
        has_images = images.shape[0] > 0

        # If running inference, we can skip image token computation
        # if they were computed already earlier for this sample.
        if use_inference_kv_cache:
            image_embeddings = None
        elif self.add_encoder and not has_images:
            # If no images provided, use an empty image embeddings tensor.
            image_embeddings = torch.tensor([], dtype=images.dtype, device=images.device).reshape(0, 0, 0)
        elif self.add_encoder and has_images:
            image_embeddings = self.vision_model(images)  # [num_tiles, img_seq_len, h_vision]
            if self._drop_vision_class_token:
                image_embeddings = image_embeddings[:, self.vision_model.class_token_len :, :]
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
            return image_embeddings, loss_mask

        language_embeddings = None
        if self.pre_process:
            input_ids_text = input_ids.clone()
            input_ids_text[input_ids_text == image_token_index] = 0
            # Note: This adds absolute position embedding but not RoPE.
            # Each image is counted as one position.
            # RoPE is added in language_model forward. Each image embedding is one position.
            language_embeddings = self.language_model.embedding(
                input_ids=input_ids_text, position_ids=position_ids
            )  # [text_seq_len, b, h_language]
            language_embeddings = language_embeddings.transpose(1, 0).contiguous()  # [b, text_seq_len, h_language]

        # Assume 1 tile per image if the number of tiles is not provided.
        if num_image_tiles is None:
            num_image_tiles = torch.ones(images.shape[0], dtype=torch.int, device=input_ids.device)

        # if torch.distributed.get_rank() == 0:  # or other ranks
        #     breakpoint()
        # torch.distributed.barrier()

        # Preprocess input, labels and loss mask.
        combined_embeddings, new_labels, new_loss_mask, attention_mask = self._preprocess_data(
            image_embeddings,
            language_embeddings,
            input_ids,
            loss_mask,
            labels,
            use_inference_kv_cache,
            image_token_index,
            num_image_tiles,
            attention_mask,
        )  # [combined_seq_len, b, h_language], [b, combined_seq_len], [b, combined_seq_len]
        # TODO: Yash return this hidden state for computing loss

        output, hidden_states = self.language_model(
            input_ids=None,
            position_ids=None,
            attention_mask=attention_mask,
            decoder_input=combined_embeddings,
            labels=new_labels,
            inference_params=inference_params,
            runtime_gather_output=runtime_gather_output,
        )
        # if labels is None output is logits (b,s,vocab_size) or its loss (b,s)

        # send hidden_state for special tokens to output_projection module.
        device = output_images.device
        image_decoder = self.image_decoder.to(device)
        image_caption_embeddings = self.get_image_caption_embeddings(input_text)  # (bs, 77, 1024)
        if new_labels is not None:
            special_token_mask = torch.zeros_like(new_labels, dtype=torch.bool)
            for idx in self.model_config.image_special_token_indices:
                special_token_mask |= new_labels == idx

            nonzero_indices = torch.nonzero(special_token_mask, as_tuple=False)
            special_token_positions = nonzero_indices[:, 1]
            special_token_indices = new_labels[special_token_mask]

            special_token_positions = special_token_positions.view(
                new_labels.size(0), -1
            )  # batch_size, no_special_tokens
            special_token_indices = special_token_indices.view(new_labels.size(0), -1)

            # if torch.distributed.get_rank() == 0:  # or other ranks
            #     breakpoint()
            # torch.distributed.barrier()
            special_token_mask = special_token_mask.transpose(0, 1).unsqueeze(-1)
            special_token_mask = special_token_mask.expand_as(hidden_states)
            selected_hidden_states = hidden_states[special_token_mask].view(
                hidden_states.size(1), -1, hidden_states.size(-1)
            )

            special_token_embeddings = self.language_model.embedding(
                input_ids=special_token_indices, position_ids=special_token_positions
            )
            special_token_embeddings = special_token_embeddings.transpose(0, 1)  # change to b,s,h

            output_projection_embeddings = self.vision_output_projection_module(
                selected_hidden_states + special_token_embeddings
            )  # (bs, no_special_tokens, 1024)
            # Image caption embeddings
            image_caption_embeddings = image_caption_embeddings.to(
                output_projection_embeddings.device, dtype=output_projection_embeddings.dtype
            )

        if labels is None or loss_mask is None:
            # return output
            return {
                'output': output,
                # 'output_projection_embeddings': output_projection_embeddings,
                # 'image_caption_embeddings': image_caption_embeddings,
                'hidden_states': hidden_states,
            }

        # for calcualating denoising loss
        # with torch.no_grad():

        # if torch.distributed.get_rank() == 0:  # or other ranks
        #     breakpoint()
        # torch.distributed.barrier()

        latents = image_decoder.to(device).vae.encode(output_images).latent_dist.sample()
        latents = latents * image_decoder.vae.config.scaling_factor

        noise = torch.randn_like(latents)
        bsz = latents.shape[0]
        # Sample a random timestep for each sample in the batch
        timesteps = torch.randint(0, image_decoder.scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
        timesteps = timesteps.long()

        # add noise to latents using timesteps
        noisy_latents = image_decoder.scheduler.add_noise(latents, noise, timesteps)
        # make added noise target
        target = noise
        # predict the added noise
        model_pred = image_decoder.unet(noisy_latents, timesteps, output_projection_embeddings).sample
        # model_pred = image_decoder.unet(
        #     noisy_latents, timesteps, image_caption_embeddings.to(dtype=noisy_latents.dtype)
        # # ).sample
        # snr = compute_snr(timesteps, image_decoder.scheduler)
        # mse_loss_weights = torch.stack([snr, 5 * torch.ones_like(timesteps)], dim=1).min(dim=1)[0] / snr

        # if torch.distributed.get_rank() == 0:  # or other ranks
        #     breakpoint()
        # torch.distributed.barrier()

        return {
            'output': output,
            'new_loss_mask': new_loss_mask,
            'output_projection_embeddings': output_projection_embeddings,
            'image_caption_embeddings': image_caption_embeddings,
            'hidden_states': hidden_states,
            # 'denoise_mse_loss_weights': mse_loss_weights,
            'denoise_model_pred': model_pred,
            'denoise_target': target,
        }
        # return (output,output_projection_embeddings, image_caption_embeddings), new_loss_mask


class BaseMimoModel(L.LightningModule, io.IOMixin, io.ConnectorMixin, fn.FNMixin):
    def __init__(
        self,
        config: CustomMimoConfig,
        # TODO: Add transformer_layer_spec when we update mcore
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

    def configure_model(self) -> "MCoreLLaVAModel":
        if not hasattr(self, "module"):
            self.module = self.config.configure_model(self.tokenizer)

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        input_text: str = None,
        loss_mask: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        output_images: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        inference_params: InferenceParams = None,
    ) -> torch.Tensor:

        output_tensor = self.module(
            images=images,
            output_images=output_images,
            input_ids=input_ids,
            position_ids=position_ids,
            loss_mask=loss_mask,
            attention_mask=attention_mask,
            labels=labels,
            inference_params=inference_params,
            input_text=input_text,
        )

        return output_tensor

    #   TODO: Yash: May be we can inherit GPTModel and not have someo of the common function implementations here.
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

    @property
    def training_loss_reduction(self) -> MimoLossReduction:
        if not self._training_loss_reduction:
            self._training_loss_reduction = MimoLossReduction()

        return self._training_loss_reduction

    @property
    def validation_loss_reduction(self) -> MimoLossReduction:
        if not self._validation_loss_reduction:
            self._validation_loss_reduction = MimoLossReduction(validation_step=True)

        return self._validation_loss_reduction


from pathlib import Path

from transformers import LlavaForConditionalGeneration

from nemo.collections.common.tokenizers.huggingface.auto_tokenizer import AutoTokenizer
from nemo.collections.multimodal.mimo.model.base import BaseMimoConfig, BaseMimoModel
from nemo.lightning import OptimizerModule, io, teardown


@io.model_importer(BaseMimoModel, "hf")
class HFLlavaMimoImporter(io.ModelConnector["LlavaForConditionalGeneration", BaseMimoModel]):
    def init(self) -> BaseMimoModel:
        return BaseMimoModel(self.config, tokenizer=self.tokenizer)

    def apply(self, output_path: Path) -> Path:

        source = LlavaForConditionalGeneration.from_pretrained(str(self))
        target = self.init()
        trainer = self.nemo_setup(target)
        self.convert_state(source, target)

        print(f"Converted Llava model to Nemo, saving to {output_path}")
        self.nemo_save(output_path, trainer)

        print(f"Converted Llava model saved to {output_path}")

        teardown(trainer, target)
        del trainer, target

        return output_path

    def convert_state(self, source, target):
        mapping = {}
        # vision module
        mapping.update(
            {
                "vision_tower.vision_model.embeddings.patch_embedding.weight": "vision_model.conv1.weight",
                "vision_tower.vision_model.embeddings.position_embedding.weight": "vision_model.position_embeddings.weight",
            }
        )
        # Update with pre-layer normalization
        mapping.update(
            {
                "vision_tower.vision_model.pre_layrnorm.weight": "vision_model.ln_pre.weight",
                "vision_tower.vision_model.pre_layrnorm.bias": "vision_model.ln_pre.bias",
            }
        )
        # Update with layer normalization layers
        mapping.update(
            {
                "vision_tower.vision_model.encoder.layers.*.layer_norm1.weight": "vision_model.decoder.layers.*.self_attention.linear_qkv.layer_norm_weight",
                "vision_tower.vision_model.encoder.layers.*.layer_norm1.bias": "vision_model.decoder.layers.*.self_attention.linear_qkv.layer_norm_bias",
                "vision_tower.vision_model.encoder.layers.*.layer_norm2.weight": "vision_model.decoder.layers.*.mlp.linear_fc1.layer_norm_weight",
                "vision_tower.vision_model.encoder.layers.*.layer_norm2.bias": "vision_model.decoder.layers.*.mlp.linear_fc1.layer_norm_bias",
            }
        )

        # Update with MLP layers (Feedforward block)
        mapping.update(
            {
                "vision_tower.vision_model.encoder.layers.*.mlp.fc1.weight": "vision_model.decoder.layers.*.mlp.linear_fc1.weight",
                "vision_tower.vision_model.encoder.layers.*.mlp.fc1.bias": "vision_model.decoder.layers.*.mlp.linear_fc1.bias",
                "vision_tower.vision_model.encoder.layers.*.mlp.fc2.weight": "vision_model.decoder.layers.*.mlp.linear_fc2.weight",
                "vision_tower.vision_model.encoder.layers.*.mlp.fc2.bias": "vision_model.decoder.layers.*.mlp.linear_fc2.bias",
            }
        )

        # Update with self-attention linear projection
        mapping.update(
            {
                "vision_tower.vision_model.encoder.layers.*.self_attn.out_proj.weight": "vision_model.decoder.layers.*.self_attention.linear_proj.weight",
                "vision_tower.vision_model.encoder.layers.*.self_attn.out_proj.bias": "vision_model.decoder.layers.*.self_attention.linear_proj.bias",
            }
        )

        # projection module

        mapping.update(
            {
                "multi_modal_projector.linear_1.weight": "vision_projection.encoder.linear_fc1.weight",
                "multi_modal_projector.linear_1.bias": "vision_projection.encoder.linear_fc1.bias",
                "multi_modal_projector.linear_2.weight": "vision_projection.encoder.linear_fc2.weight",
                "multi_modal_projector.linear_2.bias": "vision_projection.encoder.linear_fc2.bias",
            }
        )

        # Language model

        mapping.update(
            {
                # "language_model.model.embed_tokens.weight": "language_model.embedding.word_embeddings.weight",
                "language_model.model.layers.*.self_attn.o_proj.weight": "language_model.decoder.layers.*.self_attention.linear_proj.weight",
                "language_model.model.layers.*.mlp.down_proj.weight": "language_model.decoder.layers.*.mlp.linear_fc2.weight",
                "language_model.model.layers.*.input_layernorm.weight": "language_model.decoder.layers.*.self_attention.linear_qkv.layer_norm_weight",
                "language_model.model.layers.*.post_attention_layernorm.weight": "language_model.decoder.layers.*.mlp.linear_fc1.layer_norm_weight",
                "language_model.model.norm.weight": "language_model.decoder.final_layernorm.weight",
                # "language_model.lm_head.weight": "language_model.output_layer.weight",
            }
        )
        return io.apply_transforms(
            source,
            target,
            mapping=mapping,
            transforms=[
                _import_class_token,
                _import_linear_fc1,
                _import_language_qkv,
                _import_embed_tokens,
                _import_lm_head_weight,
                _import_vison_qkv,
                _transform_vision_qkv_bias,
            ],
        )

    @property
    def tokenizer(self) -> AutoTokenizer:
        """Returns the tokenizer with added special tokens, cached for reuse."""
        if not hasattr(self, "_tokenizer"):
            # Initialize and cache the tokenizer
            self._tokenizer = AutoTokenizer(str(self))

            # Define special tokens for images
            special_tokens = [f"IMG_{i}" for i in range(8)]

            # Add special tokens to the tokenizer
            self._tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})

        return self._tokenizer

    @property
    def config(self) -> BaseMimoConfig:
        image_special_tokens = [f"IMG_{i}" for i in range(8)]
        image_special_token_indices = [self.tokenizer.tokenizer.convert_tokens_to_ids(f"IMG_{i}") for i in range(8)]
        # vocab_size = get_vocab_size(self, self.tokenizer.vocab_size, 128)
        # print(f"new vocab_size {vocab_size}")
        return BaseMimoConfig(
            vocab_size=self.tokenizer.vocab_size,
            image_special_token_indices=image_special_token_indices,
            image_special_tokens=image_special_tokens,
        )


@io.state_transform(
    source_key=(
        "vision_tower.vision_model.encoder.layers.*.self_attn.q_proj.weight",
        "vision_tower.vision_model.encoder.layers.*.self_attn.k_proj.weight",
        "vision_tower.vision_model.encoder.layers.*.self_attn.v_proj.weight",
    ),
    target_key="vision_model.decoder.layers.*.self_attention.linear_qkv.weight",
)
def _import_vison_qkv(ctx: io.TransformCTX, q, k, v):
    megatron_config = ctx.target.vision_model.config
    head_num = megatron_config.num_attention_heads
    num_query_groups = megatron_config.num_query_groups
    heads_per_group = head_num // num_query_groups
    hidden_size = megatron_config.hidden_size
    head_size = megatron_config.kv_channels

    old_tensor_shape = q.size()
    new_q_tensor_shape = (head_num, head_size) + old_tensor_shape[1:]
    new_kv_tensor_shape = (num_query_groups, head_size) + old_tensor_shape[1:]

    q = q.view(*new_q_tensor_shape)
    k = k.view(*new_kv_tensor_shape)
    v = v.view(*new_kv_tensor_shape)

    qkv_weights_l = []
    for i in range(num_query_groups):
        qkv_weights_l.append(q[i * heads_per_group : (i + 1) * heads_per_group, :, :])
        qkv_weights_l.append(k[i : i + 1, :, :])
        qkv_weights_l.append(v[i : i + 1, :, :])
    qkv_weights = torch.cat(qkv_weights_l)
    assert qkv_weights.ndim == 3, qkv_weights.shape
    assert qkv_weights.shape[0] == (heads_per_group + 2) * num_query_groups, qkv_weights.shape
    assert qkv_weights.shape[1] == head_size, qkv_weights.shape
    assert qkv_weights.shape[2] == old_tensor_shape[1], qkv_weights.shape

    qkv_weights = qkv_weights.reshape([head_size * (head_num + 2 * num_query_groups), hidden_size])

    return qkv_weights


@io.state_transform(
    source_key=(
        "language_model.model.layers.*.mlp.gate_proj.weight",
        "language_model.model.layers.*.mlp.up_proj.weight",
    ),
    target_key="language_model.decoder.layers.*.mlp.linear_fc1.weight",
)
def _import_linear_fc1(down, gate):
    return torch.cat((down, gate), axis=0)


@io.state_transform(
    source_key="vision_tower.vision_model.embeddings.class_embedding",
    target_key="vision_model.class_token",
)
def _import_class_token(ctx: io.TransformCTX, class_embedding):
    # Source shape: (1024,)
    # Target shape: (1, 1, 1024)

    # Reshape the class embedding to match the target shape
    class_token = class_embedding.view(1, 1, -1)

    # Ensure the transformation is correct
    assert class_token.shape == (1, 1, 1024), f"Expected shape (1, 1, 1024), but got {class_token.shape}"

    return class_token


@io.state_transform(
    source_key=(
        "language_model.model.layers.*.self_attn.q_proj.weight",
        "language_model.model.layers.*.self_attn.k_proj.weight",
        "language_model.model.layers.*.self_attn.v_proj.weight",
    ),
    target_key="language_model.decoder.layers.*.self_attention.linear_qkv.weight",
)
def _import_language_qkv(ctx: io.TransformCTX, q, k, v):
    megatron_config = ctx.target.language_model.config
    head_num = megatron_config.num_attention_heads
    num_query_groups = megatron_config.num_query_groups
    heads_per_group = head_num // num_query_groups
    hidden_size = megatron_config.hidden_size
    head_size = megatron_config.kv_channels

    old_tensor_shape = q.size()
    new_q_tensor_shape = (head_num, head_size) + old_tensor_shape[1:]
    new_kv_tensor_shape = (num_query_groups, head_size) + old_tensor_shape[1:]

    q = q.view(*new_q_tensor_shape)
    k = k.view(*new_kv_tensor_shape)
    v = v.view(*new_kv_tensor_shape)

    qkv_weights_l = []
    for i in range(num_query_groups):
        qkv_weights_l.append(q[i * heads_per_group : (i + 1) * heads_per_group, :, :])
        qkv_weights_l.append(k[i : i + 1, :, :])
        qkv_weights_l.append(v[i : i + 1, :, :])
    qkv_weights = torch.cat(qkv_weights_l)
    assert qkv_weights.ndim == 3, qkv_weights.shape
    assert qkv_weights.shape[0] == (heads_per_group + 2) * num_query_groups, qkv_weights.shape
    assert qkv_weights.shape[1] == head_size, qkv_weights.shape
    assert qkv_weights.shape[2] == old_tensor_shape[1], qkv_weights.shape

    qkv_weights = qkv_weights.reshape([head_size * (head_num + 2 * num_query_groups), hidden_size])

    return qkv_weights


@io.state_transform(
    source_key="language_model.model.embed_tokens.weight",
    target_key="language_model.embedding.word_embeddings.weight",
)
def _import_embed_tokens(ctx: io.TransformCTX, source_embed):

    target_shape = ctx.target.state_dict()["language_model.embedding.word_embeddings.weight"].shape

    target_vocab_size = target_shape[0]
    embedding_dim = target_shape[1]
    assert (
        source_embed.shape[1] == embedding_dim
    ), f"Embedding dimension mismatch: source={source_embed.shape[1]}, target={embedding_dim}"
    target_embed = torch.empty(target_vocab_size, embedding_dim, dtype=source_embed.dtype, device=source_embed.device)
    target_embed[: source_embed.shape[0], :] = source_embed
    average_embedding = source_embed.mean(dim=0)
    target_embed[source_embed.shape[0] :, :] = average_embedding

    return target_embed


@io.state_transform(
    source_key="language_model.lm_head.weight",
    target_key="language_model.output_layer.weight",
)
def _import_lm_head_weight(ctx: io.TransformCTX, source_weight):
    target_shape = ctx.target.state_dict()["language_model.output_layer.weight"].shape
    target_vocab_size, target_embedding_dim = target_shape
    source_vocab_size, source_embedding_dim = source_weight.shape

    # Ensure the embedding dimensions match between source and target
    assert target_embedding_dim == source_embedding_dim, (
        f"Embedding dimension mismatch: " f"source={source_embedding_dim}, target={target_embedding_dim}"
    )

    target_weight = torch.empty(
        target_vocab_size, target_embedding_dim, dtype=source_weight.dtype, device=source_weight.device
    )

    target_weight[:source_vocab_size, :] = source_weight

    average_weight = source_weight.mean(dim=0)
    target_weight[source_vocab_size:, :] = average_weight

    assert target_weight.shape == target_shape, f"Expected shape {target_shape}, but got {target_weight.shape}"

    return target_weight


@io.state_transform(
    source_key=(
        "vision_tower.vision_model.encoder.layers.*.self_attn.q_proj.bias",
        "vision_tower.vision_model.encoder.layers.*.self_attn.k_proj.bias",
        "vision_tower.vision_model.encoder.layers.*.self_attn.v_proj.bias",
    ),
    target_key="vision_model.decoder.layers.*.self_attention.linear_qkv.bias",
)
def _transform_vision_qkv_bias(ctx: io.TransformCTX, q_bias, k_bias, v_bias):
    """
    Transforms and concatenates Q, K, V biases from the source model to the target model.
    """

    # Concatenate the Q, K, V biases into a single bias tensor
    qkv_bias = torch.cat([q_bias, k_bias, v_bias], dim=0)  # (3072,)

    # Ensure the concatenated bias has the correct shape
    expected_shape = (3072,)
    assert qkv_bias.shape == expected_shape, f"Expected shape {expected_shape}, but got {qkv_bias.shape}"

    return qkv_bias
