# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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

import itertools
import math
import os
import random
import tempfile
from functools import partial
from typing import Any, List, Optional, Union

import numpy as np
import pandas as pd
import torch
from einops import rearrange, repeat
from omegaconf.dictconfig import DictConfig
from omegaconf.omegaconf import OmegaConf, open_dict
from pytorch_lightning.accelerators import CPUAccelerator
from pytorch_lightning.trainer.trainer import Trainer

from nemo.collections.multimodal.data.kosmos.kosmos_dataset import MAX_NUM_IMAGES, MergedKosmosDataLoader
from nemo.collections.multimodal.data.kosmos.kosmos_dataset import (
    build_train_valid_datasets as build_media_train_valid_datasets,
)
from nemo.collections.multimodal.models.clip.megatron_clip_models import CLIPVisionTransformer
from nemo.collections.multimodal.models.kosmos.perceiver_resampler import PerceiverResampler
from nemo.collections.nlp.data.language_modeling.megatron.data_samplers import (
    MegatronPretrainingRandomSampler,
    MegatronPretrainingSampler,
)
from nemo.collections.nlp.data.language_modeling.megatron.gpt_dataset import (
    build_train_valid_test_datasets as build_text_train_valid_test_datasets,
)
from nemo.collections.nlp.models.language_modeling.megatron.gpt_model import GPTModel, post_language_model_processing
from nemo.collections.nlp.models.language_modeling.megatron_base_model import MegatronBaseModel
from nemo.collections.nlp.models.language_modeling.megatron_gpt_model import MegatronGPTModel
from nemo.collections.nlp.modules.common.megatron.build_model import build_model
from nemo.collections.nlp.modules.common.megatron.language_model import get_language_model
from nemo.collections.nlp.modules.common.megatron.module import Float16Module, MegatronModule
from nemo.collections.nlp.modules.common.megatron.utils import (
    ApexGuardDefaults,
    average_losses_across_data_parallel_group,
    get_all_params_for_weight_decay_optimization,
    get_params_for_weight_decay_optimization,
    init_method_normal,
    parallel_lm_logits,
    scaled_init_method_normal,
)
from nemo.collections.nlp.modules.common.text_generation_utils import (
    generate,
    get_computeprob_response,
    get_default_length_params,
    get_default_sampling_params,
    megatron_gpt_generate,
)
from nemo.collections.nlp.modules.common.transformer.text_generation import (
    LengthParam,
    OutputType,
    SamplingParam,
    TextGeneration,
)
from nemo.collections.nlp.parts.nlp_overrides import GradScaler, NLPSaveRestoreConnector
from nemo.collections.nlp.parts.utils_funcs import get_last_rank
from nemo.collections.vision.modules.vit.vit_backbone import VitBackbone
from nemo.core.classes.common import PretrainedModelInfo
from nemo.utils import logging

try:
    import apex.transformer.pipeline_parallel.utils
    from apex.transformer.enums import AttnMaskType
    from apex.transformer.pipeline_parallel.utils import get_num_microbatches

    HAVE_APEX = True

except (ImportError, ModuleNotFoundError):

    HAVE_APEX = False

try:
    from megatron.core import parallel_state
    from megatron.core.pipeline_parallel.schedules import get_forward_backward_func

    HAVE_MEGATRON_CORE = True

except (ImportError, ModuleNotFoundError):

    HAVE_MEGATRON_CORE = False

try:
    import transformer_engine

    HAVE_TE = True

except (ImportError, ModuleNotFoundError):
    HAVE_TE = False


class FrozenCLIPVisionTransformer(CLIPVisionTransformer):
    def __init__(self, model_cfg, pre_process=True, post_process=True):
        super().__init__(
            model_cfg, pre_process=pre_process, post_process=post_process, skip_head=True,
        )
        self.frozen = False

    def train(self, mode):
        if self.frozen:
            return self

        super().train(mode)
        return self

    def forward(self, input):
        assert self.training == False
        hidden_states = self.backbone(input)
        # Do not add header after backbone
        return hidden_states

    def freeze(self) -> None:
        for param in self.parameters():
            param.requires_grad = False

        self.eval()
        self.frozen = True


class KosmosModel(MegatronModule):
    def __init__(
        self, model_cfg, vocab_size, media_start_id=None, media_end_id=None, pre_process=True, post_process=True,
    ):
        super(KosmosModel, self).__init__()

        llm_cfg = model_cfg.llm
        vision_cfg = model_cfg.vision

        self.parallel_output = True  # TODO (yuya): Fix this hard-code
        self.media_start_id = media_start_id
        self.media_end_id = media_end_id
        self.pre_process = pre_process
        self.post_process = post_process
        self.fp16_lm_cross_entropy = llm_cfg.get('fp16_lm_cross_entropy', False)
        self.sequence_parallel = llm_cfg.sequence_parallel
        self.gradient_accumulation_fusion = llm_cfg.gradient_accumulation_fusion
        self.share_embeddings_and_output_weights = llm_cfg.share_embeddings_and_output_weights
        self.position_embedding_type = llm_cfg.get('position_embedding_type', 'learned_absolute')

        use_scaled_init_method = llm_cfg.get('use_scaled_init_method', True)
        kv_channels = llm_cfg.get('kv_channels', None)
        hidden_size = llm_cfg.hidden_size
        num_attention_heads = llm_cfg.num_attention_heads
        num_layers = llm_cfg.num_layers
        init_method_std = llm_cfg.init_method_std

        if kv_channels is None:
            assert (
                hidden_size % num_attention_heads == 0
            ), 'hidden_size must be divisible by num_attention_heads if kv_channels is None'
            kv_channels = hidden_size // num_attention_heads

        scaled_init_method = (
            scaled_init_method_normal(init_method_std, num_layers)
            if use_scaled_init_method
            else init_method_normal(init_method_std)
        )
        self.language_model, self._language_model_key = get_language_model(
            vocab_size=vocab_size,
            hidden_size=llm_cfg.hidden_size,
            max_position_embeddings=llm_cfg.max_position_embeddings,
            num_layers=llm_cfg.num_layers,
            num_attention_heads=llm_cfg.num_attention_heads,
            apply_query_key_layer_scaling=llm_cfg.get('apply_query_key_layer_scaling', True),
            kv_channels=kv_channels,
            ffn_hidden_size=llm_cfg.ffn_hidden_size,
            num_tokentypes=0,
            add_pooler=False,
            encoder_attn_mask_type=AttnMaskType.causal,
            pre_process=pre_process,
            post_process=post_process,
            init_method_std=llm_cfg.get('init_method_std', 0.02),
            scaled_init_method=scaled_init_method,
            use_cpu_initialization=llm_cfg.get('use_cpu_initialization', False),
            hidden_dropout=llm_cfg.get('hidden_dropout', 0.1),
            attention_dropout=llm_cfg.get('attention_dropout', 0.1),
            ffn_dropout=llm_cfg.get('ffn_dropout', 0.0),
            precision=llm_cfg.get('precision', 16),
            fp32_residual_connection=llm_cfg.get('fp32_residual_connection', False),
            activations_checkpoint_granularity=llm_cfg.get('activations_checkpoint_granularity', None),
            activations_checkpoint_method=llm_cfg.get('activations_checkpoint_method', None),
            activations_checkpoint_num_layers=llm_cfg.get('activations_checkpoint_num_layers', 1),
            activations_checkpoint_layers_per_pipeline=llm_cfg.get('activations_checkpoint_layers_per_pipeline', None),
            normalization=llm_cfg.get('normalization', 'layernorm'),
            layernorm_epsilon=llm_cfg.get('layernorm_epsilon', 1e-5),
            onnx_safe=llm_cfg.get('onnx_safe', False),
            bias=llm_cfg.get('bias', True),
            bias_activation_fusion=llm_cfg.get('bias_activation_fusion', True),
            bias_dropout_add_fusion=llm_cfg.get('bias_dropout_add_fusion', True),
            activation=llm_cfg.get('activation', 'gelu'),
            headscale=llm_cfg.get('headscale', False),
            transformer_block_type=llm_cfg.get('transformer_block_type', 'pre_ln'),
            openai_gelu=llm_cfg.get('openai_gelu', False),
            normalize_attention_scores=llm_cfg.get('normalize_attention_scores', True),
            position_embedding_type=llm_cfg.get('position_embedding_type', 'learned_absolute'),
            rotary_percentage=llm_cfg.get('rotary_percentage', 1.0),
            share_embeddings_and_output_weights=llm_cfg.get('share_embeddings_and_output_weights', True),
            attention_type=llm_cfg.get('attention_type', 'multihead'),
            masked_softmax_fusion=llm_cfg.get('masked_softmax_fusion', True),
            gradient_accumulation_fusion=llm_cfg.get('gradient_accumulation_fusion', False),
            persist_layer_norm=llm_cfg.get('persist_layer_norm', False),
            sequence_parallel=llm_cfg.get('sequence_parallel', False),
            transformer_engine=llm_cfg.get('transformer_engine', False),
            fp8=llm_cfg.get('fp8', False),
            fp8_e4m3=llm_cfg.get('fp8_e4m3', False),
            fp8_hybrid=llm_cfg.get('fp8_hybrid', False),
            fp8_margin=llm_cfg.get('fp8_margin', 0),
            fp8_interval=llm_cfg.get('fp8_interval', 1),
            fp8_amax_history_len=llm_cfg.get('fp8_amax_history_len', 1),
            fp8_amax_compute_algo=llm_cfg.get('fp8_amax_compute_algo', 'most_recent'),
            reduce_amax=llm_cfg.get('reduce_amax', True),
            use_emha=llm_cfg.get('use_emha', False),
        )

        if self.share_embeddings_and_output_weights:
            self.initialize_word_embeddings(
                init_method=init_method_normal(init_method_std), vocab_size=vocab_size, hidden_size=hidden_size
            )

        # TODO (yuya): check when PP is added
        self.vision_encoder = FrozenCLIPVisionTransformer(
            vision_cfg, pre_process=vision_cfg.pre_process, post_process=vision_cfg.post_process,
        )
        if vision_cfg.from_pretrained is not None:
            logging.info(f"Loading CLIP vision encoder weights from checkpoint {vision_cfg.from_pretrained}")
            self.load_vision_encoder_weights(vision_cfg.from_pretrained)
        self.perceiver = PerceiverResampler(dim=vision_cfg.hidden_size, num_latents=model_cfg.num_media_latents)
        self.vision_connector = torch.nn.Linear(vision_cfg.hidden_size, llm_cfg.hidden_size, bias=False,)

    def set_input_tensor(self, input_tensor):
        """See megatron.model.transformer.set_input_tensor()"""
        self.language_model.set_input_tensor(input_tensor)

    def encode_vision_x(self, vision_x: torch.Tensor):
        """
        Compute media tokens from vision input by passing it through vision encoder and conditioning language model.
        Args:
            vision_x (torch.Tensor): Vision input
                shape (B, T_img, F, C, H, W)
                Images in the same chunk are collated along T_img, and frames are collated along F
                Currently only F=1 is supported (single-frame videos)

        rearrange code based on https://github.com/dhansmair/flamingo-mini
        """

        assert vision_x.ndim == 6, "vision_x should be of shape (b, T_img, F, C, H, W)"
        b, T, F = vision_x.shape[:3]
        assert F == 1, "Only single frame supported"

        vision_x = rearrange(vision_x, "b T F c h w -> (b T F) c h w")
        with torch.no_grad():
            vision_x = self.vision_encoder(vision_x)
        vision_x = rearrange(vision_x, "(b T F) v d -> b T F v d", b=b, T=T, F=F)
        vision_x = self.perceiver(vision_x)  # reshapes to (b, T, n, d)
        vision_x = self.vision_connector(vision_x)
        return vision_x

    def replace_media_embeddings(self, input_ids, inputs_embeds, media=None):
        if media is None:
            return inputs_embeds

        batch_size, sequence_length, hidden_size = inputs_embeds.shape

        # calculate media features without gradients
        with torch.no_grad():
            media_features = self.encode_vision_x(media)
        num_images_per_sample = media_features.size(1)
        num_patches = media_features.size(2)

        # flatten patches
        media_features = media_features.view(batch_size, -1, hidden_size)

        # create an indices matrix used in torch.scatter
        padded_media_indices = torch.ones(
            (batch_size, num_images_per_sample), dtype=torch.long, device=input_ids.device
        )
        padded_media_indices *= sequence_length
        for idx, input_id in enumerate(input_ids):
            media_end_positions = torch.where(input_id == self.media_end_id)[0]
            # locate the first media token positions
            padded_media_indices[idx, : len(media_end_positions)] = media_end_positions - num_patches

        # use indices to create a span
        padded_media_indices = padded_media_indices.unsqueeze(-1) + torch.arange(
            num_patches, device=padded_media_indices.device
        ).repeat(*padded_media_indices.shape, 1)
        padded_media_indices = padded_media_indices.reshape(batch_size, -1)
        padded_media_indices = repeat(padded_media_indices, 'b s -> b s h', h=hidden_size)

        # concat placeholder
        updated_input_embeds = torch.cat(
            (inputs_embeds, torch.zeros((batch_size, num_patches, hidden_size), device=inputs_embeds.device)), dim=1
        )
        updated_input_embeds = updated_input_embeds.type(media_features.dtype)
        # scatter media_features
        updated_input_embeds.scatter_(1, padded_media_indices, media_features)

        # chop off placeholder
        updated_input_embeds = updated_input_embeds[:, :sequence_length]

        return updated_input_embeds

    def forward(
        self,
        input_ids,
        position_ids,
        attention_mask,
        labels=None,
        media=None,
        token_type_ids=None,
        layer_past=None,
        get_key_value=False,
        forward_method_parallel_output=None,
        encoder_input=None,
        set_inference_key_value_memory=False,
        inference_max_sequence_len=None,
        checkpoint_activations_all_layers=None,
    ):
        # input_ids: [b, s]
        # position_ids: [b, s]
        # attention_mask: [1, 1, s, s]

        # Multimodal uses different forward pass. Vision tower must be inserted.
        enc_input_ids, enc_position_ids, enc_attn_mask = input_ids, position_ids, attention_mask

        # Embeddings.
        if self.pre_process and encoder_input is None:
            embedding_module = self.language_model.embedding

            words_embeddings = embedding_module.word_embeddings(enc_input_ids)
            words_embeddings = self.replace_media_embeddings(enc_input_ids, words_embeddings, media=media)

            if self.position_embedding_type == 'learned_absolute':
                assert position_ids is not None
                position_embeddings = embedding_module.position_embeddings(position_ids)
                embeddings = words_embeddings + position_embeddings
            elif self.position_embedding_type == 'learned_parameters':
                embeddings = words_embeddings + embedding_module.position_embeddings
            else:
                embeddings = words_embeddings

            if token_type_ids is not None:
                assert embedding_module.tokentype_embeddings is not None
                embeddings = embeddings + embedding_module.tokentype_embeddings(token_type_ids)
            else:
                assert embedding_module.tokentype_embeddings is None

            # Data format change to avoid explicit tranposes : [b s h] --> [s b h].
            if embedding_module.transpose_batch_sequence:
                embeddings = embeddings.transpose(0, 1).contiguous()

            # If the input flag for fp32 residual connection is set, convert for float.
            if embedding_module.fp32_residual_connection:
                embeddings = embeddings.float()

            # Dropout.
            if self.sequence_parallel:
                embeddings = tensor_parallel.mappings.scatter_to_sequence_parallel_region(embeddings)
                with tensor_parallel.random.get_cuda_rng_tracker().fork():
                    embeddings = embedding_module.embedding_dropout(embeddings)
            else:
                embeddings = embedding_module.embedding_dropout(embeddings)

            encoder_input = embeddings
        else:
            pass

        # enc_attn_mask: [1, 1, s, s]

        if self.position_embedding_type == 'rope':
            if inference_max_sequence_len is not None:
                rotary_pos_emb = self.language_model.rotary_pos_emb(inference_max_sequence_len)
            elif self.language_model.encoder.input_tensor is not None:
                if self.sequence_parallel:
                    rotary_pos_emb = self.language_model.rotary_pos_emb(
                        self.language_model.encoder.input_tensor.size(0)
                        * parallel_state.get_tensor_model_parallel_world_size()
                    )
                else:
                    rotary_pos_emb = self.language_model.rotary_pos_emb(self.encoder.input_tensor.size(0))
            else:
                if self.sequence_parallel:
                    rotary_pos_emb = self.language_model.rotary_pos_emb(
                        encoder_input.size(0) * parallel_state.get_tensor_model_parallel_world_size()
                    )
                else:
                    rotary_pos_emb = self.language_model.rotary_pos_emb(encoder_input.size(0))
        else:
            rotary_pos_emb = None

        # encoder but decoder for GPT
        encoder_output = self.language_model.encoder(
            encoder_input,
            enc_attn_mask,
            layer_past=layer_past,
            get_key_value=get_key_value,
            set_inference_key_value_memory=set_inference_key_value_memory,
            inference_max_sequence_len=inference_max_sequence_len,
            checkpoint_activations_all_layers=checkpoint_activations_all_layers,
            rotary_pos_emb=(rotary_pos_emb, None, None)
            if rotary_pos_emb is not None
            else None,  # This assumes that this being used as a GPT/BERT model only (no cross-attention)
        )

        lm_output = encoder_output

        if self.post_process:
            return post_language_model_processing(
                lm_output,
                labels,
                self.language_model.output_layer.weight
                if not self.share_embeddings_and_output_weights
                else self.word_embeddings_weight(),
                get_key_value,
                self.parallel_output,
                forward_method_parallel_output,
                self.fp16_lm_cross_entropy,
                return_logits=False,
                sequence_parallel=self.sequence_parallel,
                gradient_accumulation_fusion=self.gradient_accumulation_fusion,
            )
        else:
            return lm_output

    def load_vision_encoder_weights(self, nemo_path):
        if torch.cuda.is_available():
            map_location = torch.device('cuda')
        else:
            map_location = torch.device('cpu')
        save_restore_connector = NLPSaveRestoreConnector()
        cwd = os.getcwd()

        with tempfile.TemporaryDirectory() as tmpdir:
            try:
                save_restore_connector._unpack_nemo_file(path2file=nemo_path, out_folder=tmpdir)

                # Change current working directory to
                os.chdir(tmpdir)
                config_yaml = os.path.join(tmpdir, save_restore_connector.model_config_yaml)
                cfg = OmegaConf.load(config_yaml)

                model_weights = os.path.join(tmpdir, save_restore_connector.model_weights_ckpt)
                state_dict = save_restore_connector._load_state_dict_from_disk(
                    model_weights, map_location=map_location
                )
            finally:
                os.chdir(cwd)

        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("model.vision_encoder."):
                new_k = k.lstrip("model.vision_encoder.")
                new_state_dict[new_k] = v

        missing, unexpected = self.vision_encoder.load_state_dict(new_state_dict, strict=False)
        print(f"Restored from {nemo_path} with {len(missing)} missing and {len(unexpected)} unexpected keys")
        if len(missing) > 0:
            print(f"Missing Keys: {missing}")
        if len(unexpected) > 0:
            print(f"Unexpected Keys: {unexpected}")

    def state_dict_for_save_checkpoint(self, destination=None, prefix='', keep_vars=False):

        state_dict_ = {}
        state_dict_[self._language_model_key] = self.language_model.state_dict_for_save_checkpoint(
            destination, prefix, keep_vars
        )
        # Save word_embeddings.
        if self.post_process and not self.pre_process:
            state_dict_[self._word_embeddings_for_head_key] = self.word_embeddings.state_dict(
                destination, prefix, keep_vars
            )
        return state_dict_

    def load_state_dict(self, state_dict, strict=True):
        """Customized load."""

        # Load word_embeddings.
        if self.post_process and not self.pre_process:
            self.word_embeddings.load_state_dict(state_dict[self._word_embeddings_for_head_key], strict=strict)
        if self._language_model_key in state_dict:
            state_dict = state_dict[self._language_model_key]
        self.language_model.load_state_dict(state_dict, strict=strict)


class MegatronKosmosModel(MegatronGPTModel):
    """
    Megatron Kosmos pretraining
    """

    def __init__(self, cfg: DictConfig, trainer: Trainer):
        super().__init__(cfg, trainer)

        self.image_size = (self.cfg.vision.img_h, self.cfg.vision.img_w)
        self.megatron_amp_O2 = getattr(self, 'megatron_amp_O2', False)
        self.enabled_data_types = self.cfg.get("enabled_data_types", [])
        logging.info(f"Data types enabled in Kosmos training: {self.enabled_data_types}")
        self.per_type_micro_batch_size = self.cfg.per_type_micro_batch_size
        self.per_type_global_batch_size = {}
        self.per_type_loss_weights = {}
        for data_type in self.enabled_data_types:
            self.per_type_global_batch_size[data_type] = (
                self.per_type_micro_batch_size[data_type] * self.cfg.global_batch_size // self.cfg.micro_batch_size
            )
            self.per_type_loss_weights[data_type] = self.cfg.per_type_loss_weights[data_type]

    def get_gpt_module_list(self):
        if isinstance(self.model, list):
            return [model.module if isinstance(model, Float16Module) else model for model in self.model]
        elif isinstance(self.model, Float16Module):
            return [self.model.module]
        else:
            return [self.model]

    def set_inference_config(self, inference_config):
        self._inference_config = inference_config

    def get_inference_config(self):
        return self._inference_config

    def model_provider_func(self, pre_process, post_process):
        """Model depends on pipeline paralellism."""
        media_start_id = self.tokenizer.token_to_id(self.cfg.media_start_token)
        media_end_id = self.tokenizer.token_to_id(self.cfg.media_end_token)

        model = KosmosModel(
            model_cfg=self.cfg,
            vocab_size=self.padded_vocab_size,
            media_start_id=media_start_id,
            media_end_id=media_end_id,
            pre_process=pre_process,
            post_process=post_process,
        )

        # Freeze vit
        model.vision_encoder.freeze()

        logging.info(
            f"Kosmos model initialized with {sum(p.numel() for p in model.parameters() if p.requires_grad)} trainable parameters"
        )

        return model

    def forward(self, tokens, text_position_ids, attention_mask, labels, media=None):
        output_tensor = self.model(tokens, text_position_ids, attention_mask, labels=labels, media=media)
        return output_tensor

    def fwd_bwd_step(self, dataloader_iter, batch_idx, forward_only):

        tensor_shape = [self.cfg.llm.encoder_seq_length, self.cfg.micro_batch_size, self.cfg.llm.hidden_size]

        # handle asynchronous grad reduction
        no_sync_func = None
        grad_sync_func = None
        param_sync_func = None
        if not forward_only and self.with_distributed_adam:
            no_sync_func = partial(self._optimizer.no_sync, greedy_grad_copy=self.megatron_amp_O2,)
            grad_sync_func = self.reduce_overlap_gradients
            param_sync_func = self.sync_overlap_parameters

        # run forward and backwards passes for an entire global batch
        # we do this inside training_step to support pipeline parallelism
        fwd_bwd_function = get_forward_backward_func()

        # TODO @akhattar: remove sync related stuff from config, add num_micro_batches_with_partial_activation_checkpoints when ready
        losses_reduced_per_micro_batch = fwd_bwd_function(
            forward_step_func=self.get_forward_output_and_loss_func(),
            data_iterator=dataloader_iter,
            model=[self.model],
            num_microbatches=get_num_microbatches(),
            forward_only=forward_only,
            tensor_shape=tensor_shape,
            dtype=self.autocast_dtype,
            grad_scaler=self.trainer.precision_plugin.scaler.scale
            if self.cfg.precision in [16, '16', '16-mixed']
            else None,
            sequence_parallel=self.cfg.get('sequence_parallel', False),
            enable_autocast=self.enable_autocast,
            no_sync_func=no_sync_func,
            grad_sync_func=grad_sync_func,
            param_sync_func=param_sync_func,
        )

        # only the last stages of the pipeline return losses
        loss_dict = {}
        if losses_reduced_per_micro_batch:
            # average loss across micro batches
            loss_tensors_list = [loss_reduced['avg'] for loss_reduced in losses_reduced_per_micro_batch]
            loss_tensor = torch.stack(loss_tensors_list)
            loss_mean = loss_tensor.mean()
            for data_type in self.enabled_data_types:
                loss_tensors_list = [loss_reduced[data_type] for loss_reduced in losses_reduced_per_micro_batch]
                loss_tensor = torch.stack(loss_tensors_list)
                loss_dict[data_type] = loss_tensor.mean()
        else:
            loss_mean = torch.tensor(0.0).cuda()

        return loss_mean, loss_dict

    def training_step(self, dataloader_iter, batch_idx):
        """
            We pass the dataloader iterator function to the micro-batch scheduler.
            The input batch to each micro-batch is fetched using the dataloader function
            in the micro-batch fwd function.
        """

        # we zero grads here because we also call backward in the megatron-core fwd/bwd functions
        self._optimizer.zero_grad()

        if self.with_distributed_adam:
            # hack to enable overlapping param sync and forward compute
            # note: the distributed optimizer monkey-patches each
            # parameter's __getattribute__ function so that it can
            # launch parameter all-gathers the first time the
            # parameter is accessed after the optimizer step. However,
            # PyTorch directly passes embedding parameters into a C++,
            # bypassing this process. A quick-and-dirty hack is to
            # manually interact with the parameter.
            modules = self.model if isinstance(self.model, list) else [self.model]
            for module in modules:
                if isinstance(module, Float16Module):
                    module = module.module
                module = module.language_model
                if hasattr(module, 'embedding'):
                    for param in module.embedding.parameters():
                        param.data_ptr()

        loss_mean, loss_dict = self.fwd_bwd_step(dataloader_iter, batch_idx, False)

        # when using sequence parallelism, the sequence parallel layernorm grads must be all-reduced
        if self.cfg.get('tensor_model_parallel_size', 1) > 1 and self.cfg.get('sequence_parallel', False):
            self.allreduce_sequence_parallel_gradients()

        if self.with_distributed_adam:
            # synchronize asynchronous grad reductions
            # note: not necessary, but reduces performance degradation
            # from multiple simultaneous NCCL calls
            self._optimizer._finish_bucket_grad_sync()
        elif self.megatron_amp_O2:
            # when using pipeline parallelism grads must be all-reduced after the pipeline (not asynchronously)
            if self.cfg.get('pipeline_model_parallel_size', 1) > 1 or self.cfg.get('sequence_parallel', False):
                # main grads are stored in the MainParamsOptimizer wrapper
                self._optimizer.allreduce_main_grads()
        else:
            # async grad allreduce is not currently implemented for O1/autocasting mixed precision training
            # so we all-reduce gradients after the pipeline
            self.allreduce_gradients()  # @sangkug we think this is causing memory to blow up (hurts perf)

        if self.cfg.get('pipeline_model_parallel_size', 1) > 1 and self.cfg.get(
            'share_embeddings_and_output_weights', True
        ):
            # when using pipeline parallelism the first and last stage must keep embeddings in sync
            self.allreduce_first_last_embeddings()

        ## logging
        # we can only log on one rank if it is rank zero so we broadcast from last rank
        # we can avoid this broadcast by updating the PTL log function to accept specific ranks
        torch.distributed.broadcast(loss_mean, get_last_rank())

        if self.cfg.precision in [16, '16', '16-mixed']:
            loss_scale = self.trainer.precision_plugin.scaler._scale
            if loss_scale is not None:
                self.log('loss_scale', loss_scale, batch_size=1)

        self.log('reduced_train_loss', loss_mean, prog_bar=True, rank_zero_only=True, batch_size=1)
        self.log_dict({'train/' + k: v for k, v in loss_dict.items()}, rank_zero_only=True, batch_size=1)
        lr = self._optimizer.param_groups[0]['lr']
        self.log('lr', lr, rank_zero_only=True, batch_size=1)
        self.log(
            'global_step', self.trainer.global_step, prog_bar=True, rank_zero_only=True, batch_size=1,
        )

        consumed_samples = self.compute_consumed_samples(self.trainer.global_step - self.init_global_step)
        # TODO: make sure compute_consumed_samples works for pipeline parallelism
        self.log(
            'consumed_samples', consumed_samples, prog_bar=True, rank_zero_only=True, batch_size=1,
        )

        if self.cfg.get('rampup_batch_size', None):
            micro_batch_size = self.cfg.get('micro_batch_size', 1)
            total_gpus_number = self.trainer.num_devices * self.trainer.num_nodes
            current_global_batch_size = get_num_microbatches() * micro_batch_size * total_gpus_number
            self.log('global_batch_size', current_global_batch_size, prog_bar=True, rank_zero_only=True, batch_size=1)

            num_microbatch_calculator = apex.transformer.pipeline_parallel.utils._GLOBAL_NUM_MICROBATCHES_CALCULATOR
            num_microbatch_calculator.update(
                consumed_samples=consumed_samples, consistency_check=True,
            )

        return loss_mean

    def get_forward_output_and_loss_func(self, validation_step=False):
        def loss_func(output_tensors, loss_masks):
            loss_list = []
            loss_for_ub = 0
            for data_type in self.enabled_data_types:
                output_tensor = output_tensors[data_type]
                loss_mask = loss_masks[data_type]
                # Loss for a micro-batch (ub)
                loss_list.append(self.loss_func(loss_mask, output_tensor))
                loss_for_ub += loss_list[-1] * self.per_type_loss_weights[data_type]
            loss_for_ub /= sum(self.per_type_loss_weights.values())

            if validation_step and not self.cfg.data.get('validation_drop_last', True):
                raise NotImplementedError(f"`validation_drop_last=False` is not implemented in Kosmos!")
                # num_valid_tokens_in_ub = loss_mask.sum()
                # if loss_for_ub.isnan():
                #     assert loss_mask.count_nonzero() == 0, 'Got NaN loss with non-empty input'
                #     loss_sum_for_ub = torch.zeros_like(num_valid_tokens_in_ub)
                # else:
                #     loss_sum_for_ub = num_valid_tokens_in_ub * loss_for_ub
                #
                # loss_sum_and_ub_size_all_gpu = torch.cat(
                #     [
                #         loss_sum_for_ub.clone().detach().view(1),
                #         torch.tensor([num_valid_tokens_in_ub]).cuda().clone().detach(),
                #     ]
                # )
                # # Could potentially reduce num_valid_samples_in_microbatch and use that to aggregate instead of len(self._validation_ds)
                # torch.distributed.all_reduce(
                #     loss_sum_and_ub_size_all_gpu, group=parallel_state.get_data_parallel_group()
                # )
                # return loss_for_ub, {'loss_sum_and_ub_size': loss_sum_and_ub_size_all_gpu}
            else:
                reduced_loss = average_losses_across_data_parallel_group([loss_for_ub] + loss_list)
                loss_dict = {data_type: reduced_loss[i + 1] for i, data_type in enumerate(self.enabled_data_types)}
                loss_dict['avg'] = reduced_loss[0]
                return loss_for_ub, loss_dict

        def fwd_output_and_loss_func(dataloader_iter, model, checkpoint_activations_all_layers=None):
            output_tensors = {}
            loss_masks = {}
            combined_batch = next(dataloader_iter)
            for data_type in self.enabled_data_types:
                if parallel_state.get_pipeline_model_parallel_world_size() == 1:
                    batch = combined_batch[data_type]
                    for k in batch.keys():
                        if self.get_attention_mask_from_fusion:
                            batch[k] = batch[k].cuda(non_blocking=True) if k not in ['attention_mask'] else None
                        else:
                            batch[k] = batch[k].cuda(non_blocking=True)
                else:
                    if parallel_state.is_pipeline_first_stage():
                        batch = combined_batch[data_type]
                        # First pipeline stage needs tokens, position_ids, and attention_mask
                        for k in batch.keys():
                            if self.get_attention_mask_from_fusion:
                                batch[k] = (
                                    batch[k].cuda(non_blocking=True)
                                    if k in ['tokens', 'position_ids', 'media']
                                    else None
                                )
                            else:
                                batch[k] = (
                                    batch[k].cuda(non_blocking=True)
                                    if k in ['tokens', 'position_ids', 'attention_mask', 'media']
                                    else None
                                )
                    elif parallel_state.is_pipeline_last_stage():
                        batch = combined_batch[data_type]
                        # Last pipeline stage needs the labels, loss_mask, and attention_mask
                        for k in batch.keys():
                            if self.get_attention_mask_from_fusion:
                                batch[k] = batch[k].cuda(non_blocking=True) if k in ['labels', 'loss_mask'] else None
                            else:
                                batch[k] = (
                                    batch[k].cuda(non_blocking=True)
                                    if k in ['labels', 'loss_mask', 'attention_mask']
                                    else None
                                )
                    else:
                        # Intermediate pipeline stage doesn't need any inputs
                        batch = {k: None for k in ['tokens', 'position_ids', 'attention_mask', 'labels', 'media']}

                output_tensor = model(
                    batch['tokens'],
                    batch['position_ids'],
                    batch['attention_mask'],
                    batch['labels'],
                    batch.get('media'),
                    checkpoint_activations_all_layers=checkpoint_activations_all_layers,
                )
                output_tensors[data_type] = output_tensor
                loss_masks[data_type] = batch['loss_mask']

            return output_tensors, partial(loss_func, loss_masks=loss_masks)

        return fwd_output_and_loss_func

    def get_forward_output_only_func(self):
        def fwd_output_only_func(batch, model):
            extra_arg = {}
            if len(batch) == 3:
                batch = [x.cuda() for x in batch]
                tokens, attention_mask, position_ids = batch
                attention_mask = attention_mask[0:1]
            else:
                (
                    tokens,
                    attention_mask,
                    position_ids,
                    set_inference_key_value_memory,
                    inference_max_sequence_len,
                ) = batch
                tokens = tokens.cuda()
                attention_mask = attention_mask.cuda()
                position_ids = position_ids.cuda()
                attention_mask = attention_mask[0:1]
                extra_arg['set_inference_key_value_memory'] = set_inference_key_value_memory[0].item()
                extra_arg['inference_max_sequence_len'] = inference_max_sequence_len[0].item()
            output_tensor = model(tokens, position_ids, attention_mask, **extra_arg)

            def id_func(output_tensor):
                return output_tensor, {'logits': output_tensor}

            return output_tensor, id_func

        return fwd_output_only_func

    def validation_step(self, dataloader_iter, batch_idx):
        """
            Our dataloaders produce a micro-batch and then we fetch
            a number of microbatches depending on the global batch size and model parallel size
            from the dataloader to produce a list of microbatches.
            The list of microbatches is then piped through the pipeline using megatron-core fwd/bwd functions.
        """
        loss_mean, loss_dict = self.fwd_bwd_step(dataloader_iter, batch_idx, True)
        loss_dict['avg'] = loss_mean
        return loss_dict

    def validation_epoch_end(self, outputs):
        loss_dict = {}
        if parallel_state.is_pipeline_last_stage():
            # only the last pipeline parallel stages return loss with their batch size
            if self.cfg.data.get('validation_drop_last', True):
                averaged_loss = torch.stack([loss['avg'] for loss in outputs]).mean()
                for data_type in self.enabled_data_types:
                    loss_dict[data_type] = torch.stack([loss[data_type] for loss in outputs]).mean()
            else:
                # Compute the avg loss by total_loss across all samples / total number of samples
                # total_loss_and_total_samples = torch.vstack(outputs).sum(axis=0)
                # avg_loss = total_loss_and_total_samples[0] / total_loss_and_total_samples[1]
                # averaged_loss = avg_loss.type(torch.float32).cuda()
                raise NotImplementedError("`validation_drop_last=False` is not supported!")
        else:
            averaged_loss = torch.tensor(0.0, dtype=torch.float32).cuda()
            for data_type in self.enabled_data_types:
                loss_dict[data_type] = torch.tensor(0.0, dtype=torch.float32).cuda()

                # we can only log on one rank if it is rank zero so we broadcast from last rank
        torch.distributed.broadcast(averaged_loss, get_last_rank())
        for data_type in self.enabled_data_types:
            torch.distributed.broadcast(loss_dict[data_type], get_last_rank())

        self.log('val_loss', averaged_loss, prog_bar=True, rank_zero_only=True, batch_size=1)
        self.log_dict({'val/' + k: v for k, v in loss_dict.items()}, rank_zero_only=True, batch_size=1)

        return averaged_loss

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def test_epoch_end(self, outputs):
        averaged_loss = average_losses_across_data_parallel_group(outputs)
        logging.info(f'test_loss: {averaged_loss[0]}')

    def loss_func(self, loss_mask, output_tensor):
        losses = output_tensor.float()
        loss_mask = loss_mask.view(-1).float()
        # TODO: add nemo version here
        loss = torch.sum(losses.view(-1) * loss_mask) / loss_mask.sum()  # sequence level nll
        return loss

    def setup(self, stage=None):
        """ PTL hook that is executed after DDP spawns.
            We setup datasets here as megatron datasets require DDP to instantiate.
            See https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#setup for more information.
        Args:
            stage (str, optional): Can be 'fit', 'validate', 'test' or 'predict'. Defaults to None.
        """
        num_parameters_on_device, total_num_parameters = self._get_total_params_across_model_parallel_groups_gpt_bert(
            self.model
        )

        logging.info(
            f'Pipeline model parallel rank: {parallel_state.get_pipeline_model_parallel_rank()}, '
            f'Tensor model parallel rank: {parallel_state.get_tensor_model_parallel_rank()}, '
            f'Number of model parameters on device: {num_parameters_on_device:.2e}. '
            f'Total number of model parameters: {total_num_parameters:.2e}.'
        )

        resume_checkpoint_path = self.trainer.ckpt_path
        if resume_checkpoint_path:
            init_consumed_samples = self._extract_consumed_samples_from_ckpt(resume_checkpoint_path)
        else:
            init_consumed_samples = 0
        self.init_consumed_samples = init_consumed_samples
        self.init_global_step = self.trainer.global_step

        rampup_batch_size = self.cfg.get('rampup_batch_size', None)
        if rampup_batch_size:
            start_batch_size = rampup_batch_size[0]
            batch_size_increment = rampup_batch_size[1]
            total_gpus_number = self.trainer.num_devices * self.trainer.num_nodes

            assert start_batch_size % (total_gpus_number) == 0, (
                'expected'
                ' start batch size ({}) to be divisible by total number of GPUs'
                ' ({})'.format(start_batch_size, total_gpus_number)
            )

            micro_batch_size = self.cfg.get('micro_batch_size', 1)
            tensor_model_parallel_size = self.cfg.get('tensor_model_parallel_size', 1)
            pipeline_model_parallel_size = self.cfg.get('pipeline_model_parallel_size', 1)
            total_data_parallel_size = total_gpus_number // (tensor_model_parallel_size * pipeline_model_parallel_size)

            assert batch_size_increment % (micro_batch_size * total_data_parallel_size) == 0, (
                'expected'
                ' batch size increment ({}) to be divisible by micro_batch_size ({}) times total data parallel size'
                ' ({})'.format(batch_size_increment, micro_batch_size, total_data_parallel_size)
            )

        if stage == 'predict':
            return
        else:
            # TODO: consider adding a ModelPT guard to check if model is being restored.
            # allowing restored models to optionally setup datasets
            self.build_train_valid_test_datasets()
            self.setup_training_data(self.cfg.data)
            self.setup_validation_data(self.cfg.data)
            self.setup_test_data(self.cfg.data)

        # when using pipeline model parallel the final stage need to initialize word embeddings
        if parallel_state.get_pipeline_model_parallel_world_size() > 1:
            if isinstance(self.model, list):
                for i, module in enumerate(self.model):
                    parallel_state.set_virtual_pipeline_model_parallel_rank(i)
                    if self.cfg.get('share_embeddings_and_output_weights', True):
                        module.sync_initial_word_embeddings()
                parallel_state.set_virtual_pipeline_model_parallel_rank(0)
            else:
                if self.cfg.get('share_embeddings_and_output_weights', True):
                    self.model.sync_initial_word_embeddings()

        if self.cfg.get('transformer_engine', False):
            self.setup_transformer_engine_tp_groups()

    def build_train_valid_test_datasets(self):
        logging.info('Building Kosmos datasets.')

        if self.trainer.limit_val_batches > 1.0 and isinstance(self.trainer.limit_val_batches, float):
            raise ValueError("limit_val_batches must be an integer or float less than or equal to 1.0.")

        global_batch_size = self.cfg.global_batch_size
        max_train_steps = self.trainer.max_steps
        eval_iters = (max_train_steps // self.trainer.val_check_interval + 1) * self.trainer.limit_val_batches
        test_iters = self.trainer.limit_test_batches

        train_valid_test_num_samples = [
            max_train_steps * global_batch_size,
            eval_iters * global_batch_size,
            test_iters * global_batch_size,
        ]

        if self.trainer.limit_val_batches <= 1.0 and isinstance(self.trainer.limit_val_batches, float):
            train_valid_test_num_samples[
                1
            ] = 1  # This is to make sure we only have one epoch on every validation iteration

        self._train_ds, self._validation_ds, self._test_ds = {}, {}, {}

        for data_type in self.enabled_data_types:
            if data_type == "text":
                (
                    self._train_ds[data_type],
                    self._validation_ds[data_type],
                    self._test_ds[data_type],
                ) = build_text_train_valid_test_datasets(
                    cfg=self.cfg,
                    trainer=self.trainer,
                    data_prefix=self.cfg.data.data_prefix,
                    data_impl=self.cfg.data.data_impl,
                    splits_string=self.cfg.data.splits_string,
                    train_valid_test_num_samples=train_valid_test_num_samples,
                    seq_length=self.cfg.data.seq_length,
                    seed=self.cfg.seed,
                    skip_warmup=self.cfg.data.get('skip_warmup', True),
                    tokenizer=self.tokenizer,
                )

            if data_type in ["image_caption", "image_interleaved"]:
                self._train_ds[data_type], self._validation_ds[data_type] = build_media_train_valid_datasets(
                    model_cfg=self.cfg,
                    consumed_samples=self.compute_consumed_samples(0)
                    * self.per_type_micro_batch_size[data_type]
                    // self.cfg.micro_batch_size,
                    tokenizer=self.tokenizer,
                    data_type=data_type,
                )
                self._test_ds[data_type] = None

        data = []
        for ds_name, ds in [("Train", self._train_ds), ("Validation", self._validation_ds), ("Test", self._test_ds)]:
            for key in self.enabled_data_types:
                # Append the name of the dataset, the key, and the length of the data under that key to the list
                if ds_name == "Train":
                    consumed_samples = (
                        self.compute_consumed_samples(0)
                        * self.per_type_micro_batch_size[key]
                        // self.cfg.micro_batch_size
                    )
                else:
                    consumed_samples = 0
                data.append([ds_name, key, len(ds[key]) if ds[key] is not None else 0, consumed_samples])

        df = pd.DataFrame(data, columns=["Dataset", "Type", "Length", "Consumed"])
        df['Length'] = df['Length'].apply(lambda x: "{:,}".format(x))
        df['Consumed'] = df['Consumed'].apply(lambda x: "{:,}".format(x))

        logging.info(f"\nFinished Building Kosmos Dataset:\n{df}")
        return self._train_ds, self._validation_ds, self._test_ds

    def build_pretraining_text_data_loader(
        self,
        dataset,
        consumed_samples,
        micro_batch_size,
        global_batch_size,
        drop_last=True,
        pad_samples_to_global_batch_size=False,
    ):
        """Buld dataloader given an input dataset."""

        logging.info(f'Building dataloader with consumed samples: {consumed_samples}')
        # Megatron sampler
        if hasattr(self.cfg.data, 'dataloader_type') and self.cfg.data.dataloader_type is not None:
            if self.cfg.data.dataloader_type == 'single':
                batch_sampler = MegatronPretrainingSampler(
                    total_samples=len(dataset),
                    consumed_samples=consumed_samples,
                    micro_batch_size=micro_batch_size,
                    data_parallel_rank=parallel_state.get_data_parallel_rank(),
                    data_parallel_size=parallel_state.get_data_parallel_world_size(),
                    drop_last=drop_last,
                    global_batch_size=global_batch_size,
                    pad_samples_to_global_batch_size=pad_samples_to_global_batch_size,
                )
            elif self.cfg.data.dataloader_type == 'cyclic':
                batch_sampler = MegatronPretrainingRandomSampler(
                    total_samples=len(dataset),
                    consumed_samples=consumed_samples,
                    micro_batch_size=micro_batch_size,
                    data_parallel_rank=parallel_state.get_data_parallel_rank(),
                    data_parallel_size=parallel_state.get_data_parallel_world_size(),
                    drop_last=self.cfg.get('drop_last', True),
                    global_batch_size=global_batch_size,
                )
            else:
                raise ValueError('cfg.data.dataloader_type must be "single" or "cyclic"')
        else:
            raise ValueError('cfg.data.dataloader_type not found. Must be "single" or "cyclic"')

        return torch.utils.data.DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            num_workers=self.cfg.data.num_workers,
            pin_memory=True,
            persistent_workers=True if self.cfg.data.num_workers > 0 else False,
        )

    @classmethod
    def list_available_models(cls) -> Optional[PretrainedModelInfo]:
        """
        This method returns a list of pre-trained model which can be instantiated directly from NVIDIA's NGC cloud.
        Returns:
            List of available pre-trained models.
        """
        return []

    def setup_training_data(self, cfg):
        consumed_samples = self.compute_consumed_samples(0)

        train_dls = {}
        for data_type in self.enabled_data_types:
            if hasattr(self, '_train_ds') and self._train_ds.get(data_type) is not None:
                if data_type == "text":
                    train_dls[data_type] = self.build_pretraining_text_data_loader(
                        self._train_ds[data_type],
                        consumed_samples=consumed_samples
                        * self.per_type_micro_batch_size[data_type]
                        // self.cfg.micro_batch_size,
                        micro_batch_size=self.per_type_micro_batch_size[data_type],
                        global_batch_size=self.per_type_global_batch_size[data_type],
                    )
                elif data_type in ["image_caption", "image_interleaved"]:
                    train_dls[data_type] = torch.utils.data.DataLoader(
                        self._train_ds[data_type],
                        batch_size=self.per_type_micro_batch_size[data_type],
                        num_workers=cfg.get(data_type).num_workers,
                        pin_memory=True,
                        drop_last=True,
                        persistent_workers=True,
                    )
                else:
                    raise ValueError(f"Unrecognized dataset type {data_type}")

        self._train_dl = MergedKosmosDataLoader(train_dls)

    def setup_validation_data(self, cfg):
        consumed_samples = 0

        validation_dls = {}
        for data_type in self.enabled_data_types:
            if hasattr(self, '_validation_ds') and self._validation_ds.get(data_type) is not None:
                if data_type == "text":
                    validation_dls[data_type] = self.build_pretraining_text_data_loader(
                        self._validation_ds[data_type],
                        consumed_samples=consumed_samples,
                        micro_batch_size=self.per_type_micro_batch_size[data_type],
                        global_batch_size=self.per_type_global_batch_size[data_type],
                    )
                elif data_type in ["image_caption", "image_interleaved"]:
                    validation_dls[data_type] = torch.utils.data.DataLoader(
                        self._validation_ds[data_type],
                        batch_size=self.per_type_micro_batch_size[data_type],
                        num_workers=cfg.num_workers,
                        pin_memory=True,
                        drop_last=True,
                        persistent_workers=True,
                    )
                else:
                    raise ValueError(f"Unrecognized dataset type {data_type}")

        self._validation_dl = MergedKosmosDataLoader(validation_dls)

    def setup_test_data(self, cfg):
        pass
