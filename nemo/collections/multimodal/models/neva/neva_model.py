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
import re
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
from transformers import CLIPVisionModel

from nemo.collections.multimodal.data.neva.neva_dataset import (
    DEFAULT_BOS_TOKEN,
    DEFAULT_EOS_TOKEN,
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DataCollatorForSupervisedDataset,
    make_supervised_data_module,
)
from nemo.collections.multimodal.models.clip.megatron_clip_models import CLIPVisionTransformer, MegatronCLIPModel
from nemo.collections.multimodal.models.kosmos.perceiver_resampler import PerceiverResampler
from nemo.collections.multimodal.parts.utils import extend_instance
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
from nemo.collections.nlp.models.language_modeling.megatron_gpt_peft_models import MegatronGPTPEFTModel
from nemo.collections.nlp.models.nlp_model import NLPModel
from nemo.collections.nlp.modules.common.megatron.adapters.parallel_adapters import AdapterName, MMLinearAdapterConfig
from nemo.collections.nlp.modules.common.megatron.build_model import build_model
from nemo.collections.nlp.modules.common.megatron.language_model import Embedding, get_language_model
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
    megatron_neva_generate,
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
from nemo.core import adapter_mixins
from nemo.core.classes.common import PretrainedModelInfo
from nemo.utils import AppState, logging, model_utils

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
    def __init__(self, model_cfg, model_parallel_config, pre_process=True, post_process=True):
        super().__init__(
            model_cfg, model_parallel_config, pre_process=pre_process, post_process=post_process, skip_head=True,
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


class NevaWordEmbeddingMixin(torch.nn.Module, adapter_mixins.AdapterModuleMixin):
    def init_vision(
        self,
        vision_encoder,
        media_start_id,
        media_end_id,
        vision_select_layer=-1,
        class_token_length=1,
        use_im_start_end=False,
        llama_tricks=False,
    ):
        self.vision_encoder = vision_encoder
        self.from_hf = isinstance(vision_encoder, CLIPVisionModel)
        self.media_start_id = media_start_id
        self.media_end_id = media_end_id
        self.class_token_length = class_token_length
        self.use_im_start_end = use_im_start_end
        self.vision_select_layer = vision_select_layer
        self.media = None
        self.set_accepted_adapter_types([MMLinearAdapterConfig._target_])
        self.llama_tricks = llama_tricks

    def set_media(self, media):
        self.media = media

    def forward(self, input_ids, **kwargs):
        media = self.media  # avoid change the signature of embedding forward function
        if self.llama_tricks and not self.use_im_start_end:
            masked_input_ids = input_ids.detach().clone()
            if self.num_embeddings < 32000:
                raise ValueError("Not supported tokenizer with llama 2!")
            else:
                masked_input_ids[masked_input_ids >= 32000] = 0
            words_embeddings = super().forward(masked_input_ids, **kwargs)

        else:
            words_embeddings = super().forward(input_ids, **kwargs)

        return self.replace_media_embeddings(input_ids, words_embeddings, media)

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
            if self.from_hf:
                vision_x = self.vision_encoder(vision_x, output_hidden_states=True)
                vision_x = vision_x.hidden_states[self.vision_select_layer]
            else:
                self.vision_encoder.backbone.transformer.return_select_layer = self.vision_select_layer
                vision_x = self.vision_encoder(vision_x)
        vision_x = rearrange(vision_x, "(b T F) v d -> b T F v d", b=b, T=T, F=F)
        vision_x = vision_x[:, :, :, self.class_token_length :]
        assert self.is_adapter_available(), "Cannot find multimodal vision adapter!"
        vision_connector = self.get_adapter_module(AdapterName.MM_LINEAR_ADAPTER)
        vision_x = vision_connector(vision_x)
        return vision_x

    def replace_media_embeddings(self, input_ids, inputs_embeds, media):
        if media is None:
            return inputs_embeds

        batch_size, sequence_length, hidden_size = inputs_embeds.shape

        # calculate media features without gradients
        media_features = self.encode_vision_x(media)  # b T F S(eq) H(idden)
        num_images_per_sample = media_features.size(1)
        num_patches = media_features.size(3)
        # flatten patches
        media_features = media_features.view(batch_size, -1, hidden_size)

        # create an indices matrix used in torch.scatter
        padded_media_indices = torch.ones(
            (batch_size, num_images_per_sample), dtype=torch.long, device=input_ids.device
        )
        padded_media_indices *= sequence_length
        for idx, input_id in enumerate(input_ids):
            media_end_positions = torch.where(input_id == self.media_end_id)[0]
            if self.use_im_start_end:
                # locate the first media token positions
                padded_media_indices[idx, : len(media_end_positions)] = media_end_positions - num_patches
                assert (
                    input_id[padded_media_indices[idx, : len(media_end_positions)] - 1] == self.media_start_id
                ).all()
            else:
                padded_media_indices[idx, : len(media_end_positions)] = media_end_positions - num_patches + 1
                assert (input_id[padded_media_indices[idx, : len(media_end_positions)]] == self.media_start_id).all()

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


class NevaModel(GPTModel):
    def __init__(
        self, mm_cfg, media_start_id, media_end_id, **kwargs,
    ):
        super(NevaModel, self).__init__(**kwargs,)

        self.mm_cfg = mm_cfg
        self.media_start_id = media_start_id
        self.media_end_id = media_end_id

        if mm_cfg.llm.from_pretrained is not None:
            logging.info(f"Loading LLM weights from checkpoint {mm_cfg.llm.from_pretrained}")
            self.load_llm_weights(self.language_model, mm_cfg.llm.from_pretrained)
        if mm_cfg.llm.freeze:
            for param in self.language_model.parameters():
                param.requires_grad = False
            self.language_model = self.language_model.eval()

        # Initialize vision encoder and freeze it
        if mm_cfg.vision_encoder.from_hf:
            vision_encoder = CLIPVisionModel.from_pretrained(
                mm_cfg.vision_encoder.from_pretrained, torch_dtype=torch.bfloat16,
            ).cuda()
            vision_encoder = vision_encoder.to(torch.bfloat16)
            if mm_cfg.vision_encoder.freeze:
                for param in vision_encoder.parameters():
                    param.requires_grad = False
                vision_encoder = vision_encoder.eval()
        else:
            vision_cfg = MegatronCLIPModel.restore_from(
                mm_cfg.vision_encoder.from_pretrained, return_config=True
            ).vision
            vision_encoder = FrozenCLIPVisionTransformer(vision_cfg, self.config)
            self.load_vision_encoder_weights(vision_encoder, mm_cfg.vision_encoder.from_pretrained)
            if mm_cfg.vision_encoder.freeze:
                vision_encoder.freeze()

        model_type = self.mm_cfg.llm.get("model_type", "nvgpt")
        # Monkey patch embedding
        if kwargs.get("pre_process", True):
            extend_instance(self.language_model.embedding.word_embeddings, NevaWordEmbeddingMixin)
            self.language_model.embedding.word_embeddings.init_vision(
                vision_encoder,
                media_start_id,
                media_end_id,
                vision_select_layer=mm_cfg.vision_encoder.get("vision_select_layer", -2),
                class_token_length=mm_cfg.vision_encoder.get("class_token_length", 1),
                use_im_start_end=mm_cfg.get("use_im_start_end", False),
                llama_tricks=(model_type == "llama_2"),
            )

    def forward(
        self, *args, **kwargs,
    ):
        media = args[-1]
        self.language_model.embedding.word_embeddings.set_media(media)
        return super().forward(*args[:-1], **kwargs)

    def _load_model_weights(self, nemo_path):
        """
        Shared method to load model weights from a given nemo_path.
        """
        if torch.cuda.is_available():
            map_location = torch.device('cuda')
        else:
            map_location = torch.device('cpu')

        save_restore_connector = NLPSaveRestoreConnector()
        cwd = os.getcwd()
        app_state = AppState()

        with tempfile.TemporaryDirectory() as tmpdir:
            try:
                if os.path.isfile(nemo_path):
                    save_restore_connector._unpack_nemo_file(path2file=nemo_path, out_folder=tmpdir)
                else:
                    tmpdir = nemo_path
                os.chdir(tmpdir)
                if app_state.model_parallel_size is not None and app_state.model_parallel_size > 1:
                    model_weights = save_restore_connector._inject_model_parallel_rank_for_ckpt(
                        tmpdir, save_restore_connector.model_weights_ckpt
                    )
                else:
                    model_weights = os.path.join(tmpdir, save_restore_connector.model_weights_ckpt)

                state_dict = save_restore_connector._load_state_dict_from_disk(
                    model_weights, map_location=map_location
                )
            finally:
                os.chdir(cwd)

        return state_dict

    def load_vision_encoder_weights(self, vision_encoder, nemo_path):
        state_dict = self._load_model_weights(nemo_path)

        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("model.vision_encoder."):
                new_k = k.replace("model.vision_encoder.", "")
                new_state_dict[new_k] = v

        missing, unexpected = vision_encoder.load_state_dict(new_state_dict, strict=False)
        print(f"Restored from {nemo_path} with {len(missing)} missing and {len(unexpected)} unexpected keys")
        if len(missing) > 0:
            print(f"Missing Keys: {missing}")
        if len(unexpected) > 0:
            print(f"Unexpected Keys: {unexpected}")

    def load_llm_weights(self, language_model, nemo_path):
        state_dict = self._load_model_weights(nemo_path)

        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("model.language_model."):
                new_k = k.replace("model.language_model.", "", 1)
                module_key, param_key = new_k.split(".", 1)
                if module_key not in new_state_dict:
                    new_state_dict[module_key] = {}
                new_state_dict[module_key][param_key] = v

        language_model.load_state_dict(new_state_dict, strict=True)
        print(f"Restored LLM weights from {nemo_path}.")


class MegatronNevaModel(MegatronGPTPEFTModel):
    """
    Megatron Neva pretraining
    """

    def __init__(self, cfg: DictConfig, trainer: Trainer):
        if getattr(self, "peft_name_keys", None) is None:
            self.peft_name_keys = []
            self.name_key_to_cfg = {}

        self.peft_name_keys += [AdapterName.MM_LINEAR_ADAPTER]
        adapter_cfg = MMLinearAdapterConfig(
            in_features=cfg.mm_cfg.vision_encoder.hidden_size, out_features=cfg.hidden_size, bias=True,
        )
        self.name_key_to_cfg.update(
            {AdapterName.MM_LINEAR_ADAPTER: adapter_cfg,}
        )
        MegatronGPTModel.__init__(self, cfg, trainer)

        self.setup_complete = False
        self.base_keys = self.get_all_keys()
        self.init_peft_modules()
        self.adapter_keys = self.get_all_keys() - self.base_keys
        if self.megatron_amp_O2:
            self.adapter_keys = set(key.replace("model.module.", "model.", 1) for key in self.adapter_keys)

    def get_all_keys(self,):
        # TODO (yuya): p-tuning need additional handle, check peft models.
        """
        Returns all the keys in the model
        """
        k = [n for n, p in self.named_parameters()]
        return set(k)

    def init_peft_modules(self):
        """
        Randomly initialize the peft params and add them to the appropriate modules.
        """
        assert len(self.peft_name_keys) > 0, "peft_name_keys have not been set no PEFT modules will be added"
        assert len(self.name_key_to_cfg) > 0, "name_key_to_cfg has not been set no PEFT modules will be added"
        logging.info(f"Before adding PEFT params:\n{self.summarize()}")
        for _, module in self.named_modules():
            if isinstance(module, adapter_mixins.AdapterModuleMixin):
                for peft_key in self.peft_name_keys:
                    peft_cfg = self.name_key_to_cfg[peft_key]
                    if model_utils.import_class_by_path(peft_cfg._target_) in module.get_accepted_adapter_types():
                        module.add_adapter(
                            name=peft_key,
                            cfg=peft_cfg,  # TODO (yuya): override this line in gpt peft models due to a conf merging issue
                        )
                if self.megatron_amp_O2:
                    for adapter_name in getattr(module, 'adapter_layer', []):
                        module.adapter_layer[adapter_name] = module.adapter_layer[adapter_name].to(self.autocast_dtype)
        logging.info(f"After adding PEFT params:\n{self.summarize()}")
        return True

    def model_provider_func(self, pre_process, post_process):
        """Model depends on pipeline paralellism."""
        media_start_id = self.tokenizer.token_to_id(DEFAULT_IM_START_TOKEN)
        media_end_id = self.tokenizer.token_to_id(DEFAULT_IM_END_TOKEN)

        model = NevaModel(
            mm_cfg=self.cfg.mm_cfg,
            media_start_id=media_start_id,
            media_end_id=media_end_id,
            config=self.model_parallel_config,
            vocab_size=self.cfg.get('override_vocab_size', self.padded_vocab_size),
            hidden_size=self.cfg.hidden_size,
            max_position_embeddings=self.cfg.max_position_embeddings,
            num_layers=self.cfg.num_layers,
            num_attention_heads=self.cfg.num_attention_heads,
            apply_query_key_layer_scaling=self.cfg.get('apply_query_key_layer_scaling', True),
            kv_channels=self.cfg.get('kv_channels', None),
            ffn_hidden_size=self.cfg.ffn_hidden_size,
            num_tokentypes=0,
            parallel_output=True,
            pre_process=pre_process,
            post_process=post_process,
            init_method_std=self.cfg.get('init_method_std', 0.02),
            use_scaled_init_method=self.cfg.get('use_scaled_init_method', True),
            fp16_lm_cross_entropy=self.cfg.get('fp16_lm_cross_entropy', False),
            megatron_amp_O2=self.cfg.get('megatron_amp_O2', False),
            hidden_dropout=self.cfg.get('hidden_dropout', 0.1),
            attention_dropout=self.cfg.get('attention_dropout', 0.1),
            ffn_dropout=self.cfg.get('ffn_dropout', 0.0),
            precision=self.cfg.get('precision', 16),
            fp32_residual_connection=self.cfg.get('fp32_residual_connection', False),
            activations_checkpoint_granularity=self.cfg.get('activations_checkpoint_granularity', None),
            activations_checkpoint_method=self.cfg.get('activations_checkpoint_method', None),
            activations_checkpoint_num_layers=self.cfg.get('activations_checkpoint_num_layers', 1),
            activations_checkpoint_layers_per_pipeline=self.cfg.get(
                'activations_checkpoint_layers_per_pipeline', None
            ),
            normalization=self.cfg.get('normalization', 'layernorm'),
            layernorm_epsilon=self.cfg.get('layernorm_epsilon', 1e-5),
            onnx_safe=self.cfg.get('onnx_safe', False),
            bias=self.cfg.get('bias', True),
            bias_activation_fusion=self.cfg.get('bias_activation_fusion', True),
            bias_dropout_add_fusion=self.cfg.get('bias_dropout_add_fusion', True),
            activation=self.cfg.get('activation', 'gelu'),
            headscale=self.cfg.get('headscale', False),
            transformer_block_type=self.cfg.get('transformer_block_type', 'pre_ln'),
            openai_gelu=self.cfg.get('openai_gelu', False),
            normalize_attention_scores=self.cfg.get('normalize_attention_scores', True),
            position_embedding_type=self.cfg.get('position_embedding_type', 'learned_absolute'),
            rotary_percentage=self.cfg.get('rotary_percentage', 1.0),
            share_embeddings_and_output_weights=self.cfg.get('share_embeddings_and_output_weights', True),
            attention_type=self.cfg.get('attention_type', 'multihead'),
            masked_softmax_fusion=self.cfg.get('masked_softmax_fusion', True),
            persist_layer_norm=self.cfg.get('persist_layer_norm', False),
            transformer_engine=self.cfg.get('transformer_engine', False),
            fp8=self.cfg.get('fp8', False),
            fp8_e4m3=self.cfg.get('fp8_e4m3', False),
            fp8_hybrid=self.cfg.get('fp8_hybrid', False),
            fp8_margin=self.cfg.get('fp8_margin', 0),
            fp8_interval=self.cfg.get('fp8_interval', 1),
            fp8_amax_history_len=self.cfg.get('fp8_amax_history_len', 1),
            fp8_amax_compute_algo=self.cfg.get('fp8_amax_compute_algo', 'most_recent'),
            reduce_amax=self.cfg.get('reduce_amax', True),
            use_emha=self.cfg.get('use_emha', False),
            ub_tp_comm_overlap=self.cfg.get('ub_tp_comm_overlap', False),
            use_flash_attention=self.cfg.get('use_flash_attention', False),
            megatron_legacy=self.cfg.get('megatron_legacy', False),
            seq_len_interpolation_factor=self.cfg.get('seq_len_interpolation_factor', None),
        )

        logging.info(
            f"Neva model initialized with {sum(p.numel() for p in model.parameters() if p.requires_grad)} trainable parameters"
        )

        return model

    def setup_optimizer_param_groups(self):
        """ModelPT override. Optimizer will get self._optimizer_param_groups"""
        if self.cfg.mm_cfg.llm.freeze:
            super().setup_optimizer_param_groups()
        else:
            MegatronGPTModel.setup_optimizer_param_groups(self)

        # filter out params doesn't have grad
        for param_group in self._optimizer_param_groups:
            params_with_grad = [param for param in param_group['params'] if param.requires_grad]
            param_group['params'] = params_with_grad

    def forward(self, tokens, text_position_ids, attention_mask, labels, media=None):
        output_tensor = self.model(tokens, text_position_ids, attention_mask, labels, media)
        return output_tensor

    def fwd_bwd_step(self, dataloader_iter, batch_idx, forward_only):
        return MegatronGPTModel.fwd_bwd_step(self, dataloader_iter, batch_idx, forward_only)

    def training_step(self, dataloader_iter, batch_idx):
        """
            We pass the dataloader iterator function to the micro-batch scheduler.
            The input batch to each micro-batch is fetched using the dataloader function
            in the micro-batch fwd function.
        """
        return MegatronGPTModel.training_step(self, dataloader_iter, batch_idx)

    def get_forward_output_and_loss_func(self, validation_step=False):
        def loss_func(output_tensor, loss_mask):
            loss_for_ub = self.loss_func(loss_mask, output_tensor)
            if validation_step and not self.cfg.data.get('validation_drop_last', True):
                raise NotImplementedError(f"`validation_drop_last=False` is not implemented in Neva!")
            else:
                reduced_loss = average_losses_across_data_parallel_group([loss_for_ub])
                return loss_for_ub, dict(avg=reduced_loss[0].unsqueeze(0))

        def fwd_output_and_loss_func(dataloader_iter, model, checkpoint_activations_all_layers=None):
            batch = next(dataloader_iter)
            if parallel_state.get_pipeline_model_parallel_world_size() == 1:
                for k in batch.keys():
                    if self.get_attention_mask_from_fusion:
                        batch[k] = batch[k].cuda(non_blocking=True) if k not in ['attention_mask'] else None
                    else:
                        batch[k] = batch[k].cuda(non_blocking=True)
            else:
                if parallel_state.is_pipeline_first_stage():
                    # First pipeline stage needs tokens, position_ids, and attention_mask
                    for k in batch.keys():
                        if self.get_attention_mask_from_fusion:
                            batch[k] = (
                                batch[k].cuda(non_blocking=True) if k in ['tokens', 'position_ids', 'media'] else None
                            )
                        else:
                            batch[k] = (
                                batch[k].cuda(non_blocking=True)
                                if k in ['tokens', 'position_ids', 'attention_mask', 'media']
                                else None
                            )
                elif parallel_state.is_pipeline_last_stage():
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
                None,  # placehpolder for loss mask
                batch['labels'],
                batch.get('media'),
                checkpoint_activations_all_layers=checkpoint_activations_all_layers,
            )

            return output_tensor, partial(loss_func, loss_mask=batch['loss_mask'])

        return fwd_output_and_loss_func

    def get_forward_output_only_func(self):
        def fwd_output_only_func(dataloader_iter, model):
            batch = next(dataloader_iter)
            extra_arg = {}
            (
                tokens,
                attention_mask,
                position_ids,
                media,
                set_inference_key_value_memory,
                inference_max_sequence_len,
            ) = batch
            tokens = tokens.cuda()
            attention_mask = attention_mask.cuda()
            position_ids = position_ids.cuda()
            attention_mask = attention_mask[0:1]
            if media is not None:
                media = media.cuda()
            labels = None
            extra_arg['set_inference_key_value_memory'] = set_inference_key_value_memory[0].item()
            extra_arg['inference_max_sequence_len'] = inference_max_sequence_len[0].item()
            # TODO : Should I add labels ?
            output_tensor = model(tokens, position_ids, attention_mask, labels, media, **extra_arg)

            def id_func(output_tensor):
                return output_tensor, {'logits': output_tensor}

            return output_tensor, id_func

        return fwd_output_only_func

    def validation_step(self, dataloader_iter, batch_idx):
        return MegatronGPTModel.validation_step(self, dataloader_iter, batch_idx)

    def on_validation_epoch_end(self):
        if not self.validation_step_outputs:
            return

        if parallel_state.is_pipeline_last_stage():
            # only the last pipeline parallel stages return loss with their batch size
            if self.cfg.data.get('validation_drop_last', True):
                averaged_loss = torch.stack(self.validation_step_outputs).mean()
            else:
                # Compute the avg loss by total_loss across all samples / total number of samples
                # total_loss_and_total_samples = torch.vstack(outputs).sum(axis=0)
                # avg_loss = total_loss_and_total_samples[0] / total_loss_and_total_samples[1]
                # averaged_loss = avg_loss.type(torch.float32).cuda()
                raise NotImplementedError("`validation_drop_last=False` is not supported!")
        else:
            averaged_loss = torch.tensor(0.0, dtype=torch.float32).cuda()

        # we can only log on one rank if it is rank zero so we broadcast from last rank
        torch.distributed.broadcast(averaged_loss, get_last_rank())
        self.log('val_loss', averaged_loss, prog_bar=True, rank_zero_only=True, batch_size=1)
        self.validation_step_outputs.clear()  # free memory

        return averaged_loss

    def on_validation_epoch_start(self):
        pass

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
        logging.info('Building Neva datasets.')
        ds_dict = make_supervised_data_module(tokenizer=self.tokenizer, model_cfg=self.cfg,)
        self._train_ds = ds_dict["train_dataset"]
        self._validation_ds = ds_dict["eval_dataset"]

        return self._train_ds, self._validation_ds

    def build_pretraining_data_loader(
        self, dataset, consumed_samples, dataset_type=None, drop_last=True, pad_samples_to_global_batch_size=False
    ):
        """Buld dataloader given an input dataset."""

        logging.info(f'Building dataloader with consumed samples: {consumed_samples}')
        # Megatron sampler
        if hasattr(self.cfg.data, 'dataloader_type') and self.cfg.data.dataloader_type is not None:
            if self.cfg.data.dataloader_type == 'single':
                batch_sampler = MegatronPretrainingSampler(
                    total_samples=len(dataset),
                    consumed_samples=consumed_samples,
                    micro_batch_size=self.cfg.micro_batch_size,
                    data_parallel_rank=parallel_state.get_data_parallel_rank(),
                    data_parallel_size=parallel_state.get_data_parallel_world_size(),
                    drop_last=drop_last,
                    global_batch_size=self.cfg.global_batch_size,
                    pad_samples_to_global_batch_size=pad_samples_to_global_batch_size,
                )
            elif self.cfg.data.dataloader_type == 'cyclic':
                batch_sampler = MegatronPretrainingRandomSampler(
                    total_samples=len(dataset),
                    consumed_samples=consumed_samples,
                    micro_batch_size=self.cfg.micro_batch_size,
                    data_parallel_rank=parallel_state.get_data_parallel_rank(),
                    data_parallel_size=parallel_state.get_data_parallel_world_size(),
                    drop_last=self.cfg.get('drop_last', True),
                )
            else:
                raise ValueError('cfg.data.dataloader_type must be "single" or "cyclic"')
        else:
            raise ValueError('cfg.data.dataloader_type not found. Must be "single" or "cyclic"')

        collate_func = DataCollatorForSupervisedDataset(self.cfg, self.tokenizer)
        return torch.utils.data.DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            collate_fn=collate_func,
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

    def setup_test_data(self, cfg):
        pass

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        # Get the original state dictionary
        original_state_dict = super().state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)
        keys_to_keep = list(self.adapter_keys)
        # TODO(yuya): maybe not hard-code vision_encoder keys here
        if self.megatron_amp_O2:
            vision_encoder_keys = [
                k.replace("model.module.", "model.", 1) for k in self.base_keys if "vision_encoder" in k
            ]
            llm_keys = [k.replace("model.module.", "model.", 1) for k in self.base_keys if "vision_encoder" not in k]
        else:
            vision_encoder_keys = [k for k in self.base_keys if "vision_encoder" in k]
            llm_keys = [k for k in self.base_keys if "vision_encoder" not in k]
        if not self.cfg.mm_cfg.llm.freeze:
            keys_to_keep += llm_keys
        if not self.cfg.mm_cfg.vision_encoder.freeze:
            keys_to_keep += vision_encoder_keys
        return {k: original_state_dict[k] for k in keys_to_keep if k in original_state_dict}

    def load_state_dict(self, state_dict, strict=False):
        logging.warning('Loading state dict for MegatronNevaModel...')
        missing_keys, unexpected_keys = NLPModel.load_state_dict(self, state_dict, strict=False)

        if len(missing_keys) > 0:
            logging.warning('Missing keys were detected during the load. Please double check.')
            logging.warning(f'Missing keys: \n{missing_keys}')
        if len(unexpected_keys) > 0:
            logging.critical('Unexpected keys were detected during the load. Please double check.')
            logging.critical(f'Unexpected keys: \n{unexpected_keys}')

    def sharded_state_dict(self, prefix: str = ''):
        return None

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: Optional[int] = None) -> Any:
        inference_config = self.get_inference_config()

        if inference_config is None:
            return None
        else:
            # need to overwrite some configuration, make it immutable
            image = os.path.join(inference_config['images_base_path'], batch['image'][0])
            prompt = batch['prompt'][0]
            inference_config = inference_config.copy()
            compute_logprob = inference_config['compute_logprob']
            if compute_logprob:
                inference_config['inputs'] = prompt
                inference_config['tokens_to_generate'] = 1
                inference_config['all_probs'] = True
                inference_config["add_BOS"] = False
                inference_config['greedy'] = True
                inference_config['image_list'] = image
                response = generate(self, **inference_config)
                compute_prob_response = get_computeprob_response(self.tokenizer, response, prompt)
                return compute_prob_response
            else:
                inference_config['inputs'] = prompt
                inference_config['image_list'] = image
                return generate(self, **inference_config)

    def generate(
        self, input_prompts, inference_config, length_params: LengthParam, sampling_params: SamplingParam = None,
    ) -> OutputType:

        # check whether the DDP is initialized
        if parallel_state.is_unitialized():

            def dummy():
                return

            import os

            if self.trainer.strategy.launcher is not None:
                self.trainer.strategy.launcher.launch(dummy, trainer=self.trainer)
            self.trainer.strategy.setup_environment()

        # set the default sampling params if it is None.
        # default do greedy sampling
        if sampling_params is None:
            sampling_params = get_default_sampling_params()

        # set the default length params if it is None.
        # default do greedy sampling
        if length_params is None:
            length_params = get_default_length_params()

        import time

        start = time.time()
        # Supports only one prompt at a time
        result = megatron_neva_generate(self.cuda(), input_prompts, length_params, sampling_params, inference_config)
        end = time.time()
        # print(f'Time taken {end - start}')

        return result
