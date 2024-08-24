# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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

import copy
import itertools
import json
import os
from functools import partial
from typing import Any, Optional, Union

import sacrebleu
import torch
import numpy as np
from omegaconf import ListConfig
from omegaconf.dictconfig import DictConfig
from omegaconf.omegaconf import OmegaConf, open_dict
from pytorch_lightning.trainer.trainer import Trainer

from nemo.collections.asr.models import ASRModel, SpeechEncDecSelfSupervisedModel
from nemo.collections.asr.parts.mixins.transcription import move_to_device
from nemo.collections.common.metrics import MetricStringToTorchMetric, TextMetricsSet
from nemo.collections.multimodal.speech_llm.data.build_dataset import (
    build_speechllm_dataloader,
    build_speechllm_dataset,
)
from nemo.collections.multimodal.speech_llm.modules.perception_modules import (
    AudioPerceptionModule,
    MultiAudioPerceptionModule,
)
from  nemo.collections.multimodal.speech_llm.modules.multi_proj_modules import SumMultiEmbedding
from nemo.collections.multimodal.speech_llm.data.lhotse_dataset import token_id_to_speech_codec_id
from nemo.collections.nlp.models.language_modeling.megatron_t5_adapter_model import MegatronT5LoraModel
from nemo.collections.nlp.models.language_modeling.megatron_t5_sft_model import MegatronT5SFTModel
from nemo.collections.nlp.models.nlp_model import NLPModel
from nemo.collections.nlp.modules.common.megatron.utils import (
    average_losses_across_data_parallel_group,
    build_position_ids,
    get_iterator_k_split,
    init_method_normal,
)
from nemo.collections.nlp.parts.nlp_overrides import NLPSaveRestoreConnector
from nemo.collections.nlp.parts.utils_funcs import get_last_rank
from nemo.core.classes.mixins import adapter_mixins
from nemo.utils import AppState, logging, model_utils

try:
    from apex.transformer.pipeline_parallel.utils import (
        _reconfigure_microbatch_calculator,
        get_current_global_batch_size,
        get_micro_batch_size,
        get_num_microbatches,
    )

    HAVE_APEX = True
except (ImportError, ModuleNotFoundError):
    HAVE_APEX = False
from nemo.collections.nlp.models.language_modeling.megatron_t5_model import MegatronT5Model

try:
    from megatron.core import parallel_state, tensor_parallel
    from megatron.core.pipeline_parallel.schedules import get_forward_backward_func

    HAVE_MEGATRON_CORE = True

except (ImportError, ModuleNotFoundError):
    HAVE_MEGATRON_CORE = False


__all__ = ["ModularizedAudioT5Model"]


default_inference_config = {'tokens_to_generate': 30}


class ModularizedAudioT5Model(MegatronT5LoraModel):
    """Modularized speech GPT model."""

    def setup_perception_modules(self, cfg):
        if 'target' in cfg.perception:
            imported_cls = model_utils.import_class_by_path(cfg.perception.target)
            self.perception = imported_cls(cfg=cfg.perception)
        else:
            self.perception = (
                AudioPerceptionModule(cfg=cfg.perception)
                if "encoders" not in cfg.perception
                else MultiAudioPerceptionModule(cfg=cfg.perception)
            )

    def __init__(self, cfg: DictConfig, trainer: Trainer):
        self.cfg = cfg
        super().__init__(cfg, trainer)
        self.val_metric, self.val_metric_name = self.setup_metric(self.cfg.data.validation_ds)
        self.val_metric = torch.nn.ModuleList(self.val_metric)
        if hasattr(self.cfg.data, "test_ds"):
            self.test_metric, self.test_metric_name = self.setup_metric(self.cfg.data.test_ds)
            self.test_metric = torch.nn.ModuleList(self.test_metric)
        # Used other keys from metadata to calulate metrics
        if hasattr(self.cfg.data, "test_ds") and hasattr(self.cfg.data.test_ds, "metric"):
            self.test_metric_label_key = self.cfg.data.test_ds.metric.get('label_key', 'labels')
        if hasattr(self.cfg.data, "validation_ds") and hasattr(self.cfg.data.validation_ds, "metric"):
            self.val_metric_label_key = self.cfg.data.validation_ds.metric.get('label_key', 'labels')
        self.setup_perception_modules(cfg)
        self.setup_optimizer_param_groups()
        # self.configure_optimizers()
        self.summarize(max_depth=3)
        # follow gpt
        self.setup_complete = False
        self.sep_id = cfg.get('sep_id', self.tokenizer.bos_id)
        self.virtual_tokens = 0
        self.model = self.frozen_model.enc_dec_model

    def load_frozen_model(self, cfg, trainer):
        self.megatron_amp_O2 = cfg.get('megatron_amp_O2', False)
        t5_cfg_base = MegatronT5Model.restore_from(cfg.get('language_model_path'), trainer=trainer, return_config=True)
        # use the incoming cfg updated by _modify_config
        t5_cfg = copy.deepcopy(cfg)
        t5_cfg.target = t5_cfg_base.target
        self.frozen_model = MegatronT5Model.restore_from(
            cfg.get('language_model_path'),
            trainer=trainer,
            override_config_path=t5_cfg,
            save_restore_connector=NLPSaveRestoreConnector(),
        )
        logging.info(f"self.frozen_model.cfg: {self.frozen_model.cfg}")

    def init_model(self, cfg: DictConfig, trainer: Trainer):
        self.cfg = cfg

        self.load_frozen_model(cfg, trainer)
        self.prompt_encoder = None
        if self.frozen_model.tokenizer is not None:
            self.tokenizer = self.frozen_model.tokenizer

        if hasattr(self.frozen_model.cfg, "encoder") and hasattr(self.frozen_model.cfg, "decoder"):
            self.hidden_size = (
                self.frozen_model.cfg.encoder.hidden_size
            )  # Encoder and decoder need to have the same hidden size and we check for this in the frozen enc-dec model.
        else:
            self.hidden_size = self.frozen_model.cfg.hidden_size

        # Handle this when moving GPT prompt learning to the base class.
        self.word_embeddings = self.frozen_model.enc_dec_model.encoder_embedding.word_embeddings

        self._reduced_loss_buffer = []
        self._inference_config = None

        self.tokenizer.legacy = cfg.get('legacy_tokenizer', False)
        self.bos_id = self.tokenizer.bos_id
        self.decoder_seq_length = cfg.get('decoder_seq_length', 40)

        # make sure the default pytorch lightning gradient clipping in the basemodel
        self.grad_clip_pl_default = False  # make distributed_fused_adam happy
        self.lowest_val_loss = None
        self.prompt_encoder = None

        self.enable_autocast = (
            True if (not self.megatron_amp_O2) and (self.autocast_dtype in [torch.float16, torch.bfloat16]) else False
        )

    def parameters(self):
        # override the same method in MegatronGPT model to include parameters ouside of LM
        all_names = []
        all_params = []
        for name, param in self.named_parameters(recurse=True):
            all_names.append(name)
            all_params.append(param)

        if isinstance(self.frozen_model, list):
            for module in self.frozen_model:
                for name, param in module.named_parameters(recurse=True):
                    all_names.append(name)
                    all_params.append(param)

        return itertools.chain(all_params)

    def setup_optimizer_param_groups(self):
        """
        ModelPT override. Optimizer will get self._optimizer_param_groups.
        Makes two optimizer param groups, one for the frozen model params
        and one for the prompt-table/prompt-encoder params. The learning
        rate for the frozen model's params will always be zero effectively
        freezing the model's params but still allowing for the needed gradients
        to be passed around in pipeline parallel models. The prompt-encoder
        and/or prompt table will use the learning rate set by the user.
        """
        self.unfreeze()
        known_groups = []
        if self.cfg.get('freeze_llm', True):
            for param in self.frozen_model.parameters():
                param.requires_grad = False
            known_groups.append('model.')
        else:
            if self.cfg.get('freeze_encoder', False):
                for param in self.frozen_model.enc_dec_model.enc_dec_model.encoder.parameters():
                    param.requires_grad = False
                known_groups.append('enc_dec_model.encoder.')
            if self.cfg.get('freeze_decoder', False):
                for param in self.frozen_model.enc_dec_model.enc_dec_model.decoder.parameters():
                    param.requires_grad = False
                known_groups.append('enc_dec_model.decoder.')
            if self.cfg.get('freeze_word_emb', False):
                names = [
                    'encoder_embedding',
                    'encoder_relative_position_embedding',
                    'decoder_relative_position_embedding',
                    'decoder_embedding',
                ]
                for pname in names:
                    for param in getattr(self.frozen_model.enc_dec_model, pname).parameters():
                        param.requires_grad = False
                known_groups.append('enc_dec_model.word_embeddings.')
                known_groups.append('enc_dec_model.relative_position_embedding.')
        if self.cfg.get('freeze_modality_adapter', False):
            self.perception.modality_adapter.freeze()
            known_groups.append('modality_adapter.')
        if self.cfg.get('freeze_audio_encoder', False):
            self.perception.encoder.freeze()
            known_groups.append('audio_encoder.')

        opt_params = []
        for _, module in self.named_modules():
            if isinstance(module, adapter_mixins.AdapterModuleMixin) and module.is_adapter_available():
                module.set_enabled_adapters(enabled=True)
                module.unfreeze_enabled_adapters()  # selectively unfreeze the adapter modules.
                opt_params += [p for p in module.parameters()]

        param_groups = []
        if "optim_param_groups" in self.cfg:
            param_groups_cfg = self.cfg.optim_param_groups
            for group, group_cfg in param_groups_cfg.items():
                module = getattr(self, group, None)
                if module is None:
                    raise ValueError(f"{group} not found in model.")
                elif hasattr(module, "parameters"):
                    known_groups.append(f"{group}.")
                    new_group = {"params": module.parameters()}
                    for k, v in group_cfg.items():
                        new_group[k] = v
                    param_groups.append(new_group)
                else:
                    raise ValueError(f"{group} does not have parameters.")

        for n, p in self.named_parameters():
            is_unknown = True
            for group in known_groups:
                if n.startswith(group):
                    is_unknown = False
            if is_unknown:
                opt_params.append(p)

        param_groups = [{"params": opt_params}] + param_groups

        self._optimizer_param_groups = param_groups
        logging.info(f"Optimizer groups set:\n{self.summarize(max_depth=2)}")

    def inject_perception_input(self, encoded, encoded_len, input_ids, input_length):
        def _concat_embs(embs1, emb1_lens, embs2, emb2_lens):
            concat_emb = []
            concat_len = []
            for emb1, emb1_len, emb2, emb2_len in zip(embs1, emb1_lens, embs2, emb2_lens):
                if self.cfg.get('ignore_dummy_audio', False) and emb1_len <= 1:  # TODO: ignore the dummy audio emb
                    new_len = emb2_len
                    new_emb = emb2[:emb2_len]
                else:
                    new_len = emb1_len + emb2_len
                    new_emb = torch.concat([emb1[:emb1_len], emb2[:emb2_len]], axis=0)
                padded_new_emb = torch.zeros(emb1.shape[0] + emb2.shape[0], emb1.shape[-1], device=emb1.device)
                padded_new_emb[:new_len, ...] = new_emb
                concat_emb.append(padded_new_emb)
                concat_len.append(new_len)
            concat_emb = torch.stack(concat_emb, dim=0)
            concat_len = torch.stack(concat_len, dim=0)
            return concat_emb, concat_len

        # [b, t, c]
        lm_embedding = self.frozen_model.enc_dec_model.encoder_embedding
        input_embeds = lm_embedding.word_embeddings(input_ids)
        if self.cfg.audio_prompt_first:
            encoder_input, encoder_length = _concat_embs(encoded, encoded_len, input_embeds, input_length)
        else:  # more streaming friendly
            encoder_input, encoder_length = _concat_embs(input_embeds, input_length, encoded, encoded_len)

        b = encoder_input.shape[0]
        max_len = encoder_input.shape[1]

        # Using causal attention mask for whole input
        # TODO(zhehuai): use prefixlm instead for the audio embeddings
        attention_mask = torch.tril(torch.ones((b, max_len, max_len), device=encoder_input.device)).view(
            b, 1, max_len, max_len
        )
        # Convert attention mask from float to bool
        attention_mask = attention_mask < 0.5
        position_ids = build_position_ids(encoder_input[:, :, 0])

        # Add position embeddings
        if hasattr(lm_embedding, "position_embeddings"):
            position_embeddings = lm_embedding.position_embeddings(position_ids)
            encoder_input = encoder_input + position_embeddings
        else:
            pass
        encoder_max_length = encoder_input.shape[1]
        if lm_embedding.transpose_batch_sequence:
            encoder_input = encoder_input.contiguous()
        if self.cfg.get("sequence_parallel", False):
            encoder_input = tensor_parallel.mappings.scatter_to_sequence_parallel_region(encoder_input)
        return encoder_input, attention_mask, encoder_length, position_ids, encoder_max_length

    def _shift_labels_by_emb_len(self, labels, label_lens, emb_lens, max_len, pad_token=0):
        shifted_labels = []
        for label, label_len, emb_len in zip(labels, label_lens, emb_lens):
            shifted_label = torch.full([max_len], pad_token, device=label.device)
            shifted_label[emb_len : emb_len + label_len] = label[:label_len]
            shifted_labels.append(shifted_label)
        shifted_labels = torch.stack(shifted_labels, dim=0)
        return shifted_labels

    def _get_text_embeddings(self, text_tokens, position_ids):
        lm_embedding = self.frozen_model.enc_dec_model.encoder_embedding
        text_embeddings = lm_embedding.word_embeddings(text_tokens)  # (batch_size, seq_len, hidden_size)
        if hasattr(lm_embedding, 'position_embeddings'):
            position_embeddings = lm_embedding.position_embeddings(position_ids)
            text_embeddings = text_embeddings + position_embeddings
        return text_embeddings

    def prepare_llm_input(self, audio_batch):

        input_signal = audio_batch['audio_signal']
        input_signal_length = audio_batch['audio_signal_length']

        input_ids, input_length, loss_mask = (
            audio_batch['contexts'],
            audio_batch['context_lengths'],
            audio_batch['loss_mask'],
        )

        # [b, t, c]
        encoded, encoded_len = self.perception(
            input_signal=input_signal,
            input_signal_length=input_signal_length,
            processed_signal=None,
            processed_signal_length=None,
        )
        encoder_input, attention_mask, encoder_length, _, encoder_max_length = self.inject_perception_input(
            encoded, encoded_len, input_ids, input_length
        )
        # generate encoder_mask from encoder_length
        enc_mask = torch.arange(encoder_input.shape[1], device=encoder_input.device)[None, :] < encoder_length[:, None]
        return encoder_input, attention_mask, enc_mask

    def forward(
        self,
        audio_batch,
        checkpoint_activations_all_layers,
    ):
        """Forward pass of the model.

        We prepend audio embeddings to the instruction and label text tokens
        as the LLM input.
        """
        if 'audio_ratio' in audio_batch:
            self.log(
                'audio_ratio', audio_batch['audio_ratio'].mean(), prog_bar=True, batch_size=1, rank_zero_only=False
            )
            self.log(
                'local_batch_size',
                audio_batch['audio_ratio'].shape[0],
                prog_bar=True,
                batch_size=1,
                rank_zero_only=False,
            )

        encoder_input, attention_mask, enc_mask = self.prepare_llm_input(audio_batch)
        # enc_input = speech and text prompt
        # dec_input and label = text output label
        b = audio_batch['answers'].shape[0]
        device = audio_batch['answers'].device
        dec_input = audio_batch['masked_answer_ids'] if 'masked_answer_ids' in audio_batch else audio_batch['answers']
        dec_input = torch.cat([torch.full([b, 1], self.bos_id, device=device), dec_input[:, :-1]], dim=-1)
        labels = audio_batch['answers']
        dec_mask = (dec_input != self.tokenizer.eos_id) * (dec_input != self.tokenizer.pad_id).long().contiguous()
        output = self.frozen_model.enc_dec_model(
            enc_input_ids=None,
            enc_attn_mask=enc_mask,
            dec_input_ids=dec_input,
            dec_attn_mask=dec_mask,
            token_type_ids=None,
            labels=labels,
            output_enc_hidden_only=False,
            enc_input=encoder_input,
        )
        loss_mask = dec_mask
        return output, loss_mask

    def get_forward_output_only_func(self):
        def fwd_output_only_func(dataloader_iter, model):
            batch = next(dataloader_iter)
            extra_arg = {}
            # take the batch produced by prepare_batch_at_step
            (
                _,
                input_embeddings,
                attention_mask,
                _,
                set_inference_key_value_memory,
                inference_max_sequence_len,
            ) = batch
            if attention_mask is not None:
                attention_mask = attention_mask.cuda()
                attention_mask = attention_mask[0:1]
            extra_arg['set_inference_key_value_memory'] = set_inference_key_value_memory[0].item()
            extra_arg['inference_max_sequence_len'] = inference_max_sequence_len[0].item()
            output_tensor = model(
                input_ids=None,
                position_ids=None,
                encoder_input=input_embeddings,
                attention_mask=attention_mask,
                **extra_arg,
            )

            if isinstance(output_tensor, tuple):
                output_tensor = output_tensor[1]  # get logits only

            def id_func(output_tensor):
                return output_tensor, {'logits': output_tensor}

            return output_tensor, id_func

        return fwd_output_only_func

    def get_forward_output_and_loss_func(self, validation_step=False):
        def fwd_output_and_loss_func(dataloader_iter, model, checkpoint_activations_all_layers=None):
            batch = next(dataloader_iter)
            batch = {key: val.cuda(non_blocking=True) for key, val in batch.items()}
            output_tensor, loss_mask = self.forward(
                batch, checkpoint_activations_all_layers=checkpoint_activations_all_layers
            )

            def loss_func(output_tensor):
                # Loss for a micro-batch (ub)
                if 'audio_ratio' in batch:
                    text_loss_weight = self.cfg.get('text_loss_weight', 1.0)
                    audio_ratio = batch['audio_ratio']
                    scaled_loss_mask = loss_mask * torch.unsqueeze(
                        (1 * audio_ratio + text_loss_weight * (1 - audio_ratio)), 1
                    ).unsqueeze(2)
                    loss_for_ub = self.loss_func(scaled_loss_mask, output_tensor)
                else:
                    loss_for_ub = self.loss_func(loss_mask, output_tensor)
                if validation_step and not self.cfg.data.get('validation_drop_last', True):
                    num_valid_tokens_in_ub = batch['loss_mask'].sum()
                    if loss_for_ub.isnan():
                        assert batch['loss_mask'].count_nonzero() == 0, 'Got NaN loss with non-empty input'
                        loss_sum_for_ub = torch.zeros_like(num_valid_tokens_in_ub)
                    else:
                        loss_sum_for_ub = num_valid_tokens_in_ub * loss_for_ub

                    loss_sum_and_ub_size_all_gpu = torch.cat(
                        [
                            loss_sum_for_ub.clone().detach().view(1),
                            torch.tensor([num_valid_tokens_in_ub]).cuda().clone().detach(),
                        ]
                    )
                    # Could potentially reduce num_valid_samples_in_microbatch and use that to aggregate instead of len(self._validation_ds)
                    torch.distributed.all_reduce(
                        loss_sum_and_ub_size_all_gpu, group=parallel_state.get_data_parallel_group()
                    )
                    return loss_for_ub, {'loss_sum_and_ub_size': loss_sum_and_ub_size_all_gpu}
                else:
                    reduced_loss = average_losses_across_data_parallel_group([loss_for_ub])
                    return loss_for_ub, {'avg': reduced_loss}

            return output_tensor, loss_func

        return fwd_output_and_loss_func

    def _build_dataset(self, data_cfg, is_train=True):
        return build_speechllm_dataset(self, data_cfg, is_train)

    def build_data_loader(self, dataset, data_cfg, consumed_samples=0, is_eval=False):
        return build_speechllm_dataloader(dataset, data_cfg, consumed_samples, is_eval=is_eval)

    @classmethod
    def _modify_config(cls, gpt_cfg, cfg, audio_cfg, add_cfg_to_tree=False):
        """
        This function modifies the original gpt pre-training config (gpt_cfg) with attributes from the finetuning config (cfg).
        The `add_cfg_to_tree` arg adds `cfg` to the top of the yaml tree which is needed for all `hparams.yaml` files when passed as an arg to `load_from_checkpoint()`.
        """
        OmegaConf.set_struct(gpt_cfg, True)
        OmegaConf.resolve(cfg)
        with open_dict(gpt_cfg):
            if 'vocab_file' in cfg.model:
                gpt_cfg.tokenizer.vocab_file = cfg.model.vocab_file
            gpt_cfg.legacy_tokenizer = cfg.model.get('legacy_tokenizer', False)
            gpt_cfg.audio_prompt_first = cfg.model.get('audio_prompt_first', True)
            gpt_cfg.ignore_dummy_audio = cfg.model.get('ignore_dummy_audio', False)
            gpt_cfg.freeze_llm = cfg.model.get('freeze_llm', True)
            gpt_cfg.freeze_word_emb = cfg.model.get('freeze_word_emb', False)
            gpt_cfg.freeze_encoder = cfg.model.get('freeze_encoder', False)
            gpt_cfg.freeze_decoder = cfg.model.get('freeze_decoder', False)
            gpt_cfg.text_loss_weight = cfg.model.get('text_loss_weight', 1.0)
            gpt_cfg.freeze_audio_encoder = cfg.model.get('freeze_audio_encoder', False)
            gpt_cfg.freeze_modality_adapter = cfg.model.get('freeze_modality_adapter', False)
            gpt_cfg.megatron_amp_O2 = cfg.model.get('megatron_amp_O2', False)
            gpt_cfg.micro_batch_size = cfg.model.data.train_ds.micro_batch_size
            gpt_cfg.global_batch_size = cfg.model.data.train_ds.global_batch_size
            gpt_cfg.sequence_parallel = cfg.model.get("sequence_parallel", False)
            gpt_cfg.tensor_model_parallel_size = cfg.model.get(
                "tensor_model_parallel_size",
                gpt_cfg.tensor_model_parallel_size if hasattr(gpt_cfg, "tensor_model_parallel_size") else 1,
            )
            gpt_cfg.activations_checkpoint_granularity = cfg.model.get("activations_checkpoint_granularity", None)
            gpt_cfg.activations_checkpoint_num_layers = cfg.model.get("activations_checkpoint_num_layers", None)
            gpt_cfg.activations_checkpoint_method = cfg.model.get("activations_checkpoint_method", None)
            gpt_cfg.data = cfg.model.data
            gpt_cfg.optim = cfg.model.optim
            gpt_cfg.precision = cfg.trainer.precision
            gpt_cfg.answer_only_loss = cfg.model.answer_only_loss
            gpt_cfg.language_model_path = cfg.model.language_model_path
            gpt_cfg.salm_model_path = cfg.model.get("salm_model_path", None)
            gpt_cfg.resume_from_checkpoint = cfg.model.resume_from_checkpoint
            gpt_cfg.save_nemo_on_validation_end = cfg.model.save_nemo_on_validation_end
            gpt_cfg.gradient_as_bucket_view = cfg.model.gradient_as_bucket_view
            # set dropout
            hidden_dropout = cfg.model.get('hidden_dropout', 0.0)
            attention_dropout = cfg.model.get('attention_dropout', 0.0)
            ffn_dropout = cfg.model.get('ffn_dropout', 0.0)
            gpt_cfg.encoder.hidden_dropout = hidden_dropout
            gpt_cfg.decoder.hidden_dropout = hidden_dropout
            gpt_cfg.encoder.attention_dropout = attention_dropout
            gpt_cfg.decoder.attention_dropout = attention_dropout
            gpt_cfg.encoder.ffn_dropout = ffn_dropout
            gpt_cfg.decoder.ffn_dropout = ffn_dropout
            if hasattr(gpt_cfg, 'embedding_dropout'):
                gpt_cfg.embedding_dropout = hidden_dropout
            # set label_smoothing
            if hasattr(gpt_cfg, 'label_smoothing'):
                gpt_cfg.label_smoothing = cfg.model.get('label_smoothing', gpt_cfg.label_smoothing)
            gpt_cfg.virtual_prompt_style = cfg.model.virtual_prompt_style
            gpt_cfg.lora_tuning = cfg.model.lora_tuning
            # for AudioGPTLoRAModel
            gpt_cfg.target = f"{cls.__module__}.{cls.__name__}"
            gpt_cfg.perception = cfg.model.perception
            gpt_cfg.pretrained_audio_model = cfg.model.get('pretrained_audio_model', None)
            gpt_cfg.perception.preprocessor = audio_cfg.preprocessor
            gpt_cfg.perception.encoder = audio_cfg.encoder
            modality_adapter_cfg = gpt_cfg.perception.modality_adapter
            modality_adapter_cfg.feat_in = audio_cfg.encoder.d_model
            gpt_cfg.perception.output_dim = gpt_cfg.encoder.hidden_size
            override_vocab_size = cfg.model.get('override_vocab_size', None)
            if override_vocab_size is not None:
                gpt_cfg.override_vocab_size = override_vocab_size
            if not hasattr(gpt_cfg, 'tokenizer'):
                gpt_cfg.tokenizer = gpt_cfg.decoder_tokenizer
            # This is needed when modifying a hparam file directly to load `.ckpt` files.
            # This is not needed to modify the cfg in `.nemo` files.
            if add_cfg_to_tree:
                OmegaConf.resolve(gpt_cfg)
                gpt_cfg.cfg = gpt_cfg

        return gpt_cfg

    @classmethod
    def load_audio_model(cls, pretrained_audio_model):
        try:
            if pretrained_audio_model.endswith('.nemo'):
                logging.info(f'Loading pretrained audio model from local file: {pretrained_audio_model}')
                audio_model = ASRModel.restore_from(pretrained_audio_model, map_location='cpu')
            else:
                logging.info(f'Loading pretrained audio model from NGC: {pretrained_audio_model}')
                audio_model = ASRModel.from_pretrained(pretrained_audio_model, map_location='cpu')
        except:
            logging.info(f'Fail in loading it with ASRModel. Try again with SpeechEncDecSelfSupervisedModel.')
            if pretrained_audio_model.endswith('.nemo'):
                logging.info(f'Loading pretrained audio model from local file: {pretrained_audio_model}')
                audio_model = SpeechEncDecSelfSupervisedModel.restore_from(pretrained_audio_model, map_location='cpu')
            else:
                logging.info(f'Loading pretrained audio model from NGC: {pretrained_audio_model}')
                audio_model = SpeechEncDecSelfSupervisedModel.from_pretrained(
                    pretrained_audio_model, map_location='cpu'
                )
        return audio_model

    @classmethod
    def restore_from_pretrained_models(
        cls,
        cfg: Optional[Union[OmegaConf, str]] = None,
        trainer: Optional[Trainer] = None,
    ):
        if not cfg.model.pretrained_audio_model:
            raise RuntimeError("PEFT training needs a pretrained audio model present.")

        if not cfg.model.language_model_path:
            raise RuntimeError("PEFT training needs a trained base model present.")

        base_model_save_restore_connector = NLPSaveRestoreConnector()
        if os.path.isdir(cfg.model.language_model_path):
            base_model_save_restore_connector.model_extracted_dir = cfg.model.language_model_path
        base_model_cfg = cls.restore_from(
            restore_path=cfg.model.language_model_path,
            trainer=trainer,
            return_config=True,
            save_restore_connector=base_model_save_restore_connector,
        )
        audio_model = cls.load_audio_model(cfg.model.pretrained_audio_model)

        model_cfg = cls._modify_config(base_model_cfg, cfg, audio_model.cfg, add_cfg_to_tree=False)

        # load llm
        model = cls.restore_from(
            restore_path=cfg.model.language_model_path,
            trainer=trainer,
            override_config_path=model_cfg,
            strict=False,
        )
        # load am
        model.perception.tokenizer = audio_model.tokenizer
        if cfg.model.get('load_audio_encoder', True) and cfg.model.get('salm_model_path') is None:
            model.perception.encoder.load_state_dict(
                audio_model.encoder.state_dict(), strict='adapter' not in cfg.model.perception
            )
            logging.info(f'Loaded pretrained audio model from {cfg.model.pretrained_audio_model}')
        else:
            logging.info(f'Not load pretrained audio model from {cfg.model.pretrained_audio_model}')
        if cfg.model.get('use_am_tokenizer', False):
            model.tokenizer = audio_model.tokenizer
            logging.info(f'Use AM tokenizer: {audio_model.tokenizer}')
        if 'inference' in cfg:
            inference_cfg = OmegaConf.to_container(cfg.inference, resolve=True)
            model.set_inference_config(inference_cfg)
        return model

    def _build_vocab(self):
        """
        Manipulate vocabulary (e.g., pad vocabulary for increased performance)/
        """
        if self._cfg.get('override_vocab_size', None) is not None:
            self.padded_vocab_size = self._cfg.override_vocab_size
        else:
            self.padded_vocab_size = self._vocab_size_with_padding(
                orig_vocab_size=self.tokenizer.vocab_size,
                make_vocab_size_divisible_by=self._cfg.get('make_vocab_size_divisible_by', 128),
                tensor_model_parallel_size=self._cfg.get('tensor_model_parallel_size', 1),
            )

    def state_dict(self, destination=None, prefix=None, keep_vars=False):
        if self.setup_complete:
            # save adapter
            return_state_dict = super().state_dict(destination, prefix, keep_vars)
            # save perception
            if not self.cfg.get('freeze_audio_encoder', False):
                perception_state_dict = self.perception.state_dict(prefix="perception.")
                return_state_dict.update(perception_state_dict)
            # store llm if not freezing it
            if not self.cfg.get('freeze_llm', True):
                llm_state_dict = self.frozen_model.state_dict(prefix="frozen_model.")
                return_state_dict.update(llm_state_dict)
        else:
            return_state_dict = self.frozen_model.state_dict(prefix="frozen_model.")
        return return_state_dict

    def load_state_dict(self, state_dict, strict: bool = True):
        """
        Loads a state_dict expecting the state_dict to contain key,values
        only for the adapter parameters.
        """
        if self.setup_complete:
            # load adapters
            super().load_state_dict(state_dict, strict)
            # load perception
            print(f"loading state_dict {self.setup_complete}: {state_dict.keys()}")
            super(NLPModel, self).load_state_dict(state_dict, strict=False)
        else:
            if len([i for i in state_dict.keys() if 'lora' in i]) > 0:
                # load adapters
                super().load_state_dict(state_dict, strict)
            # load frozen llm and maybe perception model
            print(f"loading state_dict {self.setup_complete}: {state_dict.keys()}")
            super(NLPModel, self).load_state_dict(state_dict, strict=False)

    def build_train_valid_test_datasets(self, stage):
        if stage != 'test':
            logging.info('Building GPT SFT validation datasets.')
            # Wrap this in a list since the general finetuning parent class supports multi-validation.
            self._validation_ds = self._build_dataset(self.cfg.data.validation_ds, is_train=False)

        if stage != 'validate':
            if hasattr(self.cfg.data, 'test_ds'):
                logging.info('Building GPT SFT test datasets.')
                # Wrap this in a list since the general finetuning parent class supports multi-validation.
                self._test_ds = self._build_dataset(self.cfg.data.test_ds, is_train=False)

        if stage == 'validate' or stage == 'test':
            return
        logging.info('Building GPT SFT traing datasets.')
        self._train_ds = self._build_dataset(self.cfg.data.train_ds)

    def setup_training_data(self, training_data_config=None):
        return

    def setup_validation_data(self, validation_data_config=None):
        return

    def setup_test_data(self, test_data_config=None):
        return

    def setup_training_dataloader(self):
        if hasattr(self, '_train_ds'):
            consumed_samples = self.compute_consumed_samples(0)
            self._train_dl = self.build_data_loader(
                dataset=self._train_ds,
                data_cfg=self.cfg.data.train_ds,
                consumed_samples=consumed_samples,
            )

    def setup(self, stage=None):
        self.init_consumed_samples = 0

        if stage == 'predict':
            return

        # If the user wants to manually override train and validation dataloaders before calling `.fit()`
        if self._train_dl is not None and self._validation_dl is not None:
            return
        self.build_train_valid_test_datasets(stage=stage)
        if hasattr(self, '_train_ds'):
            self.setup_training_dataloader()
        if hasattr(self, '_validation_ds'):
            self._validation_dl = self.setup_eval_dataloader(self._validation_ds, self.cfg.data.validation_ds)
        if hasattr(self.cfg.data, 'test_ds'):
            self._test_dl = self.setup_eval_dataloader(self._test_ds, self.cfg.data.test_ds)

        # when using pipeline model parallel the final stage need to initialize word embeddings
        if parallel_state.get_pipeline_model_parallel_world_size() > 1:
            if isinstance(self.frozen_model, list):
                for i, module in enumerate(self.frozen_model):
                    parallel_state.set_virtual_pipeline_model_parallel_rank(i)
                    module.sync_initial_word_embeddings()
                parallel_state.set_virtual_pipeline_model_parallel_rank(0)
            else:
                self.frozen_model.sync_initial_word_embeddings()

        if self.cfg.get('transformer_engine', False):
            self.setup_transformer_engine_tp_groups()
        self.setup_complete = True

    @property
    def _metrics_require_string2category_map(self):
        return set(["f1", "accuracy", "average_precision"])

    def setup_metric(self, data_cfg):
        metric_name = "exact_string_match"
        if not hasattr(data_cfg, "metric"):
            metric = MetricStringToTorchMetric["exact_string_match"]
        else:
            if not hasattr(data_cfg.metric, "name"):
                raise ValueError("Metric name is not provided in the metric config.")
            if data_cfg.metric.name == "loss":
                return None, "loss"
            if data_cfg.metric.name not in MetricStringToTorchMetric:
                raise KeyError(
                    f"{data_cfg.metric.name} is not supported. List of supported metrics: {MetricStringToTorchMetric.keys()}"
                )
            if data_cfg.metric.name in self._metrics_require_string2category_map:
                if data_cfg.metric.average is None:
                    raise ValueError(
                        f"{data_cfg.metric.name} requires specifying whether you want to compute a micro or macro average. Found None."
                    )
            if (
                data_cfg.metric.get('labels_are_strings', False)
                and data_cfg.metric.name in self._metrics_require_string2category_map
            ):
                if data_cfg.metric.num_classes is None:
                    raise ValueError(
                        "Number of classes is not provided in the metric section within the data config. "
                        f"Please provide the number of classes in the data config to use the {data_cfg.metric.name} metric."
                    )
                if data_cfg.metric.get('class_labels', None) is None or not isinstance(
                    data_cfg.metric.get('class_labels', None), ListConfig
                ):
                    raise ValueError(
                        "Class labels are not provided properly in the metric section witnin the data config. "
                        f"Please provide the class labels as a list of strings in the data config to use the {data_cfg.metric.name} metric."
                    )
                if len(data_cfg.metric.get('class_labels', None)) != data_cfg.metric.num_classes:
                    raise ValueError(
                        f"Number of class labels {len(data_cfg.metric.get('class_labels', None))} does not match `num_classes` : {data_cfg.metric.num_classes}"
                    )

            metric_name = data_cfg.metric.name
            metric_cls = MetricStringToTorchMetric[metric_name]
            if metric_name not in TextMetricsSet:
                metric = [metric_cls(**data_cfg.metric)]
            else:
                metric = [metric_cls()]
        return metric, metric_name

    # Override the parent batch reconfiguring logic.
    def _reconfigure_and_process_inference_batch(self, batch, data_cfg):
        global_batch_size_per_gpu = batch['tokens'].size(0)
        # This should happen only on the last batch of the dataset.
        if (
            global_batch_size_per_gpu
            != get_current_global_batch_size() // parallel_state.get_data_parallel_world_size()
        ):
            # NOTE: This is reconfiguring to make sure there is no grad-acc for validation batches.
            if (
                global_batch_size_per_gpu
                != data_cfg.global_batch_size // parallel_state.get_data_parallel_world_size()
            ):
                app_state = AppState()
                _reconfigure_microbatch_calculator(
                    rank=app_state.global_rank,
                    rampup_batch_size=None,
                    global_batch_size=global_batch_size_per_gpu * parallel_state.get_data_parallel_world_size(),
                    micro_batch_size=global_batch_size_per_gpu,
                    data_parallel_size=parallel_state.get_data_parallel_world_size(),
                )
            # NOTE: need to explicitly handle resetting for multi-validation
            else:
                app_state = AppState()
                _reconfigure_microbatch_calculator(
                    rank=app_state.global_rank,
                    rampup_batch_size=None,
                    global_batch_size=data_cfg.global_batch_size,
                    micro_batch_size=data_cfg.micro_batch_size,
                    data_parallel_size=parallel_state.get_data_parallel_world_size(),
                )

    def validation_step(self, dataloader_iter, inference=False):
        return self.inference_step(dataloader_iter, 'validation')

    def _validation_step_internal(
        self, dataloader_iter, batch_idx, dataloader_idx=0, inference=False, result_mode='validation'
    ):
        """
        Our dataloaders produce a micro-batch and then we fetch
        a number of microbatches depending on the global batch size and model parallel size
        from the dataloader to produce a list of microbatches.
        The list of microbatches is then piped through the pipeline using megatron-core fwd/bwd functions.
        """
        mode = self.training
        self.eval()
        loss = self.fwd_bwd_step(dataloader_iter, 0, True)
        self.train(mode=mode)
        self.frozen_model.eval()

        if result_mode == 'validation':
            if type(self._validation_dl) == list and len(self._validation_dl) > 1:
                self.validation_step_outputs[dataloader_idx].append(loss)
            else:
                self.validation_step_outputs.append(loss)
        else:
            if type(self.trainer.test_dataloaders) == list and len(self.trainer.test_dataloaders) > 1:
                self.test_step_outputs[dataloader_idx].append(loss)
            else:
                self.test_step_outputs.append(loss)
        return loss

    def inference_step(self, dataloader_iter, mode, dataloader_idx=0):
        batch, batch_idx, dataloader_idx = next(dataloader_iter)
        data_cfg = self.cfg.data.validation_ds if mode == 'validation' else self.cfg.data.test_ds
        self._reconfigure_and_process_inference_batch(batch, data_cfg)
        # Meta data from dataset
        metadata = batch.get('metadata', [{}] * len(batch['tokens']))
        loss = self._validation_step_internal(itertools.chain([batch]), batch_idx, dataloader_idx, result_mode=mode)

        # We need _inference_config to get generation params
        # add_BOS and tokens_to_generate are set in dataset
        if self.get_inference_config() is None:
            logging.warning(f'inference_config is not set. Use default: {default_inference_config}')
            self.set_inference_config(inference_config=default_inference_config)
        self._inference_config['add_BOS'] = data_cfg.add_bos
        self._inference_config['tokens_to_generate'] = data_cfg.get('tokens_to_generate')

        output = self.predict_step(batch, batch_idx, dataloader_idx)

        inputs_text = [self.tokenizer.ids_to_text(c.tolist()) for c in batch['contexts']]
        labels_text = [self.tokenizer.ids_to_text(a.tolist()) for a in batch['answers']]
        preds_text = output['preds_text']
        if data_cfg.get("log_every_n_steps", None) is not None:
            if batch_idx % data_cfg.log_every_n_steps == 0:
                logging.info(f"Input: `{inputs_text[0]}`")
                logging.info(f"Label: `{labels_text[0]}`")
                logging.info(f"Pred: `{preds_text[0]}`")

        outputs = {
            'loss': loss,
            'preds': preds_text,  # [str]
            'labels': labels_text,  # [str]
            'inputs': inputs_text,  # [str]
            'metadata': metadata,  # [dict]
        }

        if mode == 'validation':
            if type(self.trainer.val_dataloaders) == list and len(self.trainer.val_dataloaders) > 1:
                # super().validation_step appends just loss to self.validation_step_outputs, replace the last appended loss with the outputs dict
                self.validation_step_outputs[dataloader_idx][-1] = outputs
            else:
                # super().validation_step appends just loss to self.validation_step_outputs, replace the last appended loss with the outputs dict
                self.validation_step_outputs[-1] = outputs
        else:
            if type(self.trainer.test_dataloaders) == list and len(self.trainer.test_dataloaders) > 1:
                self.test_step_outputs[dataloader_idx][-1] = outputs
            else:
                self.test_step_outputs[-1] = outputs
        return outputs

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:

        batch = move_to_device(batch, device=self.device)
        encoder_input, attention_mask, enc_mask = self.prepare_llm_input(batch)
        # enc_input = speech and text prompt
        # dec_input and label = text output label
        predicted_token_ids, log_probs = self.frozen_model.decode(
            tokens_enc=None,
            enc_mask=enc_mask,
            num_tokens_to_generate=self._inference_config['tokens_to_generate'],
            encoder_input=encoder_input,
            tokenizer=self.tokenizer,
            bos_id=self.bos_id,
        )

        # Special ids to text function to handle stripping <eos> and special tokens with sentencepiece tokenizers.
        input_text = batch['contexts']
        preds_text = MegatronT5SFTModel.ids_to_text(predicted_token_ids, self.tokenizer)
        input_text = MegatronT5SFTModel.ids_to_text(input_text, self.tokenizer)
        labels = batch['answers']

        if labels is not None:
            labels_text = MegatronT5SFTModel.ids_to_text(labels, self.tokenizer)
        else:
            labels_text = [None] * len(preds_text)

        return {
            'input_text': input_text,
            'preds_text': preds_text,
            'labels_text': labels_text,
        }

    def on_test_epoch_end(self):
        _ = self.inference_epoch_end(self.test_step_outputs, 'test', self.cfg.data.test_ds)
        # Commenting as on_test_epoch_end was a no-op in PTL 1.9
        # return super().on_test_epoch_end()

    def on_validation_epoch_end(self):
        _ = self.inference_epoch_end(self.validation_step_outputs, 'validation', self.cfg.data.validation_ds)
        # Commenting as on_validation_epoch_end was a no-op in PTL 1.9
        # return super().on_validation_epoch_end()

    def inference_epoch_end(self, outputs, mode, data_cfg):
        # Parent class will handle logging of the loss.
        if not outputs:
            # Handle case where no metrics. This can break checkpoint save/load.
            app_state = AppState()
            monitor_mode = app_state.checkpoint_callback_params.mode
            assert monitor_mode in ['min', 'max']
            averaged_metric = 0.0 if monitor_mode == 'max' else 1e2
            logging.warning(f"No outputs to log for {mode} epoch")
            return torch.Tensor([1e2]), torch.Tensor([averaged_metric])

        if isinstance(outputs[0], dict):
            outputs = [outputs]

        averaged_loss = []
        averaged_metric = []
        # Log metrics for each provided validation/test dataset.
        for dataloader_idx, output in enumerate(outputs):
            if len(output) == 0:
                logging.warning(f"Empty output for dataloader_idx: {dataloader_idx}")
                continue
            # Expand on_validation_epoch_end from parent class MegatronGPTModel as on_validation_epoch_end doesnt take outputs arg
            loss_vals = [x['loss'] for x in output]
            if parallel_state.is_pipeline_last_stage():
                # only the last pipeline parallel stages return loss with their batch size
                if self.cfg.data.get('validation_drop_last', True):
                    loss = torch.stack(loss_vals).mean()
                else:
                    # Compute the avg loss by total_loss across all samples / total number of samples
                    total_loss_and_total_samples = torch.vstack(loss_vals).sum(axis=0)
                    avg_loss = total_loss_and_total_samples[0] / total_loss_and_total_samples[1]
                    loss = avg_loss.type(torch.float32).cuda()
            else:
                loss = torch.tensor(0.0, dtype=torch.float32).cuda()

            # we can only log on one rank if it is rank zero so we broadcast from last rank
            torch.distributed.broadcast(loss, get_last_rank())

            self.log('val_loss', loss, prog_bar=True, rank_zero_only=True, batch_size=1, sync_dist=True)

            # Determine the key used to log the loss based on the user provided name of the dataset or the dataloader index.
            loss_log_key = self._determine_log_key(data_cfg, dataloader_idx, "loss", mode)
            self.log(loss_log_key, loss, batch_size=1)
            averaged_loss.append(loss)

            # Gather the outputs object from all data parallel ranks since we are using the DistributedSampler which splits data across DDP ranks.
            gathered_outputs = [None for _ in range(parallel_state.get_data_parallel_world_size())]
            torch.distributed.all_gather_object(
                gathered_outputs,
                [
                    {'preds': x['preds'], 'labels': x['labels'], 'inputs': x['inputs'], 'metadata': x['metadata']}
                    for x in output
                ],
                group=parallel_state.get_data_parallel_group(),
            )

            # Remove duplicate examples due to distributed sampler.
            inp_label_set = set()
            deduplicated_outputs = {
                'preds': [],
                'labels': [],
                'inputs': [],
                'metadata': [],
            }
            total_size = 0
            for rank in range(0, parallel_state.get_data_parallel_world_size()):
                for batch in gathered_outputs[rank]:
                    for pred, label, input, metadata in zip(
                        batch['preds'], batch['labels'], batch['inputs'], batch['metadata']
                    ):
                        key = input + label
                        total_size += 1
                        dedup = data_cfg.get('deduplicate', True)
                        if (not dedup) or key not in inp_label_set:
                            inp_label_set.add(key)
                            deduplicated_outputs['preds'].append(pred)
                            deduplicated_outputs['labels'].append(label)
                            deduplicated_outputs['inputs'].append(input)
                            deduplicated_outputs['metadata'].append(metadata)

            # Compute metric score
            metric_name = self.val_metric_name if mode == 'validation' else self.test_metric_name
            metric_label_key = self.val_metric_label_key if mode == 'validation' else self.test_metric_label_key
            if metric_name != 'loss':
                metric_log_key = self._determine_log_key(data_cfg, dataloader_idx, metric_name, mode)
                metric_fn = self.val_metric[0] if mode == 'validation' else self.test_metric[0]
                if metric_label_key in deduplicated_outputs['metadata'][0]:
                    labels = [m[metric_label_key] for m in deduplicated_outputs['metadata']]
                else:
                    labels = deduplicated_outputs['labels']

                # sacrebleu.corpus_bleu is commonly used which does not share
                # the same interface as other metrics. We handle it separately.
                if metric_name == 'bleu':
                    metric_result = torch.Tensor(
                        [sacrebleu.corpus_bleu(deduplicated_outputs['preds'], [labels]).score]
                    ).to(self.device)
                else:
                    for pred, label in zip(deduplicated_outputs['preds'], labels):
                        _ = metric_fn(pred, label)

                    metric_result = metric_fn.compute()

                if metric_name == 'rouge':
                    for k, v in metric_result.items():
                        if 'fmeasure' in k:
                            self.log(metric_log_key + f'_{k}', v.item(), sync_dist=True)
                            logging.info(f"{mode} {metric_name} {k}: {v.item()}")
                    metric_result = metric_result['rouge1_fmeasure']
                else:
                    self.log(metric_log_key, metric_result.item(), sync_dist=True)
                    logging.info(f"{mode} {metric_name}: {metric_result.item()}")

                metric_fn.reset()
                averaged_metric.append(metric_result)

            # Write predictions to file
            if self.global_rank == 0 and data_cfg.get("write_predictions_to_file", False):
                logging.info(
                    f"Total deduplicated inference data size: {total_size} to {len(deduplicated_outputs['inputs'])}"
                )

                # Check if the user provided a prefix path to the file(s) they want to write.
                if not hasattr(data_cfg, "output_file_path_prefix") or data_cfg.output_file_path_prefix is None:
                    raise ValueError(
                        f"Cannot write predictions to file when output_file_path_prefix is not set or present in the yaml config file."
                    )
                filename_log_key = self._determine_log_key(data_cfg, dataloader_idx, None, mode)
                output_dir = data_cfg.get("output_dir", "./")
                self.write_predictions_to_file(
                    deduplicated_outputs, f"{data_cfg.output_file_path_prefix}_{filename_log_key}", output_dir
                )

            torch.distributed.barrier(group=parallel_state.get_data_parallel_group())
            outputs[dataloader_idx].clear()  # free memory

        # Logging of the averaged metrics:
        averaged_loss = sum(averaged_loss) / len(averaged_loss)
        averaged_metric = sum(averaged_metric) / len(averaged_metric) if len(averaged_metric) > 0 else None

        # Handle case where metrics can be nan or inf. This can break checkpoint save/load.
        if averaged_metric is not None and (torch.isinf(averaged_metric) or torch.isnan(averaged_metric)):
            app_state = AppState()
            monitor_mode = app_state.checkpoint_callback_params.mode
            assert monitor_mode in ['min', 'max']
            averaged_metric = 0.0 if monitor_mode == 'max' else 1e5

        if mode == 'validation':
            self.log("validation_loss", averaged_loss, batch_size=1, sync_dist=True)
            if averaged_metric is not None:
                self.log(f"validation_{self.val_metric_name}", averaged_metric, sync_dist=True)
        elif mode == 'test':
            self.log("test_loss", averaged_loss, batch_size=1, sync_dist=True)
            if averaged_metric is not None:
                self.log(f"test_{self.test_metric_name}", averaged_metric, sync_dist=True)

        # Merge the functionality of previous on_inference_epoch_end() within inference_epoch_end() func here
        app_state = AppState()
        # TODO(zhehuai): add _restore_sequence_parallelism_args after sync to HEAD
        if hasattr(self, "_train_ds"):
            _reconfigure_microbatch_calculator(
                rank=app_state.global_rank,
                rampup_batch_size=None,
                global_batch_size=self.cfg.data.train_ds.global_batch_size,
                micro_batch_size=self.cfg.data.train_ds.micro_batch_size,
                data_parallel_size=parallel_state.get_data_parallel_world_size(),
            )
        # When running `trainer.validate()`, the training dataset is not available.
        else:
            logging.warning('No training data found, reconfiguring microbatches based on validation batch sizes.')
            _reconfigure_microbatch_calculator(
                rank=app_state.global_rank,
                rampup_batch_size=None,
                global_batch_size=data_cfg.global_batch_size,
                micro_batch_size=data_cfg.micro_batch_size,
                data_parallel_size=parallel_state.get_data_parallel_world_size(),
            )

        return averaged_loss, averaged_metric

    # consistent with speech models
    def write_predictions_to_file(self, outputs, output_file_path_prefix, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        output_file_path = output_file_path_prefix + "_inputs_preds_labels.jsonl"
        output_file_path = os.path.join(output_dir, output_file_path)
        with open(output_file_path, "w") as f_json:
            assert (
                len(outputs['inputs']) == len(outputs['preds']) == len(outputs['labels']) == len(outputs['metadata'])
            )
            for i, p, l, m in zip(outputs['inputs'], outputs['preds'], outputs['labels'], outputs['metadata']):
                json_string = {'input': i, 'pred_text': p, 'text': l}
                for k, v in m.items():
                    if k not in json_string:
                        json_string[k] = v
                f_json.write(json.dumps(json_string) + '\n')

        logging.info(f'Predictions saved to {output_file_path}')

    def setup_eval_dataloader(self, datasets, data_cfg):
        dataloaders = []
        if not isinstance(datasets, list):
            return self.build_data_loader(dataset=datasets, data_cfg=data_cfg, consumed_samples=0, is_eval=True)
        for dataset in datasets:
            eval_dl = self.build_data_loader(dataset=dataset, data_cfg=data_cfg, consumed_samples=0, is_eval=True)
            dataloaders.append(eval_dl)
        return dataloaders

    def fwd_bwd_step(self, dataloader_iter, batch_idx, forward_only):
        batch = next(dataloader_iter)
        # Pass only torch.Tensor to prevent errors when process get_iterator_k_split()
        batch = {k: v for k, v in batch.items() if isinstance(v, torch.Tensor)}
        seq_length = batch['tokens'].shape[1]
        # handle the case where the batch size from dynamic bucketting is not divisible in lhotse
        data_iter = get_iterator_k_split(batch, get_num_microbatches(), enforce_divisible_batch=False)

        # handle asynchronous grad reduction
        no_sync_func = None
        grad_sync_func = None
        param_sync_func = None
        if not forward_only and self.with_distributed_adam:
            no_sync_func = partial(
                self._optimizer.no_sync,
                greedy_grad_copy=self.megatron_amp_O2,
            )
            grad_sync_func = self.reduce_overlap_gradients
            param_sync_func = self.sync_overlap_parameters

        self.model.config.no_sync_func = no_sync_func
        self.model.config.grad_sync_func = grad_sync_func
        self.model.config.param_sync_func = param_sync_func

        fwd_bwd_function = get_forward_backward_func()

        dec_seq_length = batch['answers'].shape[1]

        losses_reduced_per_micro_batch = fwd_bwd_function(
            forward_step_func=self.get_forward_output_and_loss_func(),
            data_iterator=data_iter,
            model=[self.model],
            num_microbatches=get_num_microbatches(),
            forward_only=forward_only,
            seq_length=seq_length,
            micro_batch_size=get_micro_batch_size(),
            decoder_seq_length=dec_seq_length,
        )

        # only the last stages of the pipeline return losses
        if losses_reduced_per_micro_batch:
            if (not forward_only) or self.cfg.data.get('validation_drop_last', True):
                # average loss across micro batches
                loss_tensors_list = [loss_reduced['avg'] for loss_reduced in losses_reduced_per_micro_batch]
                loss_tensor = torch.concat(loss_tensors_list)
                loss_mean = loss_tensor.mean()
            else:
                # Get the total loss since micro batches sizes are not uniform
                loss_sum_tensors_list = [
                    loss_sum['loss_sum_and_ub_size']
                    for loss_sum in losses_reduced_per_micro_batch
                    if loss_sum['loss_sum_and_ub_size'][1] > 0
                ]
                loss_sum = (
                    torch.vstack(loss_sum_tensors_list).sum(axis=0)
                    if len(loss_sum_tensors_list) > 0
                    else torch.tensor([0.0, 0.0]).cuda()
                )
                return loss_sum
        else:
            # we're not on the last pipeline stage so no losses
            if forward_only:
                loss_mean = []
            else:
                loss_mean = torch.tensor(0.0).cuda()

        return loss_mean

    def loss_func(self, loss_mask, output_tensor):
        losses = output_tensor.float()
        loss_mask = loss_mask.view(-1).float()
        loss = torch.sum(losses.view(-1) * loss_mask) / loss_mask.sum()  # sequence level nll
        return loss

    def _determine_log_key(self, data_config, dataloader_idx, metric_name, mode):
        # Function that determines whether to log based on the user provided name of the dataset or the dataloader index.
        base_key = f"{mode}_{metric_name}_" if metric_name is not None else f"{mode}_"
        # If the user provided names for each validation/test dataset, use those.
        if hasattr(data_config, "names") and data_config.names is not None:
            # With only a single validation/test dataset, the name is not a list.
            if not isinstance(data_config.names, ListConfig):
                name = data_config.names
            else:
                name = data_config.names[dataloader_idx]
            return base_key + name
        else:
            return base_key + f"dataloader{dataloader_idx}"

    def test_step(self, dataloader_iter, dataloader_idx=0):
        return self.inference_step(dataloader_iter, 'test')

    def training_step(self, dataloader_iter):
        batch, batch_idx, dataloader_idx = next(dataloader_iter)
        return super().training_step(itertools.chain([batch]), batch_idx=batch_idx)

    def setup_mcore_distributed_parallel(self):
        """Set up mcore distributed data parallel called by configure_ddp in nlp_overrides."""
        if self.with_distributed_adam and self.use_mcore_dist_optim:
            raise ValueError("T5 does not support both distributed adam and mcore distributed data parallel.")


class DecoderTextPromptModularizedAudioT5Model(ModularizedAudioT5Model):
    """Modularized speech GPT model."""

    def prepare_llm_input(self, audio_batch):

        input_signal = audio_batch['audio_signal']
        input_signal_length = audio_batch['audio_signal_length']

        # [b, t, c]
        encoded, encoded_len = self.perception(
            input_signal=input_signal,
            input_signal_length=input_signal_length,
            processed_signal=None,
            processed_signal_length=None,
        )
        encoder_input, attention_mask, encoder_length = encoded, None, encoded_len
        # generate encoder_mask from encoder_length
        enc_mask = torch.arange(encoder_input.shape[1], device=encoder_input.device)[None, :] < encoder_length[:, None]
        return encoder_input, attention_mask, enc_mask

    def forward(
        self,
        audio_batch,
        checkpoint_activations_all_layers,
    ):
        """Forward pass of the model.

        We prepend audio embeddings to the instruction and label text tokens
        as the LLM input.
        """
        if 'audio_ratio' in audio_batch:
            self.log(
                'local_batch_size',
                audio_batch['audio_ratio'].shape[0],
                prog_bar=True,
                batch_size=1,
                rank_zero_only=False,
            )

        encoder_input, _, enc_mask = self.prepare_llm_input(audio_batch)
        # enc_input = speech prompt
        # dec_input and label = text prompt and text output label
        dec_input = audio_batch['tokens']
        labels = audio_batch['labels']
        dec_mask = (dec_input != self.tokenizer.eos_id) * (dec_input != self.tokenizer.pad_id).long().contiguous()
        output = self.frozen_model.enc_dec_model(
            enc_input_ids=None,
            enc_attn_mask=enc_mask,
            dec_input_ids=dec_input,
            dec_attn_mask=dec_mask,
            token_type_ids=None,
            labels=labels,
            output_enc_hidden_only=False,
            enc_input=encoder_input,
        )
        loss_mask = audio_batch['loss_mask']
        return output, loss_mask

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:

        batch = move_to_device(batch, device=self.device)
        encoder_input, _, enc_mask = self.prepare_llm_input(batch)
        # enc_input = speech prompt
        # dec_input and label = text prompt and text output label

        predicted_token_ids, log_probs = self.frozen_model.decode(
            tokens_enc=None,
            enc_mask=enc_mask,
            num_tokens_to_generate=self._inference_config['tokens_to_generate'],
            encoder_input=encoder_input,
            tokenizer=self.tokenizer,
            bos_id=self.bos_id,
            predicted_tokens_dec=torch.cat(
                [
                    batch['contexts'],
                    torch.full_like(batch['contexts'][:, :1], self.sep_id, device=batch['contexts'].device),
                ],
                dim=1,
            ),
        )
        predicted_token_ids = predicted_token_ids[:, batch['contexts'].shape[1] + 1 :]

        # Special ids to text function to handle stripping <eos> and special tokens with sentencepiece tokenizers.
        input_text = batch['contexts']
        preds_text = MegatronT5SFTModel.ids_to_text(predicted_token_ids, self.tokenizer)
        input_text = MegatronT5SFTModel.ids_to_text(input_text, self.tokenizer)
        labels = batch['answers']

        if labels is not None:
            labels_text = MegatronT5SFTModel.ids_to_text(labels, self.tokenizer)
        else:
            labels_text = [None] * len(preds_text)

        return {
            'input_text': input_text,
            'preds_text': preds_text,
            'labels_text': labels_text,
        }

    def _build_dataset(self, data_cfg, is_train=True):
        # this is crucial so as to tell the decoder when to start generate answer after context and paddings
        assert data_cfg.add_sep == True
        return super()._build_dataset(data_cfg, is_train)


class MultiProjModularizedAudioT5Model(ModularizedAudioT5Model):
    """Modularized speech GPT model."""

    def load_frozen_model(self, cfg, trainer):
        self.megatron_amp_O2 = cfg.get('megatron_amp_O2', False)
        t5_cfg_base = MegatronT5Model.restore_from(cfg.get('language_model_path'), trainer=trainer, return_config=True)
        # use the incoming cfg updated by _modify_config
        t5_cfg = copy.deepcopy(cfg)
        t5_cfg.target = cfg.t5_target if 't5_target' in cfg else t5_cfg_base.target
        if 'n_proj_heads' in cfg:
            t5_cfg.n_proj_heads = cfg.n_proj_heads
        if 'proj_head_dims' in cfg:
            t5_cfg.proj_head_dims = cfg.proj_head_dims
        if 'proj_head_loss_weights' in cfg:
            t5_cfg.proj_head_loss_weights = cfg.proj_head_loss_weights
        self.frozen_model = MegatronT5Model.restore_from(
            cfg.get('language_model_path'),
            trainer=trainer,
            override_config_path=t5_cfg,
            save_restore_connector=NLPSaveRestoreConnector(),
            strict=False,
        )
        logging.info(f"self.frozen_model.cfg: {self.frozen_model.cfg}")
    
    def init_model(self, cfg: DictConfig, trainer: Trainer):
        self.cfg = cfg

        self.load_frozen_model(cfg, trainer)
        self.prompt_encoder = None
        if self.frozen_model.tokenizer is not None:
            self.tokenizer = self.frozen_model.tokenizer

        if hasattr(self.frozen_model.cfg, "encoder") and hasattr(self.frozen_model.cfg, "decoder"):
            self.hidden_size = (
                self.frozen_model.cfg.encoder.hidden_size
            )  # Encoder and decoder need to have the same hidden size and we check for this in the frozen enc-dec model.
        else:
            self.hidden_size = self.frozen_model.cfg.hidden_size

        # Handle this when moving GPT prompt learning to the base class.
        self.word_embeddings = self.frozen_model.enc_dec_model.encoder_embedding.word_embeddings

        self._reduced_loss_buffer = []
        self._inference_config = None

        self.tokenizer.legacy = cfg.get('legacy_tokenizer', False)
        self.bos_id = self.tokenizer.bos_id
        self.decoder_seq_length = cfg.get('decoder_seq_length', 40)

        # make sure the default pytorch lightning gradient clipping in the basemodel
        self.grad_clip_pl_default = False  # make distributed_fused_adam happy
        self.lowest_val_loss = None
        self.prompt_encoder = None

        self.enable_autocast = (
            True if (not self.megatron_amp_O2) and (self.autocast_dtype in [torch.float16, torch.bfloat16]) else False
        )
        
    def __init__(self, cfg: DictConfig, trainer: Trainer):
        self.cfg = cfg
        super().__init__(cfg, trainer)
        if cfg.get('salm_model_path') is not None:
            torch_state_dict = torch.load(cfg.get('salm_model_path'))['state_dict']
            self.setup_complete = False
            # breakpoint()
            self.load_state_dict(torch_state_dict, strict=False)
            logging.info(f"loading from {cfg.get('salm_model_path')}: {torch_state_dict.keys()}")

        # Extend word embedding table if self.padded_vocab_size is larger than the size of the pre-trained word embedding
        pretrained_emb = self.frozen_model.enc_dec_model.decoder_embedding.word_embeddings
        self.frozen_model.enc_dec_model.decoder_embedding = SumMultiEmbedding(
            config=self.frozen_model.enc_dec_model.config,
            hidden_size=self.hidden_size,
            orig_vocab_size=self.frozen_model.proj_head_dims[0],
            vocab_size=self.padded_vocab_size,
            max_sequence_length=self.frozen_model.cfg.max_position_embeddings,
            init_method=init_method_normal(0.2),
            num_tokentypes=0,
            embedding_dropout_prob=self.frozen_model.cfg.embedding_dropout,
            position_embedding_type=self.frozen_model.cfg.encoder.get('position_embedding_type', 'learned_absolute'),
        )
        self.frozen_model.enc_dec_model.decoder_embedding.word_embeddings.weight.data[
            : pretrained_emb.weight.shape[0]
        ] = pretrained_emb.weight.data
        self.word_embeddings = self.frozen_model.enc_dec_model.encoder_embedding.word_embeddings

    @classmethod
    def _modify_config(cls, gpt_cfg, cfg, audio_cfg, add_cfg_to_tree=False):
        """
        This function modifies the original gpt pre-training config (gpt_cfg) with attributes from the finetuning config (cfg).
        The `add_cfg_to_tree` arg adds `cfg` to the top of the yaml tree which is needed for all `hparams.yaml` files when passed as an arg to `load_from_checkpoint()`.
        """
        gpt_cfg = super()._modify_config(gpt_cfg, cfg, audio_cfg, add_cfg_to_tree)
        OmegaConf.set_struct(gpt_cfg, True)
        OmegaConf.resolve(cfg)
        with open_dict(gpt_cfg):
            t5_target = cfg.model.get('t5_target', None)
            if t5_target is not None:
                gpt_cfg.t5_target = t5_target
            n_proj_heads = cfg.model.get('n_proj_heads', None)
            if n_proj_heads is not None:
                gpt_cfg.n_proj_heads = n_proj_heads
            proj_head_dims = cfg.model.get('proj_head_dims', None)
            if proj_head_dims is not None:
                gpt_cfg.proj_head_dims = proj_head_dims
            proj_head_loss_weights = cfg.model.get('proj_head_loss_weights', None)
            if proj_head_loss_weights is not None:
                gpt_cfg.proj_head_loss_weights = proj_head_loss_weights
            attention_map_mode = cfg.model.get('attention_map_mode', None)
            if attention_map_mode is not None:
                gpt_cfg.attention_map_mode = attention_map_mode
            sampling = cfg.model.get('sampling', None)
            if sampling is not None:
                gpt_cfg.sampling = sampling
            # This is needed when modifying a hparam file directly to load `.ckpt` files.
            # This is not needed to modify the cfg in `.nemo` files.
            if add_cfg_to_tree:
                OmegaConf.resolve(gpt_cfg)
                gpt_cfg.cfg = gpt_cfg

        return gpt_cfg

    def prepare_llm_input(self, audio_batch):
        input_signal = audio_batch['audio_signal']
        input_signal_length = audio_batch['audio_signal_length']

        # [b, t, c]
        encoded, encoded_len = self.perception(
            input_signal=input_signal,
            input_signal_length=input_signal_length,
            processed_signal=None,
            processed_signal_length=None,
        )
        
        encoder_input, attention_mask, encoder_length, _, encoder_max_length = self.inject_perception_input(
            encoded, encoded_len, audio_batch['instructions'], audio_batch['instruction_lengths']
        )
        # generate encoder_mask from encoder_length
        enc_mask = torch.arange(encoder_input.shape[1], device=encoder_input.device)[None, :] < encoder_length[:, None]
        return encoder_input, attention_mask, enc_mask

    def forward(
        self,
        audio_batch,
        checkpoint_activations_all_layers,
        **kwargs,
    ):
        """Forward pass of the model.

        We prepend audio embeddings to the instruction and label text tokens
        as the LLM input.
        """
        if 'audio_ratio' in audio_batch:
            self.log(
                'local_batch_size',
                audio_batch['audio_ratio'].shape[0],
                prog_bar=True,
                batch_size=1,
                rank_zero_only=False,
            )

        encoder_input, _, enc_mask = self.prepare_llm_input(audio_batch)
        # enc_input = speech prompt
        # dec_input and label = text prompt and text output label
        dec_input = audio_batch['tokens']
        labels = audio_batch['labels']

        dec_mask = (dec_input[:, :, 0] != self.tokenizer.pad_id).long().contiguous()
        output = self.frozen_model.enc_dec_model(
            enc_input_ids=None,
            enc_attn_mask=enc_mask,
            dec_input_ids=dec_input,
            dec_attn_mask=dec_mask,
            token_type_ids=None,
            labels=labels,
            output_enc_hidden_only=False,
            enc_input=encoder_input,
            **kwargs,
        )
        
        loss_mask = audio_batch['loss_mask']
        return output, loss_mask
    

    def fwd_bwd_step(self, dataloader_iter, batch_idx, forward_only, return_attention_probs=False):
        batch = next(dataloader_iter)
        # Pass only torch.Tensor to prevent errors when process get_iterator_k_split()
        batch = {k: v for k, v in batch.items() if isinstance(v, torch.Tensor)}
        seq_length = batch['tokens'].shape[1]
        # handle the case where the batch size from dynamic bucketting is not divisible in lhotse
        data_iter = get_iterator_k_split(batch, get_num_microbatches(), enforce_divisible_batch=False)

        # handle asynchronous grad reduction
        no_sync_func = None
        grad_sync_func = None
        param_sync_func = None
        if not forward_only and self.with_distributed_adam:
            no_sync_func = partial(
                self._optimizer.no_sync,
                greedy_grad_copy=self.megatron_amp_O2,
            )
            grad_sync_func = self.reduce_overlap_gradients
            param_sync_func = self.sync_overlap_parameters

        self.model.config.no_sync_func = no_sync_func
        self.model.config.grad_sync_func = grad_sync_func
        self.model.config.param_sync_func = param_sync_func

        fwd_bwd_function = get_forward_backward_func()

        dec_seq_length = batch['answers'].shape[1]

        output = fwd_bwd_function(
            forward_step_func=self.get_forward_output_and_loss_func(),
            data_iterator=data_iter,
            model=[self.model],
            num_microbatches=get_num_microbatches(),
            forward_only=forward_only,
            seq_length=seq_length,
            micro_batch_size=get_micro_batch_size(),
            decoder_seq_length=dec_seq_length,
        )
        attention_probs = sum([item['attention_probs'] for item in output], [])

        # only the last stages of the pipeline return something
        if output:
            if (not forward_only) or self.cfg.data.get('validation_drop_last', True):
                # average loss across micro batches
                loss_tensors_list = [item['avg'] for item in output]
                loss_tensor = torch.concat(loss_tensors_list)
                loss_mean = loss_tensor.mean()
            else:
                # Get the total loss since micro batches sizes are not uniform
                loss_sum_tensors_list = [
                    item['loss_sum_and_ub_size']
                    for item in output
                    if item['loss_sum_and_ub_size'][1] > 0
                ]
                loss_sum = (
                    torch.vstack(loss_sum_tensors_list).sum(axis=0)
                    if len(loss_sum_tensors_list) > 0
                    else torch.tensor([0.0, 0.0]).cuda()
                )
                if return_attention_probs:
                    return loss_sum, attention_probs
                else:
                    return loss_sum
        else:
            # we're not on the last pipeline stage so no losses
            if forward_only:
                loss_mean = []
            else:
                loss_mean = torch.tensor(0.0).cuda()

        if return_attention_probs:
            return loss_mean, attention_probs
        else:
            return loss_mean

    def get_forward_output_and_loss_func(self, validation_step=False):
        def fwd_output_and_loss_func(dataloader_iter, model, checkpoint_activations_all_layers=None):
            batch = next(dataloader_iter)
            batch = {key: val.cuda(non_blocking=True) for key, val in batch.items()}
            output, loss_mask = self.forward(
                batch, 
                checkpoint_activations_all_layers=checkpoint_activations_all_layers, 
                return_all_selfattention_probs=True,
                return_all_crossattention_probs=True,
            )

            def loss_func(output):
                output_tensor, attention_probs = output
                # Loss for a micro-batch (ub)
                if 'audio_ratio' in batch:
                    text_loss_weight = self.cfg.get('text_loss_weight', 1.0)
                    audio_ratio = batch['audio_ratio']
                    scaled_loss_mask = loss_mask * torch.unsqueeze(
                        (1 * audio_ratio + text_loss_weight * (1 - audio_ratio)), 1, 
                    ).unsqueeze(2)
                    loss_for_ub = self.loss_func(scaled_loss_mask, output_tensor)
                else:
                    loss_for_ub = self.loss_func(loss_mask, output_tensor)
                if validation_step and not self.cfg.data.get('validation_drop_last', True):
                    num_valid_tokens_in_ub = batch['loss_mask'].sum()
                    if loss_for_ub.isnan():
                        assert batch['loss_mask'].count_nonzero() == 0, 'Got NaN loss with non-empty input'
                        loss_sum_for_ub = torch.zeros_like(num_valid_tokens_in_ub)
                    else:
                        loss_sum_for_ub = num_valid_tokens_in_ub * loss_for_ub

                    loss_sum_and_ub_size_all_gpu = torch.cat(
                        [
                            loss_sum_for_ub.clone().detach().view(1),
                            torch.tensor([num_valid_tokens_in_ub]).cuda().clone().detach(),
                        ]
                    )
                    # Could potentially reduce num_valid_samples_in_microbatch and use that to aggregate instead of len(self._validation_ds)
                    torch.distributed.all_reduce(
                        loss_sum_and_ub_size_all_gpu, group=parallel_state.get_data_parallel_group()
                    )
                    return loss_for_ub, {'loss_sum_and_ub_size': loss_sum_and_ub_size_all_gpu, 'attention_probs': attention_probs}
                else:
                    reduced_loss = average_losses_across_data_parallel_group([loss_for_ub])
                    return loss_for_ub, {'avg': reduced_loss, 'attention_probs': attention_probs}

            return output, loss_func

        return fwd_output_and_loss_func
    
    def _validation_step_internal(
        self, dataloader_iter, batch_idx, dataloader_idx=0, inference=False, result_mode='validation'
    ):
        """
        Our dataloaders produce a micro-batch and then we fetch
        a number of microbatches depending on the global batch size and model parallel size
        from the dataloader to produce a list of microbatches.
        The list of microbatches is then piped through the pipeline using megatron-core fwd/bwd functions.
        """
        mode = self.training
        self.eval()
        loss, attention_probs = self.fwd_bwd_step(dataloader_iter, 0, True, return_attention_probs=True)
        self.train(mode=mode)
        self.frozen_model.eval()

        if result_mode == 'validation':
            if type(self._validation_dl) == list and len(self._validation_dl) > 1:
                self.validation_step_outputs[dataloader_idx].append(loss)
            else:
                self.validation_step_outputs.append(loss)
        else:
            if type(self.trainer.test_dataloaders) == list and len(self.trainer.test_dataloaders) > 1:
                self.test_step_outputs[dataloader_idx].append(loss)
            else:
                self.test_step_outputs.append(loss)
        return loss, attention_probs

    def inference_step(self, dataloader_iter, mode, dataloader_idx=0):
        batch, batch_idx, dataloader_idx = next(dataloader_iter)
        data_cfg = self.cfg.data.validation_ds if mode == 'validation' else self.cfg.data.test_ds
        self._reconfigure_and_process_inference_batch(batch, data_cfg)
        # Meta data from dataset
        metadata = batch.get('metadata', [{}] * len(batch['tokens']))
        speaker_contexts = batch.get('speaker_contexts', [{}] * len(batch['tokens']))
        loss, attention_probs = self._validation_step_internal(itertools.chain([batch]), batch_idx, dataloader_idx, result_mode=mode)
        if self.cfg.attention_map_mode == 'validation':
            self_attention_probs, cross_attention_probs = attention_probs

        # We need _inference_config to get generation params
        # add_BOS and tokens_to_generate are set in dataset
        if self.get_inference_config() is None:
            logging.warning(f'inference_config is not set. Use default: {default_inference_config}')
            self.set_inference_config(inference_config=default_inference_config)
        self._inference_config['add_BOS'] = data_cfg.add_bos
        self._inference_config['tokens_to_generate'] = data_cfg.get('tokens_to_generate')

        output = self.predict_step(batch, batch_idx, dataloader_idx)
        # pdb.set_trace()

        inputs_text = output['input_text']
        answers = output['answers']
        preds = output['preds']
        if self.cfg.attention_map_mode == 'inference':
            self_attention_probs, cross_attention_probs = output['attention_probs']
        if data_cfg.get("log_every_n_steps", None) is not None:
            if batch_idx % data_cfg.log_every_n_steps == 0:
                logging.info(f"Input: `{inputs_text[0]}`")

        outputs = {
            'loss': loss,
            'preds': preds,  # 2d array
            'answers': answers,  # 2d array
            'inputs': inputs_text,  # [str]
            'speaker_contexts': speaker_contexts,  # 2d array
            'self_attention_probs': self_attention_probs,  # 2d array
            'cross_attention_probs': cross_attention_probs,  # 2d array
            'metadata': metadata,  # [dict]
        }

        # import pdb; pdb.set_trace()

        if mode == 'validation':
            if type(self.trainer.val_dataloaders) == list and len(self.trainer.val_dataloaders) > 1:
                # super().validation_step appends just loss to self.validation_step_outputs, replace the last appended loss with the outputs dict
                self.validation_step_outputs[dataloader_idx][-1] = outputs
            else:
                # super().validation_step appends just loss to self.validation_step_outputs, replace the last appended loss with the outputs dict
                self.validation_step_outputs[-1] = outputs
        else:
            if type(self.trainer.test_dataloaders) == list and len(self.trainer.test_dataloaders) > 1:
                self.test_step_outputs[dataloader_idx][-1] = outputs
            else:
                self.test_step_outputs[-1] = outputs
        return outputs

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        batch = move_to_device(batch, device=self.device)
        encoder_input, _, enc_mask = self.prepare_llm_input(batch)
        # enc_input = speech prompt
        # dec_input and label = text prompt and text output label

        predicted_token_ids, log_probs, attention_probs = self.frozen_model.decode(
            tokens_enc=None,
            enc_mask=enc_mask,
            num_tokens_to_generate=self._inference_config['tokens_to_generate'],
            encoder_input=encoder_input,
            tokenizer=self.tokenizer,
            bos_id=self.bos_id,
            predicted_tokens_dec=batch["tokens"][:, 0:1, :],
            sampling_method=self.cfg.sampling.get("sampling_method", "greedy-search")
        )

        # Special ids to text function to handle stripping <eos> and special tokens with sentencepiece tokenizers.
        answers = batch['target_texts']

        return {
            'input_text': batch['source_texts'],
            'preds': predicted_token_ids,
            'answers': batch['answers'],
            'attention_probs': attention_probs,
        }

    def inference_epoch_end(self, outputs, mode, data_cfg):
        # Parent class will handle logging of the loss.
        if not outputs:
            # Handle case where no metrics. This can break checkpoint save/load.
            app_state = AppState()
            monitor_mode = app_state.checkpoint_callback_params.mode
            assert monitor_mode in ['min', 'max']
            averaged_metric = 0.0 if monitor_mode == 'max' else 1e2
            logging.warning(f"No outputs to log for {mode} epoch")
            return torch.Tensor([1e2]), torch.Tensor([averaged_metric])

        if isinstance(outputs[0], dict):
            outputs = [outputs]
        
        averaged_loss = []
        # Log metrics for each provided validation/test dataset.
        for dataloader_idx, output in enumerate(outputs):
            if len(output) == 0:
                logging.warning(f"Empty output for dataloader_idx: {dataloader_idx}")
                continue
            # Expand on_validation_epoch_end from parent class MegatronGPTModel as on_validation_epoch_end doesnt take outputs arg
            loss_vals = [x['loss'] for x in output]
            if parallel_state.is_pipeline_last_stage():
                # only the last pipeline parallel stages return loss with their batch size
                if self.cfg.data.get('validation_drop_last', True):
                    loss = torch.stack(loss_vals).mean()
                else:
                    # Compute the avg loss by total_loss across all samples / total number of samples
                    total_loss_and_total_samples = torch.vstack(loss_vals).sum(axis=0)
                    avg_loss = total_loss_and_total_samples[0] / total_loss_and_total_samples[1]
                    loss = avg_loss.type(torch.float32).cuda()
            else:
                loss = torch.tensor(0.0, dtype=torch.float32).cuda()

            # we can only log on one rank if it is rank zero so we broadcast from last rank
            torch.distributed.broadcast(loss, get_last_rank())

            self.log('val_loss', loss, prog_bar=True, rank_zero_only=True, batch_size=1, sync_dist=True)

            # Determine the key used to log the loss based on the user provided name of the dataset or the dataloader index.
            loss_log_key = self._determine_log_key(data_cfg, dataloader_idx, "loss", mode)
            self.log(loss_log_key, loss, batch_size=1)
            averaged_loss.append(loss)

            # Gather the outputs object from all data parallel ranks since we are using the DistributedSampler which splits data across DDP ranks.
            gathered_outputs = [None for _ in range(parallel_state.get_data_parallel_world_size())]
            torch.distributed.all_gather_object(
                gathered_outputs,
                [
                    {'preds': x['preds'], 'answers': x['answers'], 'inputs': x['inputs'], 'speaker_contexts': x['speaker_contexts'], 'metadata': x['metadata'], 'self_attention_probs': x['self_attention_probs'], 'cross_attention_probs': x['cross_attention_probs']}
                    for x in output
                ],
                group=parallel_state.get_data_parallel_group(),
            )

            # Remove duplicate examples due to distributed sampler.
            inp_label_set = set()
            deduplicated_outputs = {
                'text_preds': [],
                'text_answers': [],
                'speech_preds': [],
                'speech_answers': [],
                'inputs': [],
                'speaker_contexts': [],
                'self_attn': [],
                'cross_attn': [],
                'metadata': [],
            }
            total_size = 0
            for rank in range(0, parallel_state.get_data_parallel_world_size()):
                for batch in gathered_outputs[rank]:
                    for pred, answer, input, speaker_context, self_attn, cross_attn, metadata in zip(
                        batch['preds'], batch['answers'], batch['inputs'], batch['speaker_contexts'], batch['self_attention_probs'], batch['cross_attention_probs'], batch['metadata']
                    ):
                        key = input
                        total_size += 1
                        dedup = data_cfg.get('deduplicate', True)
                        if (not dedup) or key not in inp_label_set:
                            inp_label_set.add(key)
                            # Remove leading BOS
                            pred = pred[1:]
                            text_pred, speech_pred = self.parse_decoder_outputs(pred, self.tokenizer.eos_id, self.cfg.data.train_ds.speech_pad_id)
                            text_answer, speech_answer = self.parse_decoder_outputs(answer, self.tokenizer.eos_id, self.cfg.data.train_ds.speech_pad_id)
                            deduplicated_outputs['text_preds'].append(MegatronT5SFTModel.ids_to_text(text_pred.unsqueeze(0), self.tokenizer))
                            deduplicated_outputs['text_answers'].append(MegatronT5SFTModel.ids_to_text(text_answer.unsqueeze(0), self.tokenizer))
                            deduplicated_outputs['speech_preds'].append(speech_pred.cpu().numpy())
                            deduplicated_outputs['speech_answers'].append(speech_answer.cpu().numpy())
                            deduplicated_outputs['inputs'].append(input)
                            # deduplicated_outputs['speaker_contexts'].append(speaker_context)
                            deduplicated_outputs['self_attn'].append(self_attn.cpu().numpy())
                            deduplicated_outputs['cross_attn'].append(cross_attn.cpu().numpy())
                            deduplicated_outputs['metadata'].append(metadata)
                            # import pdb; pdb.set_trace()
            # Write predictions to file
            if self.global_rank == 0 and data_cfg.get("write_predictions_to_file", False):
                logging.info(
                    f"Total deduplicated inference data size: {total_size} to {len(deduplicated_outputs['inputs'])}"
                )

                # Check if the user provided a prefix path to the file(s) they want to write.
                if not hasattr(data_cfg, "output_file_path_prefix") or data_cfg.output_file_path_prefix is None:
                    raise ValueError(
                        f"Cannot write predictions to file when output_file_path_prefix is not set or present in the yaml config file."
                    )
                filename_log_key = self._determine_log_key(data_cfg, dataloader_idx, None, mode)
                output_dir = data_cfg.get("output_dir", "./")
                self.write_predictions_to_file(
                    deduplicated_outputs, f"{data_cfg.output_file_path_prefix}_{filename_log_key}", output_dir
                )

            torch.distributed.barrier(group=parallel_state.get_data_parallel_group())
            outputs[dataloader_idx].clear()  # free memory

        # Logging of the averaged metrics:
        averaged_loss = sum(averaged_loss) / len(averaged_loss)

        if mode == 'validation':
            self.log("validation_loss", averaged_loss, batch_size=1, sync_dist=True)
        elif mode == 'test':
            self.log("test_loss", averaged_loss, batch_size=1, sync_dist=True)

        # Merge the functionality of previous on_inference_epoch_end() within inference_epoch_end() func here
        app_state = AppState()
        # TODO(zhehuai): add _restore_sequence_parallelism_args after sync to HEAD
        if hasattr(self, "_train_ds"):
            _reconfigure_microbatch_calculator(
                rank=app_state.global_rank,
                rampup_batch_size=None,
                global_batch_size=self.cfg.data.train_ds.global_batch_size,
                micro_batch_size=self.cfg.data.train_ds.micro_batch_size,
                data_parallel_size=parallel_state.get_data_parallel_world_size(),
            )
        # When running `trainer.validate()`, the training dataset is not available.
        else:
            logging.warning('No training data found, reconfiguring microbatches based on validation batch sizes.')
            _reconfigure_microbatch_calculator(
                rank=app_state.global_rank,
                rampup_batch_size=None,
                global_batch_size=data_cfg.global_batch_size,
                micro_batch_size=data_cfg.micro_batch_size,
                data_parallel_size=parallel_state.get_data_parallel_world_size(),
            )

        return averaged_loss, averaged_loss
    
    def parse_decoder_outputs(self, decoder_output, text_separator, speech_pad_id=1001):
        # Split text and speech part based on the position of the first separator token
        first_sep_pos = torch.argmax((decoder_output[:, 0] == text_separator).long())
        text_tokens = decoder_output[:first_sep_pos, 0]
        speech_tokens = decoder_output[first_sep_pos+1:, 1:]

        # Get speech token ids
        n_speech_codebook = self.frozen_model.n_proj_heads-1
        speech_tokens = token_id_to_speech_codec_id(
            speech_tokens.unsqueeze(0), 
            n_speech_codebooks=n_speech_codebook, 
            codebook_sizes=self.frozen_model.proj_head_dims
        ).squeeze(0)
        
        # Remove padded parts of speech tokens
        speech_pad_pos = (torch.sum(speech_tokens == speech_pad_id, axis=1) == n_speech_codebook)
        # speech_pad_mask = (torch.cumsum(speech_pad_pos, 0) == 0)
        speech_pad_mask = (speech_pad_pos == 0)
        return text_tokens, speech_tokens[speech_pad_mask]

    # consistent with speech models
    def write_predictions_to_file(self, outputs, output_file_path_prefix, output_dir):
        inputs_path = os.path.join(output_dir, output_file_path_prefix + "_inputs.jsonl")
        for folder_name in ['speech_pred', 'speech_answer', 'speaker_contexts', 'self_attn', 'cross_attn']:
            os.makedirs(os.path.join(output_dir, 'npy', folder_name), exist_ok=True)
        # speaker_contexts_path = os.path.join(output_dir, 'npy', 'speaker_contexts', output_file_path_prefix + "_speaker_context_{}.npy")
        self_attn_path = os.path.join(output_dir, 'npy', 'self_attn', output_file_path_prefix + "_self_attn_{}.npy")
        cross_attn_path = os.path.join(output_dir, 'npy', 'cross_attn', output_file_path_prefix + "_cross_attn_{}.npy")
        speech_pred_path = os.path.join(output_dir, 'npy', 'speech_pred', output_file_path_prefix + "_speech_pred_{}.npy")
        speech_answer_path = os.path.join(output_dir, 'npy', 'speech_answer', output_file_path_prefix + "_speech_answer_{}.npy")

        with open(inputs_path, "w") as f_json:
            assert (
                len(outputs['inputs']) == len(outputs['text_preds']) == len(outputs['text_answers']) == len(outputs['speech_preds']) == len(outputs['speech_answers']) == len(outputs['self_attn']) == len(outputs['cross_attn']) == len(outputs['metadata'])
            )
            for i, (input_, text_pred, text_answer, speech_pred, speech_answer, self_attn, cross_attn, md) in enumerate(zip(outputs['inputs'], outputs['text_preds'], outputs['text_answers'], outputs['speech_preds'], outputs['speech_answers'], outputs['self_attn'], outputs['cross_attn'], outputs['metadata'])):
                json_string = {"index": i, 'preds': text_pred, 'answers': text_answer}
                for k, v in md.items():
                    if k not in json_string:
                        json_string[k] = v
                f_json.write(json.dumps(json_string) + '\n')

                # np.save(speaker_contexts_path.format(i), speaker_context.cpu().numpy())
                np.save(self_attn_path.format(i), self_attn)
                np.save(cross_attn_path.format(i), cross_attn)
                np.save(speech_pred_path.format(i), speech_pred)
                np.save(speech_answer_path.format(i), speech_answer)

        logging.info(f'Predictions saved to {output_dir}')

    def _build_dataset(self, data_cfg, is_train=True):
        # this is crucial so as to tell the decoder when to start generate answer after context and paddings
        assert data_cfg.add_sep == True
        return super()._build_dataset(data_cfg, is_train)