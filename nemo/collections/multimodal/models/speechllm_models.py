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

import itertools
import os
from functools import partial
from typing import Dict, Optional, Union

import torch
from omegaconf.dictconfig import DictConfig
from omegaconf.omegaconf import OmegaConf, open_dict
from pytorch_lightning.trainer.trainer import Trainer

from nemo.collections.asr.data import audio_to_text_dataset
from nemo.collections.asr.data.audio_to_text_dali import AudioToBPEDALIDataset, DALIOutputs
from nemo.collections.asr.models import ASRModel
from nemo.collections.asr.parts.preprocessing.perturb import process_augmentations
from nemo.collections.multimodal.data.audio_text_qa_dataset import AudioQuestionAnswerDataset
from nemo.collections.multimodal.modules.speechllm_perception import AudioPerceptionModel
from nemo.collections.multimodal.parts.utils.data_utils import get_num_samples_from_files
from nemo.collections.nlp.data.language_modeling.megatron.base_dataset_utils import (
    get_datasets_weights_and_num_samples,
)
from nemo.collections.nlp.data.language_modeling.megatron.blendable_dataset import BlendableDataset
from nemo.collections.nlp.models.language_modeling.megatron_gpt_peft_models import (
    MegatronGPTLoRAModel,
    MegatronGPTPEFTModel,
)
from nemo.collections.nlp.models.language_modeling.megatron_gpt_prompt_learning_model import (
    MegatronGPTPromptLearningModel,
)
from nemo.collections.nlp.modules.common.megatron.utils import (
    average_losses_across_data_parallel_group,
    build_position_ids,
    get_iterator_k_split,
)
from nemo.collections.nlp.modules.common.text_generation_utils import (
    get_default_length_params,
    get_default_sampling_params,
    megatron_gpt_generate,
)
from nemo.collections.nlp.modules.common.transformer.text_generation import LengthParam, SamplingParam
from nemo.collections.nlp.parts.nlp_overrides import NLPSaveRestoreConnector, PEFTSaveRestoreConnector
from nemo.collections.nlp.parts.utils_funcs import get_last_rank
from nemo.core.classes.mixins import AccessMixin, adapter_mixins
from nemo.core.neural_types import AcousticEncodedRepresentation, AudioSignal, LengthsType, NeuralType, SpectrogramType
from nemo.utils import AppState, logging, model_utils

try:
    from apex.transformer.pipeline_parallel.utils import get_micro_batch_size, get_num_microbatches

    HAVE_APEX = True

except (ImportError, ModuleNotFoundError):
    HAVE_APEX = False

try:
    from megatron.core import parallel_state, tensor_parallel
    from megatron.core.enums import ModelType
    from megatron.core.pipeline_parallel.schedules import get_forward_backward_func

    HAVE_MEGATRON_CORE = True

except (ImportError, ModuleNotFoundError):
    HAVE_MEGATRON_CORE = False


__all__ = ["ModularizedAudioGPTModel"]


class ModularizedAudioGPTModel(MegatronGPTLoRAModel):
    """Modularized speech GPT model."""

    def __init__(self, cfg: DictConfig, trainer: Trainer):
        self.cfg = cfg
        super().__init__(cfg, trainer)
        self.perception = AudioPerceptionModel(cfg=cfg.perception)
        self.setup_optimizer_param_groups()
        self.configure_optimizers()
        self.summarize()

    def parameters(self):
        # override the same method in MegatronGPT model to include parameters ouside of LM
        all_names = []
        all_params = []
        for name, param in self.named_parameters(recurse=True):
            all_names.append(name)
            all_params.append(param)

        if isinstance(self.model, list):
            for module in self.model:
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
            for param in self.model.parameters():
                param.requires_grad = False
            known_groups.append('model.')
        # TODO(heh): double check this part works properly
        if self.cfg.get('freeze_matcher', False):
            self.perception.matcher.freeze()
            known_groups.append('matcher.')
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
        logging.info(f"Optimizer groups set:\n{self.summarize()}")

    def prepare_llm_input(self, audio_batch):
        def _concat_embs(embs1, emb1_lens, embs2, emb2_lens):
            concat_emb = []
            concat_len = []
            for emb1, emb1_len, emb2, emb2_len in zip(embs1, emb1_lens, embs2, emb2_lens):
                new_len = emb1_len + emb2_len
                new_emb = torch.concat([emb1[:emb1_len], emb2[:emb2_len]], axis=0)
                padded_new_emb = torch.zeros(emb1.shape[0] + emb2.shape[0], emb1.shape[-1], device=emb1.device)
                padded_new_emb[:new_len, ...] = new_emb
                concat_emb.append(padded_new_emb)
                concat_len.append(new_len)
            concat_emb = torch.stack(concat_emb, dim=0)
            concat_len = torch.stack(concat_len, dim=0)
            return concat_emb, concat_len

        def _shift_labels_by_emb_len(labels, label_lens, emb_lens, max_len, pad_token=0):
            shifted_labels = []
            for label, label_len, emb_len in zip(labels, label_lens, emb_lens):
                shifted_label = torch.full([max_len], pad_token, device=label.device)
                shifted_label[emb_len : emb_len + label_len] = label[:label_len]
                shifted_labels.append(shifted_label)
            shifted_labels = torch.stack(shifted_labels, dim=0)
            return shifted_labels

        input_signal = audio_batch['audio_signal']
        input_signal_length = audio_batch['audio_signal_length']

        input_ids, input_length, labels, loss_mask = (
            audio_batch['tokens'],
            audio_batch['tokens_length'],
            audio_batch['labels'],
            audio_batch['loss_mask'],
        )

        # [b, t, c]
        encoded, encoded_len = self.perception(
            input_signal=input_signal,
            input_signal_length=input_signal_length,
            processed_signal=None,
            processed_signal_length=None,
        )
        # [b, t, c]
        lm_embedding = self.model.language_model.embedding
        input_embeds = lm_embedding.word_embeddings(input_ids)
        encoder_input, encoder_length = _concat_embs(encoded, encoded_len, input_embeds, input_length)
        labels = _shift_labels_by_emb_len(labels, input_length, encoded_len, encoder_input.shape[1], pad_token=0)
        # Loss mask where answer tokens are 1.0 and all other tokens are 0.0
        loss_mask = _shift_labels_by_emb_len(loss_mask, input_length, encoded_len, encoder_input.shape[1], pad_token=0)

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
            encoder_input = encoder_input
        encoder_input = encoder_input.transpose(0, 1).contiguous()
        if self.cfg.get("sequence_parallel", False):
            encoder_input = tensor_parallel.mappings.scatter_to_sequence_parallel_region(encoder_input)

        return encoder_input, attention_mask, labels, loss_mask, encoder_length

    def forward(
        self, audio_batch, checkpoint_activations_all_layers,
    ):
        """Forward pass of the model.

        We prepend audio embeddings to the instruction and label text tokens 
        as the LLM input.
        """
        encoder_input, attention_mask, labels, loss_mask, _ = self.prepare_llm_input(audio_batch)
        output = self.model(
            input_ids=None,
            position_ids=None,
            encoder_input=encoder_input,
            attention_mask=attention_mask,
            labels=labels,
            checkpoint_activations_all_layers=checkpoint_activations_all_layers,
        )

        return output, loss_mask

    def get_forward_output_and_loss_func(self, validation_step=False):
        def fwd_output_and_loss_func(dataloader_iter, model, checkpoint_activations_all_layers=None):
            batch = next(dataloader_iter)
            batch = {key: val.cuda(non_blocking=True) for key, val in batch.items()}
            output_tensor, loss_mask = self.forward(
                batch, checkpoint_activations_all_layers=checkpoint_activations_all_layers
            )
            output_tensor = output_tensor[0]  # get loss only, ingore logits

            def loss_func(output_tensor):
                # Loss for a micro-batch (ub)
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
        if 'augmentor' in data_cfg:
            augmentor = process_augmentations(
                data_cfg['augmentor'], global_rank=self.global_rank, world_size=self.world_size
            )
        else:
            augmentor = None

        if data_cfg.get('is_tarred', False):
            raise ValueError("Does not support tar.")

        if isinstance(data_cfg.manifest_filepath, str):
            manifest_filepath = data_cfg.manifest_filepath.split(',')
        else:
            manifest_filepath = data_cfg.manifest_filepath

        if not is_train:
            dataset = AudioQuestionAnswerDataset(
                manifest_filepath=manifest_filepath,
                tokenizer=self.tokenizer,
                sample_rate=data_cfg.sample_rate,
                int_values=data_cfg.get('int_values', False),
                augmentor=augmentor,
                max_duration=getattr(data_cfg, 'max_duration', None),
                min_duration=getattr(data_cfg, 'min_duration', None),
                max_utts=getattr(data_cfg, 'max_utts', -1),
                trim=getattr(data_cfg, 'trim_silence', False),
                channel_selector=getattr(data_cfg, 'channel_selector', None),
                max_seq_length=data_cfg.max_seq_length,
                min_seq_length=data_cfg.min_seq_length,
                add_bos=data_cfg.get('add_bos', False),
                add_eos=data_cfg.get('add_eos', True),
                add_sep=data_cfg.get('add_sep', False),
                sep_id=self.sep_id,
                max_num_samples=data_cfg.get('max_num_samples', None),
                seed=data_cfg.get('seed', 1234),
                separate_prompt_and_response_with_newline=data_cfg.get(
                    'separate_prompt_and_response_with_newline', True
                ),
                answer_only_loss=self.cfg.get('answer_only_loss', True),
                truncation_field=data_cfg.get('truncation_field', 'context'),
                pad_to_max_length=False,
                prompt_template=data_cfg.get('prompt_template', None),
                virtual_tokens=self.virtual_tokens,
                tokens_to_generate=data_cfg.get(
                    'tokens_to_generate', 0
                ),  # used at inference time to allocate tensor positions for tokens that will be generated by inf procedure.
            )
            return [dataset]

        else:
            datasets = []
            # Construct the data prefix list for `get_datasets_weights_and_num_samples()`
            # that is of the format [weight1,file_name1,weight2,file_name2,...]
            concat_sampling_probabilities = data_cfg.get('concat_sampling_probabilities', None)
            if concat_sampling_probabilities is None:
                concat_sampling_probabilities = [1.0 / len(manifest_filepath)] * len(manifest_filepath)
            elif len(data_cfg.get('concat_sampling_probabilities', None)) != len(manifest_filepath):
                raise ValueError(
                    (
                        f"concat_sampling_probabilities must be of the same size as manifest_filepath.",
                        f"Provided size {len(data_cfg.concat_sampling_probabilities)}, number of datasets {len(manifest_filepath)}",
                    )
                )
            data_prefix = []
            for weight, prefix in zip(concat_sampling_probabilities, manifest_filepath):
                data_prefix.append(weight)
                data_prefix.append(prefix)

            num_samples_per_dataset = get_num_samples_from_files(manifest_filepath)
            num_train_samples = [len(manifest_filepath) * max(num_samples_per_dataset)]
            _, _, num_train_samples_per_dataset = get_datasets_weights_and_num_samples(data_prefix, num_train_samples)
            num_train_samples_after_blend = sum([x[0] for x in num_train_samples_per_dataset])

            for file_path, num_samples in zip(manifest_filepath, num_train_samples_per_dataset):
                dataset = AudioQuestionAnswerDataset(
                    manifest_filepath=file_path,
                    tokenizer=self.tokenizer,
                    sample_rate=data_cfg.sample_rate,
                    int_values=data_cfg.get('int_values', False),
                    augmentor=augmentor,
                    max_duration=getattr(data_cfg, 'max_duration', None),
                    min_duration=getattr(data_cfg, 'min_duration', None),
                    max_utts=getattr(data_cfg, 'max_utts', -1),
                    trim=getattr(data_cfg, 'trim_silence', False),
                    channel_selector=getattr(data_cfg, 'channel_selector', None),
                    max_seq_length=data_cfg.max_seq_length,
                    min_seq_length=data_cfg.min_seq_length,
                    add_bos=data_cfg.get('add_bos', False),
                    add_eos=data_cfg.get('add_eos', True),
                    add_sep=data_cfg.get('add_sep', False),
                    sep_id=self.sep_id,
                    max_num_samples=num_samples[0],
                    seed=data_cfg.get('seed', 1234),
                    separate_prompt_and_response_with_newline=data_cfg.get(
                        'separate_prompt_and_response_with_newline', True
                    ),
                    answer_only_loss=self.cfg.get('answer_only_loss', True),
                    truncation_field=data_cfg.get('truncation_field', 'context'),
                    pad_to_max_length=False,
                    prompt_template=data_cfg.get('prompt_template', None),
                    virtual_tokens=self.virtual_tokens,
                    tokens_to_generate=data_cfg.get(
                        'tokens_to_generate', 0
                    ),  # used at inference time to allocate tensor positions for tokens that will be generated by inf procedure.
                )
                datasets.append(dataset)

            dataset = BlendableDataset(
                datasets=datasets, weights=concat_sampling_probabilities, size=num_train_samples_after_blend
            )
            return dataset

    @classmethod
    def _modify_config(cls, gpt_cfg, cfg, audio_cfg, add_cfg_to_tree=False):
        """
        This function modifies the original gpt pre-training config (gpt_cfg) with attributes from the finetuning config (cfg).
        The `add_cfg_to_tree` arg adds `cfg` to the top of the yaml tree which is needed for all `hparams.yaml` files when passed as an arg to `load_from_checkpoint()`.
        """
        OmegaConf.set_struct(gpt_cfg, True)
        OmegaConf.resolve(cfg)
        with open_dict(gpt_cfg):
            gpt_cfg.megatron_amp_O2 = cfg.model.get('megatron_amp_O2', False)
            gpt_cfg.micro_batch_size = cfg.model.data.train_ds.micro_batch_size
            gpt_cfg.global_batch_size = cfg.model.data.train_ds.global_batch_size
            gpt_cfg.sequence_parallel = cfg.model.get("sequence_parallel", False)
            gpt_cfg.activations_checkpoint_granularity = cfg.model.get("activations_checkpoint_granularity", None)
            gpt_cfg.activations_checkpoint_num_layers = cfg.model.get("activations_checkpoint_num_layers", None)
            gpt_cfg.activations_checkpoint_method = cfg.model.get("activations_checkpoint_method", None)
            gpt_cfg.data = cfg.model.data
            gpt_cfg.optim = cfg.model.optim
            gpt_cfg.precision = cfg.trainer.precision
            gpt_cfg.answer_only_loss = cfg.model.answer_only_loss
            gpt_cfg.restore_from_path = cfg.model.restore_from_path
            gpt_cfg.resume_from_checkpoint = cfg.model.resume_from_checkpoint
            gpt_cfg.save_nemo_on_validation_end = cfg.model.save_nemo_on_validation_end
            gpt_cfg.gradient_as_bucket_view = cfg.model.gradient_as_bucket_view
            gpt_cfg.hidden_dropout = cfg.model.get('hidden_dropout', 0.0)
            gpt_cfg.attention_dropout = cfg.model.get('attention_dropout', 0.0)
            gpt_cfg.ffn_dropout = cfg.model.ffn_dropout
            gpt_cfg.peft = cfg.model.peft
            # for AudioGPTLoRAModel
            gpt_cfg.target = f"{cls.__module__}.{cls.__name__}"
            gpt_cfg.perception = cfg.model.perception
            gpt_cfg.perception.preprocessor = audio_cfg.preprocessor
            gpt_cfg.perception.encoder = audio_cfg.encoder
            matcher_cfg = gpt_cfg.perception.matcher
            matcher_cfg.feat_in = audio_cfg.encoder.d_model
            gpt_cfg.perception.output_dim = gpt_cfg.hidden_size
            # This is needed when modifying a hparam file directly to load `.ckpt` files.
            # This is not needed to modify the cfg in `.nemo` files.
            if add_cfg_to_tree:
                OmegaConf.resolve(gpt_cfg)
                gpt_cfg.cfg = gpt_cfg

        return gpt_cfg

    @classmethod
    def restore_from_pretrained_models(
        cls, cfg: Optional[Union[OmegaConf, str]] = None, trainer: Optional[Trainer] = None,
    ):
        if not cfg.model.pretrained_audio_model:
            raise RuntimeError("PEFT training needs a pretrained audio model present.")

        if not cfg.model.restore_from_path:
            raise RuntimeError("PEFT training needs a trained base model present.")

        base_model_save_restore_connector = NLPSaveRestoreConnector()
        if os.path.isdir(cfg.model.restore_from_path):
            base_model_save_restore_connector.model_extracted_dir = cfg.model.restore_from_path
        base_model_cfg = cls.restore_from(
            restore_path=cfg.model.restore_from_path,
            trainer=trainer,
            return_config=True,
            save_restore_connector=base_model_save_restore_connector,
        )
        pretrained_audio_model = cfg.model.pretrained_audio_model
        if pretrained_audio_model.endswith('.nemo'):
            logging.info(f'Loading pretrained audio model from local file: {pretrained_audio_model}')
            audio_model = ASRModel.restore_from(pretrained_audio_model, map_location='cpu')
        else:
            logging.info(f'Loading pretrained audio model from NGC: {pretrained_audio_model}')
            audio_model = ASRModel.from_pretrained(pretrained_audio_model, map_location='cpu')

        model_cfg = cls._modify_config(base_model_cfg, cfg, audio_model.cfg, add_cfg_to_tree=False)
        resume_from_checkpoint = trainer._checkpoint_connector.resume_from_checkpoint_fit_path
        save_restore_connector = PEFTSaveRestoreConnector(
            peft_model_nemo_path=cfg.model.peft.restore_from_path, peft_model_ckpt_path=resume_from_checkpoint
        )
        if os.path.isdir(cfg.model.restore_from_path):
            save_restore_connector.model_extracted_dir = cfg.model.restore_from_path

        # load llm
        model = cls.restore_from(
            restore_path=cfg.model.restore_from_path,
            trainer=trainer,
            override_config_path=model_cfg,
            save_restore_connector=save_restore_connector,
            strict=False,
        )
        # load am
        model.perception.encoder.load_state_dict(audio_model.encoder.state_dict(), strict=True)
        logging.info(f'Loaded pretrained audio model from {pretrained_audio_model}')
        return model

    def state_dict(self, destination=None, prefix=None, keep_vars=False):
        if self.setup_complete:
            # Once setup is complete we only need adapter and perception model.
            return_state_dict = self.get_peft_state_dict()
            state_dict = self.model.state_dict(prefix="model.perception.")
            return_state_dict.update(state_dict)
            return return_state_dict
        else:
            # we want all the params with the same keys as calling self.state_dict()
            # but we can't call self.state_dict() here as it would be a recursive call.
            # so we call self.model.state_dict(prefix="model.") which will return all the keys and params same as calling self.state_dict()
            return self.model.state_dict(prefix="model.")

    def load_state_dict(self, state_dict, strict: bool = True):
        if self.setup_complete:
            print(f"loading state_dict: {state_dict.keys()}")
            super(MegatronGPTPEFTModel, self).load_state_dict(state_dict, strict=False)
        else:
            super(MegatronGPTPEFTModel, self).load_state_dict(state_dict, strict=strict)
