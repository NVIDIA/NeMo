# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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
import json
import os
from functools import partial
from typing import Any, Optional

import torch
import torch.nn.functional as F
from omegaconf import DictConfig, ListConfig, OmegaConf
from omegaconf.omegaconf import open_dict
from pytorch_lightning.trainer.trainer import Trainer
from torch import Tensor

from nemo.collections.common.metrics import MetricStringToTorchMetric
from nemo.collections.common.metrics.classification_accuracy import ExactStringPerCategoryMatchMetric
from nemo.collections.nlp.data.language_modeling.megatron.base_dataset_utils import (
    get_datasets_weights_and_num_samples,
)
from nemo.collections.nlp.data.language_modeling.megatron.blendable_dataset import BlendableDataset
from nemo.collections.nlp.data.language_modeling.megatron.gpt_universal_prompt_learning_dataset import (
    GPTUniversalPromptLearningT0Dataset,
)
from nemo.collections.nlp.data.language_modeling.megatron.megatron_batch_samplers import (
    MegatronPretrainingBatchSampler,
)
from nemo.collections.nlp.models.language_modeling.megatron_base_model import MegatronBaseModel
from nemo.collections.nlp.models.language_modeling.megatron_gpt_model import MegatronGPTModel
from nemo.collections.nlp.modules.common.megatron.utils import (
    average_losses_across_data_parallel_group,
    init_method_normal,
    scaled_init_method_normal,
)
from nemo.collections.nlp.modules.common.text_generation_utils import (
    get_model_parallel_src_rank,
    megatron_gpt_generate,
)
from nemo.collections.nlp.modules.common.transformer.text_generation import LengthParam, SamplingParam, TextGeneration
from nemo.collections.nlp.modules.common.universal_prompt_encoder import UniversalPromptEncoder
from nemo.collections.nlp.parts.nlp_overrides import NLPSaveRestoreConnector
from nemo.collections.nlp.parts.utils_funcs import get_last_rank
from nemo.utils import logging

try:
    from apex.transformer import parallel_state, tensor_parallel
    from apex.transformer.pipeline_parallel.schedules.fwd_bwd_pipelining_without_interleaving import (
        forward_backward_pipelining_without_interleaving,
    )
    from apex.transformer.pipeline_parallel.schedules.fwd_bwd_no_pipelining import forward_backward_no_pipelining

    HAVE_APEX = True

except (ImportError, ModuleNotFoundError):
    HAVE_APEX = False


__all__ = ['MegatronGPTUniversalPromptLearningModel']


class MegatronGPTUniversalPromptLearningModel(MegatronBaseModel, TextGeneration):
    """
    """

    def __init__(self, cfg: DictConfig, trainer: Trainer):
        super().__init__(cfg, trainer)
        if self.trainer.precision == 32:
            self.autocast_dtype = torch.float
        elif self.trainer.precision == 16:
            self.autocast_dtype = torch.half
        elif self.trainer.precision == 'bf16':
            self.autocast_dtype = torch.bfloat16
        else:
            raise ValueError('precision must be in [32, 16, "bf16"]')

        self.cfg = cfg
        # TODO: Enable amp_o2 training
        self.megatron_amp_o2 = False
        self.sequence_parallel = self.cfg.get("sequence_parallel", False)
        self.backup_sequence_parallel = self.cfg.get("sequence_parallel", False)
        self.backup_activations_checkpoint_granularity = self.cfg.get("activations_checkpoint_granularity", None)
        self.backup_activations_checkpoint_method = self.cfg.get('activations_checkpoint_method', None)
        self.load_frozen_model(self.cfg, trainer)
        self.pipeline_parallel = self.cfg.get('pipeline_model_parallel_size', 1) > 1
        self.tokenizer = self.frozen_model.tokenizer
        self.pad_token_id = self.tokenizer.pad_id if self.tokenizer.pad_id is not None else self.tokenizer.unk_id

        # Load templates for assigning virtual prompt token positions

        # make sure the default pytorch lightning gradient clipping in the basemodel
        self.grad_clip_pl_default = True
        self.lowest_val_loss = None

        self._reduced_loss_buffer = []
        self._inference_config = None
        self.hidden_size = self.frozen_model.cfg.hidden_size
        self.padded_vocab_size = self.frozen_model.padded_vocab_size
        if self.frozen_model.model.pre_process:
            self.word_embeddings = self.frozen_model.model.language_model.embedding.word_embeddings
        self._prompt_encoder_key = 'prompt_encoder'
        self.val_metric, self.val_metric_name = self.setup_metric(self.cfg.data.validation_ds)
        self.val_metric = torch.nn.ModuleList(self.val_metric)

    @property
    def _metrics_require_string2category_map(self):
        return set(["f1", "accuracy", "average_precision"])

    @property
    def virtual_token_length(self):
        vlen = 0
        for pcfg in self.cfg.perceiver:
            vlen += pcfg.hidden_steps
        return vlen

    def setup_metric(self, data_cfg):
        # XNLI is a special case.
        metric_name = "exact_string_match"
        if hasattr(self.cfg, "eval_languages"):
            metric = [ExactStringPerCategoryMatchMetric(self.cfg.eval_languages)]
        else:
            if not hasattr(data_cfg, "metric"):
                metric = MetricStringToTorchMetric["exact_string_match"]
            else:
                if not hasattr(data_cfg.metric, "name"):
                    raise ValueError("Metric name is not provided in the metric config.")
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
            metric = MetricStringToTorchMetric[metric_name]
            # GLUE will not have a "src_file_name" attribute and will always have only a single metric.
            if hasattr(data_cfg, "src_file_name") or hasattr(data_cfg, "file_names"):
                if hasattr(data_cfg, "src_file_name") and isinstance(data_cfg.src_file_name, ListConfig):
                    # We pass average and num_classes to the metric constructor via kwargs even if they don't exist for each metric.
                    metric = [
                        metric(average=data_cfg.metric.average, num_classes=data_cfg.metric.num_classes)
                        for _ in range(len(data_cfg.src_file_name))
                    ]
                elif hasattr(data_cfg, "file_names") and isinstance(data_cfg.file_names, ListConfig):
                    metric = [
                        metric(average=data_cfg.metric.average, num_classes=data_cfg.metric.num_classes)
                        for _ in range(len(data_cfg.file_names))
                    ]
                else:
                    metric = [metric(average=data_cfg.metric.average, num_classes=data_cfg.metric.num_classes)]
            else:
                metric = [metric()]  # GLUE does need to specify average or num_classes.

        return metric, metric_name

    def add_virtual_prompt_params_to_param_group(self):
        """
        Passes only prompt table and prompt encoder params to the optimizer.
        """
        virtual_prompt_params = {'params': []}
        for i, pcfg in enumerate(self.cfg.perceiver):
            if pcfg.trainable:
                virtual_prompt_params['params'].extend([param for param in self.prompt_encoder[i].parameters()])
        self._optimizer_param_groups = (virtual_prompt_params,)

    def load_frozen_model(self, cfg, trainer):
        save_restore_connector = NLPSaveRestoreConnector()

        # Load frozen model from unpacked directory
        if os.path.isdir(cfg.get('language_model_path')):
            save_restore_connector.model_extracted_dir = cfg.get('language_model_path')

        frozen_model_cfg = MegatronGPTModel.restore_from(
            cfg.get('language_model_path'),
            trainer=trainer,
            return_config=True,
            save_restore_connector=save_restore_connector,
        )

        # Need to overwrite some params in frozen model's config before restoring
        with open_dict(frozen_model_cfg):
            frozen_model_cfg.megatron_amp_O2 = False
            frozen_model_cfg.optim.name = "fused_adam"
            frozen_model_cfg.micro_batch_size = self.cfg.micro_batch_size
            frozen_model_cfg.global_batch_size = self.cfg.global_batch_size
            frozen_model_cfg.precision = trainer.precision
            frozen_model_cfg.sequence_parallel = self.sequence_parallel
            frozen_model_cfg.activations_checkpoint_granularity = self.cfg.get(
                "activations_checkpoint_granularity", None
            )
            frozen_model_cfg.activations_checkpoint_num_layers = self.cfg.get(
                "activations_checkpoint_num_layers", None
            )
            frozen_model_cfg.activations_checkpoint_method = self.cfg.get("activations_checkpoint_method", None)

        if cfg.get('language_model_path', None):
            self.frozen_model = MegatronGPTModel.restore_from(
                cfg.get('language_model_path'),
                trainer=trainer,
                save_restore_connector=save_restore_connector,
                override_config_path=frozen_model_cfg,
            ).to(dtype=self.autocast_dtype)

    def state_dict(self, destination=None, prefix=None, keep_vars=False):
        """
        Custom state dict that only contains prompt table and prompt encoder parameters. 
        No frozen model parameters are stored in the state dict. Prompt encoder parameters 
        are only in state dict for intermediate checkpoints saved during training. Final
        nemo checkpoints at the end of training will contain prompt table parameters only. 
        """
        if self.frozen_model.model.pre_process:
            state_dict_ = {}
            state_dict_[self._prompt_encoder_key] = self.prompt_encoder.state_dict()
            return state_dict_
        else:
            state_dict_ = {}

            return state_dict_

    def load_state_dict(self, state_dict, strict: bool = True):
        """
        Custom load state dict method that only loads prompt table and prompt encoder
        parameters. Matching load method for this class' custom state dict method. 
        """
        if self.frozen_model.model.pre_process and self._prompt_encoder_key in state_dict:
            state_dict_ = state_dict[self._prompt_encoder_key]

            if not hasattr(self, "prompt_encoder"):
                self.init_prompt_encoder()
            self.prompt_encoder.load_state_dict(state_dict_, strict)

    def setup_optimizer_param_groups(self):
        """
        ModelPT override. Optimizer will get self._optimizer_param_groups. 
        Only want virtual prompt params to be passed to the optimizer.
        """
        # Freeze frozen model
        for param in self.frozen_model.parameters():
            param.requires_grad = False

        if self.frozen_model.model.pre_process:
            self.add_virtual_prompt_params_to_param_group()
        else:
            self._optimizer_param_groups = ({'params': []},)

    def set_input_tensor(self, input_tensor):
        """Set input tensor to be used instead of forward()'s input.
        When doing pipeline parallelism the input from the previous
        stage comes from communication, not from the input, so the
        model's forward_step_func won't have it. This function is thus
        used by internal code to bypass the input provided by the
        forward_step_func"""

        self.frozen_model.model.set_input_tensor(input_tensor)

    def embed_input_train(self, input_ids: Tensor, prompt_input_mask):
        # Replace virtual token ids with padding for forward pass through vocab embeddings
        discrete_token_ids = input_ids.clone()
        discrete_token_embeds = self.word_embeddings(discrete_token_ids).clone()
        # [b, s, d] -> [s, b, d]
        prompt_input_emb = discrete_token_embeds.transpose(0, 1).contiguous()
        v_embeds_list = []
        for i, p_encoder in enumerate(self.prompt_encoder):
            if not self.cfg.perceiver[i].trainable:
                # make sure p_encoder in eval mode
                p_encoder.eval()
            virtual_token_embeds = p_encoder(prompt_input_emb, prompt_input_mask)
            #  [s, b, d] -> [b, s, d]
            virtual_token_embeds = virtual_token_embeds.transpose(0, 1).contiguous()
            v_embeds_list.append(virtual_token_embeds)
        v_embeds = torch.concat(v_embeds_list, axis=1)
        return v_embeds, discrete_token_embeds

    def training_step(self, batch, batch_idx):
        # we zero grads here because we also call backward in the apex fwd/bwd functions
        self._optimizer.zero_grad()
        loss_mean = self.fwd_bwd_step(batch, forward_only=False)
        self.allreduce_gradients()

        # logging
        # we can only log on one rank if it is rank zero so we broadcast from last rank
        # we can avoid this broadcast by updating the PTL log function to accept specific ranks
        torch.distributed.broadcast(loss_mean, get_last_rank())

        if self.cfg.precision == 16 and hasattr(self.trainer.precision_plugin.scaler, "_scale"):
            loss_scale = self.trainer.precision_plugin.scaler._scale
            if loss_scale is not None:
                self.log('loss_scale', loss_scale)

        self.log('reduced_train_loss', loss_mean, prog_bar=True, rank_zero_only=True)
        lr = self._optimizer.param_groups[0]['lr']
        self.log('lr', lr, rank_zero_only=True)
        self.log('global_step', self.trainer.global_step, prog_bar=True, rank_zero_only=True)
        self.log(
            'consumed_samples',
            self.compute_consumed_samples(self.trainer.global_step - self.init_global_step),
            prog_bar=True,
            rank_zero_only=True,
        )
        return loss_mean

    def forward(
        self,
        input_ids,
        position_ids,
        attention_mask,
        prompt_input_mask,
        labels=None,
        inference=True,
        set_inference_key_value_memory=False,
        inference_max_sequence_len=None,
    ):
        """
        Special forward method for p-tuning/prompt-tuning pretrained
        GPT style models. Bypasses the vocab token preprocessing done
        in the MegatronGPT class.
        """
        # Get embeddings for text tokens and insert virtual token embeddings
        if self.frozen_model.model.pre_process:
            if inference and set_inference_key_value_memory:
                all_input_ids = input_ids[0]  # all the input tokens, used to compute virtual token emb
                input_ids = input_ids[1]  # the input token ids so far to generate the next token id
                virtual_token_emb, _ = self.embed_input_train(all_input_ids, prompt_input_mask)
                input_embeds = self.word_embeddings(input_ids).clone()
            elif inference and not set_inference_key_value_memory:
                input_ids = input_ids[1]
                discrete_token_ids = input_ids.clone()
                input_embeds = self.word_embeddings(discrete_token_ids).clone()
                virtual_token_emb = None
            else:
                virtual_token_emb, input_embeds = self.embed_input_train(input_ids, prompt_input_mask)

            position_embeddings = self.frozen_model.model.language_model.embedding.position_embeddings(position_ids)
            encoder_input = input_embeds + position_embeddings
            if virtual_token_emb is not None:
                encoder_input = torch.concat([virtual_token_emb, encoder_input], axis=1)
            encoder_input = encoder_input.transpose(0, 1).contiguous()
            if self.sequence_parallel:
                encoder_input = tensor_parallel.mappings.scatter_to_sequence_parallel_region(encoder_input)
        else:
            encoder_input = None

        # Call forward on GPT model with preprocessed embeddings
        if self.autocast_dtype == torch.float32:
            output = self.frozen_model.model(
                input_ids=None,
                position_ids=None,
                encoder_input=encoder_input,
                attention_mask=attention_mask,
                labels=labels,
                set_inference_key_value_memory=set_inference_key_value_memory,
                inference_max_sequence_len=inference_max_sequence_len,
            )
        else:
            with torch.autocast(device_type="cuda", dtype=self.autocast_dtype):
                output = self.frozen_model.model(
                    input_ids=None,
                    position_ids=None,
                    encoder_input=encoder_input,
                    attention_mask=attention_mask,
                    labels=labels,
                    set_inference_key_value_memory=set_inference_key_value_memory,
                    inference_max_sequence_len=inference_max_sequence_len,
                )

        return output

    def on_validation_start(self) -> None:
        self._overwrite_checkpointing_for_inference(False, None, None)
        torch.distributed.barrier()
        return super().on_validation_start()

    def _overwrite_checkpointing_for_inference(
        self, sequence_parallel, activations_checkpoint_granularity, activations_checkpoint_method
    ):
        self.cfg['sequence_parallel'] = sequence_parallel
        self.cfg["activations_checkpoint_granularity"] = activations_checkpoint_granularity
        self.cfg['activations_checkpoint_method'] = activations_checkpoint_method
        for name, layer in self.named_modules():
            if name.startswith('prompt_encoder'):
                # skip prompt encoder module
                continue
            if hasattr(layer, 'sequence_parallel'):
                layer.sequence_parallel = sequence_parallel
            if hasattr(layer, "sequence_parallel_enabled"):
                layer.sequence_parallel_enabled = sequence_parallel
            if hasattr(layer, "activations_checkpoint_granularity"):
                layer.activations_checkpoint_granularity = activations_checkpoint_granularity
            if hasattr(layer, "activations_checkpoint_method"):
                layer.activations_checkpoint_method = activations_checkpoint_method

    def fwd_bwd_step(self, batch, forward_only):
        """
            Dataloader produces a global batch which is turned into a list of microbatches.
            The list of microbatches is then piped through the pipeline using Apex fwd/bwd functions.
        """
        disable_autocast = False
        sequence_parallel_enabled = self.sequence_parallel
        # Get seq length of batch
        _, seq_length = batch[0].shape
        tensor_shape = [seq_length + self.virtual_token_length, self.cfg.micro_batch_size, self.hidden_size]

        if self.pipeline_parallel:
            losses_reduced_per_micro_batch = forward_backward_pipelining_without_interleaving(
                forward_step_func=self.get_forward_output_and_loss_func(),
                batch=batch,
                model=self,
                forward_only=forward_only,
                tensor_shape=tensor_shape,
                dtype=self.autocast_dtype,
                disable_autocast=disable_autocast,
                grad_scaler=self.trainer.precision_plugin.scaler if self.cfg.precision == 16 else None,
                sequence_parallel_enabled=sequence_parallel_enabled,
            )
        else:
            losses_reduced_per_micro_batch = forward_backward_no_pipelining(
                forward_step_func=self.get_forward_output_and_loss_func(),
                batch=batch,
                model=self,
                forward_only=forward_only,
                tensor_shape=tensor_shape,
                dtype=self.autocast_dtype,
                disable_autocast=disable_autocast,
                grad_scaler=self.trainer.precision_plugin.scaler if self.cfg.precision == 16 else None,
            )

        # only the last stages of the pipeline return losses
        if losses_reduced_per_micro_batch:
            # average loss across micro batches
            loss_tensors_list = [loss_reduced['avg'] for loss_reduced in losses_reduced_per_micro_batch]
            loss_tensor = torch.concat(loss_tensors_list)
            loss_mean = loss_tensor.mean()
        else:
            # we're not on the last pipeline stage so no losses
            loss_mean = torch.tensor(0.0).cuda()

        return loss_mean

    def get_forward_output_and_loss_func(self):
        def fwd_output_and_loss_func(batch, model):
            batch = [x.cuda(non_blocking=True) for x in batch]
            input_ids, labels, loss_mask, position_ids, attention_mask, prompt_input_mask, answer_starts = batch
            labels = F.pad(labels, (self.virtual_token_length, 0, 0, 0), value=0)
            loss_mask = F.pad(loss_mask, (self.virtual_token_length, 0, 0, 0), value=0)
            output_tensor = model(input_ids, position_ids, attention_mask, prompt_input_mask, labels, inference=False)

            if isinstance(output_tensor, tuple):
                output_tensor, _ = output_tensor

            def loss_func(output_tensor):
                loss = self.frozen_model.loss_func(loss_mask, output_tensor)
                reduced_loss = average_losses_across_data_parallel_group([loss])
                return loss, {'avg': reduced_loss}

            return output_tensor, loss_func

        return fwd_output_and_loss_func

    def init_prompt_encoder(self):
        """
        Init the prompt encoder needed for universal prompt encoder
        """
        if not parallel_state.is_unitialized():
            padding_emb = self.word_embeddings(torch.tensor([self.pad_token_id]).cuda())
        perceiver_confs = OmegaConf.to_container(self.cfg.perceiver)
        if not hasattr(self, 'prompt_encoder'):
            self.prompt_encoder = torch.nn.ModuleList()
        for i, perceiver_conf in enumerate(perceiver_confs):
            trainable = perceiver_conf['trainable']
            if not trainable and len(perceiver_confs) == len(self.prompt_encoder):
                # not trainable and prompt encoder is populated already, use the previous initialized weights
                continue
            init_method_std = perceiver_conf['init_method_std']
            if i == len(self.prompt_encoder):
                # only instantiate it when it doesn't exist
                del perceiver_conf['init_method_std']
                encoder_init = init_method_normal(init_method_std)
                output_init = scaled_init_method_normal(init_method_std, perceiver_conf['num_layers'])
                perceiver_conf['init_method'] = encoder_init
                perceiver_conf['output_layer_init_method'] = output_init
                del perceiver_conf['trainable']
                module = UniversalPromptEncoder(perceiver_conf, output_dim=self.frozen_model.cfg.hidden_size)
            else:
                module = self.prompt_encoder[i]
            if not trainable:
                for param in module.parameters():
                    # no grad
                    param.requires_grad = False
            # # zero out parameters
            if init_method_std == 0.0 and trainable:
                torch.nn.init.constant_(module.output_linear.weight, 0.0)
                if not parallel_state.is_unitialized():
                    with torch.no_grad():
                        module.output_linear.bias[:] = padding_emb[0]
            if i == len(self.prompt_encoder):
                # append a new prompt encoder only when it doesn't not exist
                self.prompt_encoder.append(module)

    def setup(self, stage=None):
        # NOTE: super().__init__ will try and setup train/val/test datasets, but we sidestep this using a if self._train_ds is not None condition
        # We then set things up for real only once setup() of this class is called.
        resume_checkpoint_path = self.trainer._checkpoint_connector.resume_from_checkpoint_fit_path
        if resume_checkpoint_path:
            init_consumed_samples = self._extract_consumed_samples_from_ckpt(resume_checkpoint_path)
        else:
            init_consumed_samples = 0
        self.init_consumed_samples = init_consumed_samples

        if stage == 'test' or stage == 'predict' or stage == 'validate':
            return

        if self.frozen_model.model.pre_process:
            self.init_prompt_encoder()

        self.setup_training_data()
        self.setup_validation_data()

    def setup_training_data(self, training_data_config=None):
        self._train_ds = self.build_dataset(data_cfg=self.cfg.data.train_ds, is_train=True,)
        consumed_samples = self.compute_consumed_samples(0)
        self._train_dl = self.build_data_loader(
            dataset=self._train_ds,
            data_cfg=self.cfg.data.train_ds,
            sequence_parallel=self.sequence_parallel,
            consumed_samples=consumed_samples,
        )

    def setup_validation_data(self, validation_data_config=None):
        self._validation_ds = self.build_dataset(data_cfg=self.cfg.data.validation_ds, is_train=False,)
        self._validation_dl = []
        for dataset in self._validation_ds:
            eval_dl = self.build_data_loader(
                dataset=dataset, data_cfg=self.cfg.data.validation_ds, sequence_parallel=False, consumed_samples=0,
            )
            self._validation_dl.append(eval_dl)

    def build_dataset(
        self, data_cfg, is_train=True,
    ):
        # Construct the data prefix list for `get_datasets_weights_and_num_samples()` that is of the format [weight1,file_name1,weight2,file_name2,...]
        if is_train:
            if data_cfg.concat_sampling_probabilities is None or not isinstance(
                data_cfg.concat_sampling_probabilities, ListConfig
            ):
                raise ValueError(
                    f"concat_sampling_probabilities must be a ListConfig with the same number of files in file_names. Found: {data_cfg.concat_sampling_probabilities}"
                )

            if len(data_cfg.get('concat_sampling_probabilities', None)) != len(data_cfg.file_names):
                raise ValueError(
                    f"concat_sampling_probabilities must be of the same size as file_names. Provided size {len(data_cfg.concat_sampling_probabilities)}, number of datasets {len(data_cfg.file_names)}"
                )

            data_prefix = []
            for weight, prefix in zip(data_cfg.concat_sampling_probabilities, data_cfg.file_names):
                data_prefix.append(weight)
                data_prefix.append(prefix)

            if self.trainer.max_steps is None or self.trainer.max_steps <= 0:
                raise ValueError(
                    f'Trainer max_steps must be set to a positive integer. Found {self.trainer.max_steps}'
                )
            num_train_samples = [self.trainer.max_steps * self.cfg.global_batch_size]
            _, _, num_train_samples_per_dataset = get_datasets_weights_and_num_samples(data_prefix, num_train_samples)
            num_train_samples_after_blend = sum([x[0] for x in num_train_samples_per_dataset])
        else:
            num_train_samples_per_dataset = [[None]] * len(data_cfg.file_names)

        datasets = []
        for file_path, num_samples in zip(data_cfg.file_names, num_train_samples_per_dataset):
            dataset = GPTUniversalPromptLearningT0Dataset(
                data=file_path,
                tokenizer=self.tokenizer,
                answer_only_loss=data_cfg.get('answer_only_loss', True),
                pad_token_id=self.pad_token_id,
                virtual_token_len=self.virtual_token_length,
                max_seq_length=data_cfg.max_seq_length,
                add_bos=data_cfg.get("add_bos", False),
                add_eos=data_cfg.get("add_eos", False),
                max_num_samples=num_samples[0],
                seed=data_cfg.get('seed', 1234),
            )
            datasets.append(dataset)
        if is_train:
            dataset = BlendableDataset(
                datasets=datasets, weights=data_cfg.concat_sampling_probabilities, size=num_train_samples_after_blend
            )
            return dataset
        else:
            return datasets

    def build_data_loader(self, dataset, data_cfg, sequence_parallel, consumed_samples=0):
        if isinstance(dataset, BlendableDataset):
            collate_fun = dataset.datasets[0].collate_fn
        else:
            collate_fun = dataset.collate_fn
        if sequence_parallel:
            collate_fn = partial(collate_fun, tp_workers=parallel_state.get_tensor_model_parallel_world_size())
        else:
            collate_fn = partial(collate_fun, tp_workers=0)

        batch_sampler = MegatronPretrainingBatchSampler(
            total_samples=len(dataset),
            consumed_samples=consumed_samples,
            micro_batch_size=self.cfg.micro_batch_size,
            global_batch_size=self.cfg.global_batch_size,
            data_parallel_rank=parallel_state.get_data_parallel_rank(),
            data_parallel_size=parallel_state.get_data_parallel_world_size(),
            drop_last=True,
        )
        return torch.utils.data.DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            collate_fn=collate_fn,
            num_workers=data_cfg.num_workers,
            pin_memory=data_cfg.pin_memory,
        )

    def _get_predicted_text(self, number_tokens, context_lengths, batch, result):
        preds_text = []
        labels_text = []
        for bid in range(len(number_tokens)):
            start_id = context_lengths[bid]
            pred_text_tokens = result['token_ids'][bid][start_id:]
            pred_text = self.frozen_model.tokenizer.ids_to_text(pred_text_tokens)
            label_text_tokens = batch[1][bid][start_id - 1 : start_id + number_tokens[bid] - 1]
            label_text = self.frozen_model.tokenizer.ids_to_text(label_text_tokens)
            preds_text.append(pred_text.strip())
            labels_text.append(label_text.strip())
        return preds_text, labels_text

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        loss_mean = self.fwd_bwd_step(batch, forward_only=True)
        torch.distributed.broadcast(loss_mean, get_last_rank())
        # run inference
        input_ids = batch[0]
        # add three as buffer
        input_ids = torch.nn.functional.pad(input_ids, (0, 1 + 3, 0, 0))
        context_lengths = batch[-1]
        token_to_gen = input_ids.shape[1] - context_lengths.max()

        length_params: LengthParam = {
            "max_length": token_to_gen,
            "min_length": 1,
        }

        sampling_params: SamplingParam = {
            "use_greedy": True,
            "temperature": 1.0,
            "top_k": 0,
            "top_p": 0,
            "repetition_penalty": 1.0,
            "add_BOS": False,
            "all_probs": False,
            "compute_logprob": False,
        }

        result = megatron_gpt_generate(
            self, (input_ids, context_lengths), self.tokenizer, length_params, sampling_params
        )

        number_tokens = [int(i.sum()) for i in batch[2]]
        preds_text, labels_text = self._get_predicted_text(number_tokens, context_lengths, batch, result)
        metric = self.val_metric[dataloader_idx]
        for _, (pred, label) in enumerate(zip(preds_text, labels_text)):
            _ = metric(pred.lower(), label.lower())

        return {
            'loss': loss_mean,
            'preds': preds_text,
            'labels': labels_text,
        }

    def set_inference_config(self, inference_config, data_cfg):
        self._inference_config = inference_config
        self._test_data_cfg = data_cfg

    def get_inference_config(self):
        return self._inference_config

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

    def validation_epoch_end(self, outputs):
        return self.common_epoch_end(outputs, 'validation')

    def common_epoch_end(self, outputs, mode='validation'):
        averaged_loss = []
        averaged_metric = []
        metric_name = self.val_metric_name if mode == 'validation' else self.test_metric_name
        # Log metrics for each provided validation/test dataset.
        if isinstance(outputs[0], dict):
            outputs = [outputs]
        for dataloader_idx, output in enumerate(outputs):
            if parallel_state.is_pipeline_last_stage():
                # only the last pipeline parallel stages return loss
                loss = torch.stack([x['loss'] for x in output]).mean()
            else:
                loss = torch.tensor(0.0).cuda()

            # we can only log on one rank if it is rank zero so we broadcast from last rank
            torch.distributed.broadcast(loss, get_last_rank())
            # Determine the key used to log the loss based on the user provided name of the dataset or the dataloader index.
            loss_log_key = self._determine_log_key(self.cfg.data.validation_ds, dataloader_idx, "loss", mode)
            # Determine the key used to log the eval metric based on the user provided name of the dataset or the dataloader index.
            metric_log_key = self._determine_log_key(self.cfg.data.validation_ds, dataloader_idx, metric_name, mode)
            self.log(loss_log_key, loss)
            metric_object = (
                self.val_metric[dataloader_idx] if mode == 'validation' else self.test_metric[dataloader_idx]
            )
            metric = metric_object.compute()
            self.log(metric_log_key, metric)
            logging.info(f"{metric_log_key}: {metric}")
            metric_object.reset()

            averaged_loss.append(loss)
            averaged_metric.append(metric)

        # Logging of the averaged metrics:
        averaged_loss = sum(averaged_loss) / len(averaged_loss)
        averaged_metric = sum(averaged_metric) / len(averaged_metric)

        if mode == 'validation':
            self.log("validation_loss", averaged_loss)
            self.log(f"validation_{self.val_metric_name}", averaged_metric)
        elif mode == 'test':
            self.log("test_loss", averaged_loss)
            self.log(f"test_{self.test_metric_name}", averaged_metric)
        self._overwrite_checkpointing_for_inference(
            self.backup_sequence_parallel,
            self.backup_activations_checkpoint_granularity,
            self.backup_activations_checkpoint_method,
        )
        self.back_model_state = None
        self.back_opt_state = None
        torch.distributed.barrier()
        return averaged_loss, averaged_metric

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def test_epoch_end(self, outputs):
        return self.common_epoch_end(outputs, 'test')

    def on_predict_epoch_end(self, outputs):
        inputs = {}
        labels = {}
        preds = {}
        for dataloader_idx, output in enumerate(outputs):
            inputs_list = inputs.get(dataloader_idx, [])
            labels_list = labels.get(dataloader_idx, [])
            preds_list = preds.get(dataloader_idx, [])
            inputs_list.extend([i['input'] for i in output])
            labels_list.extend([i['labels'] for i in output])
            preds_list.extend([i['preds'] for i in output])
            inputs_list = list(itertools.chain(*inputs_list))
            labels_list = list(itertools.chain(*labels_list))
            preds_list = list(itertools.chain(*preds_list))
            if torch.distributed.get_rank() == get_model_parallel_src_rank():
                dp_rank = parallel_state.get_data_parallel_rank()
                if self._test_data_cfg.output_filepath:
                    task_name = self._test_data_cfg.names[dataloader_idx]
                    contents = []
                    for i, l, p in zip(inputs_list, labels_list, preds_list):
                        item = {'input': i, 'output': p, 'gt': l}
                        item_str = json.dumps(item)
                        contents.append(item_str)
                    with open(self._test_data_cfg.output_filepath + f'{dp_rank}_' + task_name + '.json', 'w') as f:
                        f.write('\n'.join(contents))
        torch.distributed.barrier()
        return inputs, labels, preds

    def on_train_end(self):
        pass

    def get_forward_output_only_func(self):
        """
        Used for generate method only for now.
        """

        def fwd_output_only_func(batch, model):
            extra_arg = {}
            (
                all_tokens,
                tokens,
                position_ids,
                attention_mask,
                prompt_input_mask,
                set_inference_key_value_memory,
                inference_max_sequence_len,
            ) = batch

            all_tokens = all_tokens.cuda()
            tokens = tokens.cuda()
            position_ids = position_ids.cuda()
            attention_mask = attention_mask.cuda()
            prompt_input_mask = prompt_input_mask.cuda()
            extra_arg['set_inference_key_value_memory'] = set_inference_key_value_memory[0].item()
            extra_arg['inference_max_sequence_len'] = inference_max_sequence_len[0].item()

            output_tensor = model((all_tokens, tokens), position_ids, attention_mask, prompt_input_mask, **extra_arg)

            def id_func(output_tensor):
                return output_tensor, {'logits': output_tensor}

            return output_tensor, id_func

        return fwd_output_only_func

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: Optional[int] = None) -> Any:
        inference_config = self.get_inference_config()
        if inference_config is None:
            return None
        else:
            input_ids = batch[0]
            # add three as buffer
            input_ids = torch.nn.functional.pad(input_ids, (0, 1 + 3, 0, 0))
            context_lengths = batch[-1]
            token_to_gen = input_ids.shape[1] - context_lengths.max()
            length_params: LengthParam = {
                "max_length": token_to_gen,
                "min_length": inference_config["min_tokens_to_generate"],
            }

            sampling_params: SamplingParam = {
                "use_greedy": inference_config["greedy"],
                "temperature": inference_config["temperature"],
                "top_k": inference_config["top_k"],
                "top_p": inference_config["top_p"],
                "repetition_penalty": inference_config["repetition_penalty"],
                "add_BOS": inference_config["add_BOS"],
                "all_probs": inference_config["all_probs"],
                "compute_logprob": inference_config["compute_logprob"],
            }

            input_texts = []
            for bid in range(len(input_ids)):
                input_text = self.tokenizer.ids_to_text(input_ids[bid][: context_lengths[bid]])
                input_texts.append(input_text)

            # Call same generate code as in MegatronGPT
            result = megatron_gpt_generate(
                self.cuda(), (input_ids, context_lengths), self.tokenizer, length_params, sampling_params
            )

            number_tokens = [int(i.sum()) for i in batch[2]]
            preds_text, labels_text = self._get_predicted_text(number_tokens, context_lengths, batch, result)
            return {
                'preds': preds_text,
                'labels': labels_text,
                'input': input_texts,
            }

    def backward(self, *args, **kwargs):
        """ LightningModule hook to do backward.
            We want this to do nothing since we run backward in the fwd/bwd functions from apex.
            No need to call it here.
        """
        return

    def optimizer_zero_grad(self, *args, **kwargs):
        """ LightningModule hook to zero grad.
            We want this to do nothing as we are zeroing grads during the training_step.
        """
        return

    @classmethod
    def list_available_models(cls):
        pass
