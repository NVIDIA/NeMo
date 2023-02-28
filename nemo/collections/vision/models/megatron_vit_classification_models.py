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
from typing import Any, List, Optional, Union
from functools import partial

import numpy as np
import torch
from omegaconf.dictconfig import DictConfig
from pytorch_lightning.trainer.trainer import Trainer

from nemo.collections.vision.data.megatron.vit_dataset import build_train_valid_datasets
from nemo.collections.vision.data.megatron.data_samplers import MegatronVisionPretrainingRandomBatchSampler
from nemo.collections.nlp.data.language_modeling.megatron.megatron_batch_samplers import (
    MegatronPretrainingBatchSampler,
    MegatronPretrainingRandomBatchSampler,
)

from nemo.collections.vision.models.vision_base_model import MegatronVisionModel
from nemo.collections.vision.modules.vit.vit_backbone import VitBackbone, VitMlpHead

from nemo.collections.nlp.modules.common.megatron.module import (
    MegatronModule,
    Float16Module,
)

from nemo.collections.nlp.modules.common.megatron.utils import (
    ApexGuardDefaults,
    get_linear_layer,
    init_method_normal,
    parallel_lm_logits,
    scaled_init_method_normal,
    average_losses_across_data_parallel_group,
    get_all_params_for_weight_decay_optimization,
    get_params_for_weight_decay_optimization,
)

from nemo.collections.nlp.parts.utils_funcs import get_last_rank
from nemo.utils import logging
from nemo.core.classes.common import PretrainedModelInfo

try:
    from apex.transformer import parallel_state
    from apex.transformer.pipeline_parallel.schedules.common import build_model
    from apex.transformer.pipeline_parallel.schedules.fwd_bwd_pipelining_without_interleaving import (
        forward_backward_pipelining_without_interleaving,
    )
    from apex.transformer.pipeline_parallel.schedules.fwd_bwd_pipelining_with_interleaving import (
        _forward_backward_pipelining_with_interleaving,
    )
    from apex.transformer.pipeline_parallel.schedules.fwd_bwd_no_pipelining import forward_backward_no_pipelining

    HAVE_APEX = True
except (ImportError, ModuleNotFoundError):
    HAVE_APEX = False


class VitClassificationModel(MegatronModule):
    """Vision Transformer Model."""

    def __init__(self, model_cfg, num_classes, finetune=False,
                 pre_process=True, post_process=True):
        super(VitClassificationModel, self).__init__()

        scaled_init_method = (
            scaled_init_method_normal(model_cfg.init_method_std, model_cfg.num_layers)
            if model_cfg.use_scaled_init_method
            else init_method_normal(model_cfg.init_method_std)
        )

        self.hidden_size = model_cfg.hidden_size
        self.num_classes = num_classes
        self.finetune = finetune
        self.pre_process = pre_process
        self.post_process = post_process
        self.backbone = VitBackbone(
            model_cfg,
            init_method=init_method_normal(model_cfg.init_method_std),
            scaled_init_method=scaled_init_method,
            pre_process=self.pre_process,
            post_process=self.post_process,
            single_token_output=True
        )

        if self.post_process:
            if not self.finetune:
                self.head = VitMlpHead(self.hidden_size, self.num_classes)
            else:
                self.head = get_linear_layer(
                    self.hidden_size,
                    self.num_classes,
                    torch.nn.init.zeros_
                )

    def set_input_tensor(self, input_tensor):
        """See megatron.model.transformer.set_input_tensor()"""
        self.backbone.set_input_tensor(input_tensor)

    def forward(self, input):
        hidden_states = self.backbone(input)

        if self.post_process:
            hidden_states = self.head(hidden_states)
        hidden_states = hidden_states.contiguous()
        return hidden_states


class MegatronVitClassificationModel(MegatronVisionModel):
    """Megatron Vision Transformer Model."""

    def __init__(self, cfg: DictConfig, trainer: Trainer):
        if not HAVE_APEX:
            raise ImportError(
                "Apex was not found. Please see the NeMo README for installation instructions: https://github.com/NVIDIA/NeMo#megatron-gpt."
            )
        super().__init__(cfg, trainer=trainer)

        self._validate_trainer()

        #TODO(yuya): clean up all default values
        self.megatron_amp_o2 = cfg.get('megatron_amp_O2', False)

        if not self.megatron_amp_o2 and self.cfg.get('virtual_pipeline_model_parallel_size', None):
            raise ValueError('Virtual pipeline model parallel is only supported when using megatron_amp_O2')

        # build_model returns a list of modules which are used for interleaved pipeline parallelism
        self.model = build_model(
            model_provider_func=self.model_provider_func,
            wrap_with_ddp=False,
            virtual_pipeline_model_parallel_size=self.cfg.get('virtual_pipeline_model_parallel_size', None),
        )

        # if we're not using interleaved, then self.model is a module.
        if self.cfg.get('virtual_pipeline_model_parallel_size', None) is None:
            self.model = self.model[0]

        if self.megatron_amp_o2:

            if not self.with_distributed_adam:
                # Pre-allocate the model on GPU to have master parameters allocated on the same device with matching data type
                if isinstance(self.model, list):
                    for module in self.model:
                        module.cuda(torch.cuda.current_device())
                else:
                    self.model.cuda(torch.cuda.current_device())

            # Model wrapper to convert both model and inputs to half precision
            if isinstance(self.model, list):
                converted_model = []
                for module in self.model:
                    converted_model.append(Float16Module(module=module, precision=cfg.precision))
                    self.model = converted_model
            else:
                self.model = Float16Module(module=self.model, precision=cfg.precision)

        if self.trainer.precision == 32:
            self.autocast_dtype = torch.float
        elif self.trainer.precision == 16:
            self.autocast_dtype = torch.half
        elif self.trainer.precision == 'bf16':
            self.autocast_dtype = torch.bfloat16
        else:
            raise ValueError('precision must be in [32, 16, "bf16"]')


    def model_provider_func(self, pre_process, post_process):
        """Model depends on pipeline paralellism."""
        model = VitClassificationModel(
            model_cfg=self.cfg,
            num_classes=self.cfg.get("num_classes"), #TODO(yuya): clean this up
            finetune=self.cfg.get("finetune", False),
            pre_process=pre_process,
            post_process=post_process,
        )
        return model

    def setup_optimizer_param_groups(self):
        """ModelPT override. Optimizer will get self._optimizer_param_groups"""
        if self.cfg.get('do_layer_norm_weight_decay', False):
            if isinstance(self.model, list):
                self._optimizer_param_groups = get_all_params_for_weight_decay_optimization(self.model)
            else:
                self._optimizer_param_groups = get_all_params_for_weight_decay_optimization([self.model])

        else:
            self._optimizer_param_groups = get_params_for_weight_decay_optimization(self.model)

    def configure_optimizers(self):

        if self.with_distributed_adam:

            # Disable overlapped grad sync for embedding grad when
            # pipeline parallelism is enabled
            if parallel_state.get_pipeline_model_parallel_world_size() > 1:
                if parallel_state.is_pipeline_first_stage(ignore_virtual=True):
                    if isinstance(self.model, list):
                        module = self.model[0]  # only the first virtual rank has the embeddings
                    else:
                        module = self.model
                    # if module.share_token_embeddings:
                    #     param = module.word_embeddings_weight()
                    #     param._disable_greedy_grad_copy = not self.megatron_amp_o2
                    #     param._disable_overlap_grad_sync = True
                if parallel_state.is_pipeline_last_stage(ignore_virtual=True):
                    if isinstance(self.model, list):
                        module = self.model[-1]  # only the last virtual rank has the embeddings
                    else:
                        module = self.model
                    # if module.share_token_embeddings:
                    #     param = module.word_embeddings_weight()
                    #     param._disable_greedy_grad_copy = not self.megatron_amp_o2
                    #     param._disable_overlap_grad_sync = True

            # Disable overlapped grad sync for layer norm grads when
            # sequence parallelism is enabled
            for param in self.parameters():
                if getattr(param, 'sequence_parallel_enabled', False):
                    param._disable_greedy_grad_copy = not self.megatron_amp_o2
                    param._disable_overlap_grad_sync = True

        return super().configure_optimizers()

    def forward(self, tokens):
        output_tensor = self.model(tokens)
        return output_tensor

    def _get_fwd_bwd_function(self):
        if self.cfg.get('pipeline_model_parallel_size', 1) > 1:
            if self.cfg.get('virtual_pipeline_model_parallel_size', None) is not None:
                fwd_bwd_function = _forward_backward_pipelining_with_interleaving
            else:
                fwd_bwd_function = forward_backward_pipelining_without_interleaving
        else:
            fwd_bwd_function = forward_backward_no_pipelining
        return fwd_bwd_function

    def training_step(self, batch, batch_idx):
        """
            Our dataloaders produce a micro-batch and then we fetch
            a number of microbatches depending on the global batch size and model parallel size
            from the dataloader to produce a list of microbatches.
            Batch should be a list of microbatches and those microbatches should on CPU.
            Microbatches are then moved to GPU during the pipeline.
            The list of microbatches is then piped through the pipeline using Apex fwd/bwd functions.
        """

        # we zero grads here because we also call backward in the apex fwd/bwd functions
        self._optimizer.zero_grad()

        if parallel_state.is_pipeline_first_stage(ignore_virtual=True) or parallel_state.is_pipeline_last_stage(
            ignore_virtual=True
        ):
            # we prepare the micro batches for the apex fwd/bwd function
            batch_for_pipeline = self.process_global_batch(batch)
        else:
            # The intermediate pipeline stages do not need any inputs from data loader
            # GPT3 uses decoder with AttnMask:causal, thus doesn't need attention_mask
            batch_for_pipeline = None

        # TODO (yuya): fix this shape
        tensor_shape = [self.cfg.encoder_seq_length, self.cfg.micro_batch_size, self.cfg.hidden_size]

        # handle asynchronous grad reduction
        if self.with_distributed_adam:
            if self.megatron_amp_o2:
                # copy grads to main grad
                custom_sync_context_handler = lambda: self._optimizer.no_sync(greedy_grad_copy=True)
            else:
                # keep grad tensors around
                custom_sync_context_handler = lambda: self._optimizer.no_sync(greedy_grad_copy=False)
        else:
            if self.megatron_amp_o2 and not self.cfg.get('sequence_parallel', False):
                custom_sync_context_handler = self._optimizer.no_sync
            else:
                # TODO: enable async grad all reduce for O1/autocast mixed precision training
                custom_sync_context_handler = None

        # run forward and backwards passes for an entire global batch
        # we do this inside training_step to support pipeline parallelism
        fwd_bwd_function = self._get_fwd_bwd_function()

        losses_reduced_per_micro_batch = fwd_bwd_function(
            forward_step_func=self.get_forward_output_and_loss_func(),
            batch=batch_for_pipeline,
            model=self.model,
            forward_only=False,
            tensor_shape=tensor_shape,
            dtype=self.autocast_dtype,
            grad_scaler=self.trainer.precision_plugin.scaler if self.cfg.precision == 16 else None,
            custom_sync_context_handler=custom_sync_context_handler,
            sequence_parallel_enabled=self.cfg.get('sequence_parallel', False),
        )

        # only the last stages of the pipeline return losses
        if losses_reduced_per_micro_batch:
            # average loss across micro batches
            loss_tensors_list = [loss_reduced['loss'] for loss_reduced in losses_reduced_per_micro_batch]
            loss_tensor = torch.stack(loss_tensors_list)
            loss_mean = loss_tensor.mean()
        else:
            loss_mean = torch.tensor(0.0).cuda()

        # when using sequence parallelism, the sequence parallel layernorm grads must be all-reduced
        if self.cfg.get('tensor_model_parallel_size', 1) > 1 and self.cfg.get('sequence_parallel', False):
            self.allreduce_sequence_parallel_gradients()

        if self.with_distributed_adam:
            # launch grad reductions
            # Note: grads in first pipeline stage have already been
            # reduced
            if not parallel_state.is_pipeline_first_stage():
                self.reduce_overlap_gradients()
        elif self.megatron_amp_o2:
            # # when using pipeline parallelism grads must be all-reduced after the pipeline (not asynchronously)
            # if self.cfg.get('pipeline_model_parallel_size', 1) > 1 or self.cfg.get('sequence_parallel', False):
            #     # main grads are stored in the MainParamsOptimizer wrapper
            #     self._optimizer.allreduce_main_grads()
            self._optimizer.allreduce_main_grads()
        else:
            # async grad allreduce is not currently implemented for O1/autocasting mixed precision training
            # so we all-reduce gradients after the pipeline
            self.allreduce_gradients()  # @sangkug we think this is causing memory to blow up (hurts perf)

        # if self.cfg.get('pipeline_model_parallel_size', 1) > 1:
        #     # when using pipeline parallelism the first and last stage must keep embeddings in sync
        #     self.allreduce_first_last_embeddings()

        ## logging
        # we can only log on one rank if it is rank zero so we broadcast from last rank
        # we can avoid this broadcast by updating the PTL log function to accept specific ranks
        torch.distributed.broadcast(loss_mean, get_last_rank())

        if self.cfg.precision == 16:
            loss_scale = self.trainer.precision_plugin.scaler._scale
            if loss_scale is not None:
                self.log('loss_scale', loss_scale)

        self.log('reduced_train_loss', loss_mean, prog_bar=True, rank_zero_only=True)
        lr = self._optimizer.param_groups[0]['lr']
        self.log('lr', lr, rank_zero_only=True)
        self.log('global_step', self.trainer.global_step + 1, prog_bar=True, rank_zero_only=True)
        self.log(
            'consumed_samples',
            self.compute_consumed_samples(self.trainer.global_step + 1 - self.init_global_step),
            prog_bar=True,
            rank_zero_only=True,
        )

        return loss_mean

    def backward(self, *args, **kwargs):
        """ LightningModule hook to do backward.
            We want this to do nothing since we run backward in the fwd/bwd functions from apex.
            No need to call it here.
        """
        pass

    def optimizer_zero_grad(self, *args, **kwargs):
        """ LightningModule hook to zero grad.
            We want this to do nothing as we are zeroing grads during the training_step.
        """
        pass

    def _append_module_grads(self, module, grads):
        for param in module.parameters():
            if getattr(param, 'sequence_parallel_enabled', False):
                if self.megatron_amp_o2:
                    grad = param.main_grad
                else:
                    grad = param.grad
                grads.append(grad.data)

    def allreduce_sequence_parallel_gradients(self):
        """ All-reduce layernorm parameters across model parallel nodes when sequence parallelism is used.
            Modified from megatron-lm:
            https://gitlab-master.nvidia.com/ADLR/megatron-lm/-/blob/3f91f09bb2ab32f9904b47f46f19d2fc3f518ed8/megatron/training.py#L425
        """

        grads = []
        if isinstance(self.model, list):
            for module in self.model:
                self._append_module_grads(module, grads)
        else:
            self._append_module_grads(self.model, grads)

        coalesced = torch._utils._flatten_dense_tensors(grads)
        torch.distributed.all_reduce(coalesced, group=parallel_state.get_tensor_model_parallel_group())
        for buf, synced in zip(grads, torch._utils._unflatten_dense_tensors(coalesced, grads)):
            buf.copy_(synced)

    # def allreduce_first_last_embeddings(self):
    #
    #     # Modified from megatron-lm: https://github.com/NVIDIA/Megatron-LM/blob/d41696840ed0a7edb7e0499eb82a48ae112d9bb3/megatron/training.py#L407
    #     # All-reduce word_embeddings' grad across first and last stages to ensure
    #     # that word_embeddings parameters stay in sync.
    #     # This should only run for models that support pipelined model parallelism
    #     # (BERT and GPT-2).
    #     if parallel_state.get_pipeline_model_parallel_world_size() > 1 and (
    #         parallel_state.is_pipeline_first_stage(ignore_virtual=True)
    #         or parallel_state.is_pipeline_last_stage(ignore_virtual=True)
    #     ):
    #         if parallel_state.is_pipeline_first_stage(ignore_virtual=True):
    #             if isinstance(self.model, list):
    #                 module = self.model[0]  # only the first virtual rank has the embeddings
    #             else:
    #                 module = self.model
    #         if parallel_state.is_pipeline_last_stage(ignore_virtual=True):
    #             if isinstance(self.model, list):
    #                 module = self.model[-1]  # only the last virtual rank has the embeddings
    #             else:
    #                 module = self.model
    #         if module.share_token_embeddings:
    #             word_embeddings_weight = module.word_embeddings_weight()
    #             if self.megatron_amp_o2:
    #                 # O2 recipe stores a "main" copy of weights and grads
    #                 grad = word_embeddings_weight.main_grad
    #             else:
    #                 grad = word_embeddings_weight.grad
    #             torch.distributed.all_reduce(grad, group=parallel_state.get_embedding_group())

    def get_forward_output_and_loss_func(self):

        def loss_func(labels, output_tensor):
            logits = output_tensor.contiguous().float()
            loss = torch.nn.functional.cross_entropy(logits, labels)

            outputs = torch.argmax(logits, -1)
            correct = (outputs == labels).float()
            accuracy = torch.mean(correct)

            averaged_loss = average_losses_across_data_parallel_group([loss, accuracy])

            return loss, {"loss": averaged_loss[0], "accuracy": averaged_loss[1]}

        def fwd_output_and_loss_func(batch, model):
            if parallel_state.get_pipeline_model_parallel_world_size() == 1:
                batch = [x.cuda(non_blocking=True) for x in batch]
                tokens, labels = batch
            else:
                # Vision transformer doesn't need attention mask
                if parallel_state.is_pipeline_first_stage():
                    # Fist pipeline stage needs only the tokens and position_ids
                    tokens = batch[0].cuda(non_blocking=True)
                    labels = None
                elif parallel_state.is_pipeline_last_stage():
                    # Last pipeline stage needs only the labels and loss_mask
                    labels = batch[1].cuda(non_blocking=True)
                    tokens = None
                else:
                    # Intermediate pipeline stage doesn't need any inputs
                    tokens, labels = None, None

            output_tensor = model(tokens)
            return output_tensor, partial(loss_func, labels)

        return fwd_output_and_loss_func

    def get_forward_output_only_func(self):
        def fwd_output_only_func(batch, model):
            raise NotImplementedError

        return fwd_output_only_func

    def validation_step(self, batch, batch_idx):
        """
            Our dataloaders produce a micro-batch and then we fetch
            a number of microbatches depending on the global batch size and model parallel size
            from the dataloader to produce a list of microbatches.
            The list of microbatches is then piped through the pipeline using Apex fwd/bwd functions.
        """

        batch_for_pipeline = self.process_global_batch(batch, self.cfg.global_batch_size)
        tensor_shape = [self.cfg.encoder_seq_length, self.cfg.micro_batch_size, self.cfg.hidden_size]

        # run forward passes for an entire global batch
        # we do this inside validation_step to support pipeline parallelism
        fwd_bwd_function = self._get_fwd_bwd_function()

        losses_reduced_per_micro_batch = fwd_bwd_function(
            forward_step_func=self.get_forward_output_and_loss_func(),
            batch=batch_for_pipeline,
            model=self.model,
            forward_only=True,
            tensor_shape=tensor_shape,
            dtype=self.autocast_dtype,
            sequence_parallel_enabled=self.cfg.get('sequence_parallel', False),
        )

        def _get_metric_with_batch_size(metric_key):
            # only the last stage of the pipeline returns losses
            if losses_reduced_per_micro_batch:
                actual_batch_size = batch[0].shape[0]  # Might be lesser than global_batch_size if drop_last=False
                expected_batch_size = self.cfg.global_batch_size // parallel_state.get_data_parallel_world_size()
                if actual_batch_size == expected_batch_size:
                    loss_with_batch_size_list = [
                        [loss_reduced[metric_key].item(), self.cfg.micro_batch_size]
                        for loss_reduced in losses_reduced_per_micro_batch
                    ]
                else:
                    loss_with_batch_size_list = []
                    total_samples_remaining = actual_batch_size
                    for loss_reduced in losses_reduced_per_micro_batch:
                        if total_samples_remaining <= 0:
                            break
                        if total_samples_remaining // self.cfg.micro_batch_size >= 1:
                            loss_with_batch_size_list.append([loss_reduced[metric_key].item(), self.cfg.micro_batch_size])
                        else:
                            loss_with_batch_size_list.append([loss_reduced[metric_key].item(), total_samples_remaining])
                        total_samples_remaining = total_samples_remaining - self.cfg.micro_batch_size
            else:
                # we're not on the last pipeline stage so no losses
                loss_with_batch_size_list = []
            return loss_with_batch_size_list

        return _get_metric_with_batch_size('loss'), _get_metric_with_batch_size('accuracy')

    def validation_epoch_end(self, outputs):
        # TODO (yuya): need fix later, check with Sean
        if not outputs:
            return

        if parallel_state.is_pipeline_last_stage():
            loss_outputs = [output[0] for output in outputs]
            acc_outputs = [output[1] for output in outputs]

            def _get_average_metric(metric_outputs):
                # only the last pipeline parallel stages return metric with their batch size
                total_num_samples = 0
                total_metric = 0
                for metric_with_batch_size in metric_outputs:
                    metric_with_batch_size_array = np.array(metric_with_batch_size).flatten()
                    batch_metrices = metric_with_batch_size_array[0::2]
                    batch_sizes = metric_with_batch_size_array[1::2]
                    total_num_samples += sum(batch_sizes)
                    total_metric += np.dot(batch_metrices, batch_sizes)
    
                avg_metric = total_metric / total_num_samples
                return avg_metric

            averaged_metrics = torch.tensor(
                [_get_average_metric(loss_outputs), _get_average_metric(acc_outputs)],
                dtype=torch.float32).cuda()
        else:
            averaged_metrics = torch.tensor([0.0, 0.0], dtype=torch.float32).cuda()

        # we can only log on one rank if it is rank zero so we broadcast from last rank
        torch.distributed.broadcast(averaged_metrics, get_last_rank())

        averaged_loss, averaged_acc = averaged_metrics

        self.log('global_step', self.trainer.global_step, prog_bar=True, rank_zero_only=True)
        self.log('val_loss', averaged_loss, prog_bar=True, rank_zero_only=True)
        self.log('val_accuracy', averaged_acc, prog_bar=True, rank_zero_only=True)

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def test_epoch_end(self, outputs):
        averaged_loss = average_losses_across_data_parallel_group(outputs)
        logging.info(f'test_loss: {averaged_loss[0]}')

    def process_global_batch(self, global_batch, global_batch_size=None):
        """ Prepares the global batch for apex fwd/bwd functions.
            Global batch is a list of micro batches.
        """
        tokens = global_batch[0] # images
        labels = global_batch[1]

        expected_batch_size = None
        if global_batch_size is not None:
            expected_batch_size = global_batch_size // parallel_state.get_data_parallel_world_size()
        current_batch_size = tokens.shape[0]
        if expected_batch_size is not None and expected_batch_size > current_batch_size:
            logging.info(
                'Got batch size of '
                + str(current_batch_size)
                + ' , expected batch size :'
                + str(expected_batch_size)
                + '. Appending dummy data.'
            )
            pad_length = expected_batch_size - current_batch_size
            pad_dim = (int(pad_length), tokens.shape[1])
            tokens = torch.cat((tokens, torch.ones(pad_dim, dtype=tokens.dtype)))
            labels = torch.cat((labels, torch.ones(pad_dim, dtype=labels.dtype)))

        return [tokens, labels]

    def build_train_valid_test_datasets(self):
        logging.info('Building datasets for ViT...')
        if self.trainer.limit_val_batches > 1.0 and isinstance(self.trainer.limit_val_batches, float):
            raise ValueError("limit_val_batches must be an integer or float less than or equal to 1.0.")

        self._train_ds, self._validation_ds = build_train_valid_datasets(
            model_cfg=self.cfg,
            data_path=self.cfg.data.data_path,
            image_size=(self.cfg.img_h, self.cfg.img_w),
        )
        self._test_ds = None

        if self._train_ds is not None:
            logging.info(f'Length of train dataset: {len(self._train_ds)}')
        if self._validation_ds is not None:
            logging.info(f'Length of val dataset: {len(self._validation_ds)}')
        if self._test_ds is not None:
            logging.info(f'Length of test dataset: {len(self._test_ds)}')
        logging.info(f'Finished building datasets for ViT.')

        return self._train_ds, self._validation_ds, self._test_ds

    def build_pretraining_data_loader(self, dataset, consumed_samples, dataset_type=None, drop_last=True):
        """Buld dataloader given an input dataset."""

        logging.info(f'Building dataloader with consumed samples: {consumed_samples}')
        # Megatron sampler
        if hasattr(self.cfg.data, 'dataloader_type') and self.cfg.data.dataloader_type is not None:
            if self.cfg.data.dataloader_type == 'single':
                batch_sampler = MegatronPretrainingBatchSampler(
                    total_samples=len(dataset),
                    consumed_samples=consumed_samples,
                    micro_batch_size=self.cfg.micro_batch_size,
                    global_batch_size=self.cfg.global_batch_size,
                    data_parallel_rank=parallel_state.get_data_parallel_rank(),
                    data_parallel_size=parallel_state.get_data_parallel_world_size(),
                    drop_last=drop_last,
                )
            elif self.cfg.data.dataloader_type == 'cyclic':
                batch_sampler = MegatronVisionPretrainingRandomBatchSampler(
                    dataset=dataset,
                    total_samples=len(dataset),
                    consumed_samples=consumed_samples,
                    micro_batch_size=self.cfg.micro_batch_size,
                    global_batch_size=self.cfg.global_batch_size,
                    data_parallel_rank=parallel_state.get_data_parallel_rank(),
                    data_parallel_size=parallel_state.get_data_parallel_world_size(),
                    drop_last=drop_last,
                    data_sharding=self.cfg.data.get("data_sharding", True),
                )
            else:
                raise ValueError('cfg.data.dataloader_type must be "single" or "cyclic"')
        else:
            raise ValueError('cfg.data.dataloader_type not found. Must be "single" or "cyclic"')

        return torch.utils.data.DataLoader(
            dataset, batch_sampler=batch_sampler, num_workers=self.cfg.data.num_workers, pin_memory=True,
        )

    def setup(self, stage=None):
        """ PTL hook that is executed after DDP spawns.
            We setup datasets here as megatron datasets require DDP to instantiate.
            See https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#setup for more information.
        Args:
            stage (str, optional): Can be 'fit', 'validate', 'test' or 'predict'. Defaults to None.
        """

        # log number of parameters
        if isinstance(self.model, list):
            num_parameters_on_device = sum(
                [sum([p.nelement() for p in model_module.parameters()]) for model_module in self.model]
            )
            # if parallel_state.get_pipeline_model_parallel_world_size() > 1 and parallel_state.is_pipeline_last_stage(
            #     ignore_virtual=True
            # ):
            #     # substract the embedding weights on the last virtual stage
            #     num_word_embedding_parameters = sum([p.nelement() for p in self.model[-1].word_embeddings_weight()])
            #     num_parameters_on_device -= num_word_embedding_parameters
        else:
            num_parameters_on_device = sum([p.nelement() for p in self.model.parameters()])

            # if parallel_state.get_pipeline_model_parallel_world_size() > 1 and parallel_state.is_pipeline_last_stage(
            #     ignore_virtual=True
            # ):
            #     # substract the embedding weights on the last stage
            #     num_word_embedding_parameters = sum([p.nelement() for p in self.model.word_embeddings_weight()])
            #
            #     num_parameters_on_device -= num_word_embedding_parameters

        # to be summed across data parallel group
        total_num_parameters = torch.tensor(num_parameters_on_device).cuda()

        torch.distributed.all_reduce(total_num_parameters, group=parallel_state.get_model_parallel_group())

        logging.info(
            f'Pipeline model parallel rank: {parallel_state.get_pipeline_model_parallel_rank()}, '
            f'Tensor model parallel rank: {parallel_state.get_tensor_model_parallel_rank()}, '
            f'Number of model parameters on device: {num_parameters_on_device:.2e}. '
            f'Total number of model parameters: {total_num_parameters:.2e}.'
        )

        resume_checkpoint_path = self.trainer._checkpoint_connector.resume_from_checkpoint_fit_path
        if resume_checkpoint_path:
            init_consumed_samples = self._extract_consumed_samples_from_ckpt(resume_checkpoint_path)
        else:
            init_consumed_samples = 0
        self.init_consumed_samples = init_consumed_samples
        self.init_global_step = self.trainer.global_step

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
                    # module.sync_initial_word_embeddings()
                parallel_state.set_virtual_pipeline_model_parallel_rank(0)
            else:
                # self.model.sync_initial_word_embeddings()
                pass
    def setup_training_data(self, cfg):
        if hasattr(self, '_train_ds') and self._train_ds is not None:
            consumed_samples = self.compute_consumed_samples(0)
            logging.info(
                f'Setting up train dataloader with len(len(self._train_ds)): {len(self._train_ds)} and consumed samples: {consumed_samples}'
            )
            self._train_dl = self.build_pretraining_data_loader(self._train_ds, consumed_samples)

    def setup_validation_data(self, cfg):
        if hasattr(self, '_validation_ds') and self._validation_ds is not None:
            consumed_samples = 0
            logging.info(
                f'Setting up validation dataloader with len(len(self._validation_ds)): {len(self._validation_ds)} and consumed samples: {consumed_samples}'
            )
            drop_last = True
            if not self.cfg.data.get('validation_drop_last', True):
                logging.info(f'Drop last in validation dataset is set to False')
                drop_last = False
            self._validation_dl = self.build_pretraining_data_loader(
                self._validation_ds, consumed_samples, "validation", drop_last
            )

    def setup_test_data(self, cfg):
        if hasattr(self, '_test_ds') and self._test_ds is not None:
            consumed_samples = 0
            logging.info(
                f'Setting up test dataloader with len(len(self._test_ds)): {len(self._test_ds)} and consumed samples: {consumed_samples}'
            )
            self._test_dl = self.build_pretraining_data_loader(self._test_ds, consumed_samples)

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: Optional[int] = None) -> Any:
        raise NotImplementedError

    def transfer_batch_to_device(self, batch: Any, device: torch.device, dataloader_idx: int) -> Any:
        """ PTL hook: https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#transfer-batch-to-device
            When using pipeline parallelism, we need the global batch to remain on the CPU,
            since the memory overhead will be too high when using a large number of microbatches.
            Microbatches are transferred from CPU to GPU inside the pipeline.
        """
        return batch

    def _validate_trainer(self):
        """ Certain trainer configurations can break training.
            Here we try to catch them and raise an error.
        """
        if self.trainer.accumulate_grad_batches > 1:
            raise ValueError(
                f'Gradient accumulation is done within training_step. trainer.accumulate_grad_batches must equal 1'
            )

    @classmethod
    def list_available_models(cls) -> Optional[PretrainedModelInfo]:
        return None

    def on_save_checkpoint(self, checkpoint) -> None:
        """LightningModule hook:
        https://pytorch-lightning.readthedocs.io/en/stable/common/lightning_module.html#on-save-checkpoint
        """
        if isinstance(self.model, list):
            for i in range(len(self.model)):
                parallel_state.set_virtual_pipeline_model_parallel_rank(i)
                checkpoint[f'model{i}'] = self.model[i].module.state_dict_for_save_checkpoint()
            parallel_state.set_virtual_pipeline_model_parallel_rank(0)

    def on_load_checkpoint(self, checkpoint) -> None:
        """LightningModule hook:
        https://pytorch-lightning.readthedocs.io/en/stable/common/lightning_module.html#on-load-checkpoint
        """
        if isinstance(self.model, list):
            for i in range(len(self.model)):
                parallel_state.set_virtual_pipeline_model_parallel_rank(i)
                self.model[i].module.load_state_dict(checkpoint[f'model{i}'], strict=True)
            parallel_state.set_virtual_pipeline_model_parallel_rank(0)

    def parameters(self):
        if isinstance(self.model, list):
            return itertools.chain.from_iterable(module.parameters() for module in self.model)
        else:
            return self.model.parameters()