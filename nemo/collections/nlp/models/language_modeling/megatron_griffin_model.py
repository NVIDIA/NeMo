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
import queue
import warnings
from typing import Any, Dict, Iterator, List, Optional
from megatron.core.models.griffin.griffin_model import GriffinModel
from pkg_resources import packaging
from importlib.metadata import version
import torch
import torch.nn.functional as F
from omegaconf.dictconfig import DictConfig
from pytorch_lightning.trainer.trainer import Trainer

from nemo.collections.nlp.data.language_modeling.megatron import dataset_utils
from nemo.collections.nlp.data.language_modeling.megatron.data_samplers import (
    MegatronPretrainingRandomSampler,
    MegatronPretrainingSampler,
)
from nemo.collections.nlp.models.language_modeling.megatron_base_model import MegatronBaseModel
from nemo.collections.nlp.modules.common.megatron.build_model import build_model
from nemo.collections.nlp.modules.common.megatron.module import Float16Module
from nemo.collections.nlp.modules.common.megatron.utils import (
    ApexGuardDefaults,
    average_losses_across_data_parallel_group,
    get_params_for_weight_decay_optimization,
)
from nemo.collections.nlp.parts.utils_funcs import get_last_rank
from nemo.core.classes.common import PretrainedModelInfo
from nemo.core.neural_types import ChannelType, MaskType, NeuralType
from nemo.utils import logging

try:
    from apex.transformer.pipeline_parallel.utils import get_num_microbatches

    HAVE_APEX = True

except (ImportError, ModuleNotFoundError):

    HAVE_APEX = False

try:
    import logging

    HAVE_LDDL = True
except (ImportError, ModuleNotFoundError):
    HAVE_LDDL = False

try:
    from megatron.core import parallel_state
    from megatron.core.pipeline_parallel.schedules import get_forward_backward_func
    from megatron.core.transformer.module import Float16Module as MCoreFloat16Module
    from megatron.core.transformer.transformer_config import TransformerConfig

    HAVE_MEGATRON_CORE = True

except (ImportError, ModuleNotFoundError):
    TransformerConfig = ApexGuardDefaults
    HAVE_MEGATRON_CORE = False

try:
    from transformer_engine.pytorch import module as te_module

    HAVE_TE = True

except (ImportError, ModuleNotFoundError):
    HAVE_TE = False

class MegatronGriffinModel(MegatronBaseModel):
    """
    Megatron Griffin pretraining.
    """

    def __init__(self, cfg: DictConfig, trainer: Trainer):
        if not HAVE_MEGATRON_CORE:
            raise ImportError(
                "megatron-core was not found. Please see the NeMo README for installation instructions: https://github.com/NVIDIA/NeMo#megatron-gpt."
            )
        self.megatron_amp_O2 = cfg.get('megatron_amp_O2', False)
        self.cfg = cfg
        self.mcore_gpt=True
        if not self.megatron_amp_O2 and self.cfg.get('virtual_pipeline_model_parallel_size', None):
            raise ValueError('Virtual pipeline model parallel is only supported when using megatron_amp_O2')

        super().__init__(cfg, trainer=trainer, no_lm_init=False)

        self._validate_trainer()

        self.transformer_config = self.build_transformer_config()
        self.transformer_config.gated_linear_unit = True

        self.vocab_size = cfg.get('vocab_size', 256_128)

        self.get_attention_mask_from_fusion = self.cfg.get('get_attention_mask_from_fusion', False)
        self.rampup_batch_size = self.cfg.get('rampup_batch_size', None)
        if self.rampup_batch_size:
            self.prev_consumed_samples = 0
            self.if_first_step = 0
            self.prev_global_batch_size = None

        self.enable_autocast = (
            True if (not self.megatron_amp_O2) and (self.autocast_dtype in [torch.float16, torch.bfloat16]) else False
        )

        # used in NVIDIA NGC PyTorch containers
        # buffer used during train_step for logging average loss over gradient accumulation steps
        self._reduced_lm_loss_buffer = []
        self._reduced_sop_loss_buffer = []

        # build_model returns a list of modules which are used for interleaved pipeline parallelism
        self.model = build_model(
            model_provider_func=self.model_provider_func,
            wrap_with_ddp=False,
            virtual_pipeline_model_parallel_size=self.cfg.get('virtual_pipeline_model_parallel_size', None),
        )

        # if we're not using interleaved, then self.model is a module.
        if self.cfg.get('virtual_pipeline_model_parallel_size', None) is None:
            self.model = self.model[0]

        if self.megatron_amp_O2:

            if not self.with_distributed_adam:
                # Pre-allocate the model on GPU to have master parameters allocated on the same device with matching data type
                if isinstance(self.model, list):
                    for module in self.model:
                        module.cuda(torch.cuda.current_device())
                else:
                    self.model.cuda(torch.cuda.current_device())

            # Model wrapper to convert both model and inputs to half precision
            self._wrap_model_for_O2()

        if hasattr(self, '_nsys_profile_enabled'):
            mp_size = cfg.get('tensor_model_parallel_size', 1) * cfg.get('pipeline_model_parallel_size', 1)
            data_parallel_world_size = trainer.world_size // mp_size
            grad_accum_steps = cfg.get('global_batch_size') // (cfg.get('micro_batch_size') * data_parallel_world_size)
            self._nsys_profile_start_step *= grad_accum_steps
            self._nsys_profile_end_step *= grad_accum_steps
        self.initialize_ub = self.cfg.get('ub_tp_comm_overlap', False)

    def model_provider_func(self, pre_process, post_process):
        
        model = GriffinModel(
            config=self.transformer_config,
            vocab_size=self.vocab_size,
            )
    
        return model

    def _validate_trainer(self):
        """ Certain trainer configurations can break training.
            Here we try to catch them and raise an error.
        """
        if self.trainer.accumulate_grad_batches > 1:
            raise ValueError(
                f'Gradient accumulation is done within training_step. trainer.accumulate_grad_batches must equal 1'
            )

    def forward(
        self,
        input_ids,
        attention_mask,
    ):
            
        output_tensor = self.model(
            input_ids,
            attention_mask,
        )
        return output_tensor
    

    def _make_data_iterator_list(self, data_iterator: Iterator) -> List[Iterator]:
        """ Convert data iterator into form expected by Megatron

            With interleaved pipeline parallelism, Megatron expects a
            list of one data iterator per model chunk. Each model
            chunk independently gets data from its data iterator, so
            we need to interact with the data iterator multiple times
            for each microbatch step. Instead of incorporating this
            logic into the data loader, we cache the iterator's output
            to the first model chunk and reuse it in the other model
            chunks.
        """

        if not isinstance(self.model, list) or len(self.model) == 1:
            return data_iterator  # TODO @tmoon: Remove
            # TODO @tmoon: Use once available in Megatron-LM
            # return DataIteratorList([data_iterator])

        class CachingIterator:
            """Iterator wrapper that caches values"""

            class Proxy:
                """Returns values from caching iterator wrapper

                Assumed to never advance past the caching iterator.
                """

                def __init__(self):
                    self.cache = queue.Queue()

                def __iter__(self):
                    return self

                def __next__(self):
                    return self.cache.get_nowait()

            def __init__(self, iterator: Iterator):
                self.iterator = iterator
                self.proxies = []

            def make_proxy(self):
                self.proxies.append(CachingIterator.Proxy())
                return self.proxies[-1]

            def __iter__(self):
                return self

            def __next__(self):
                val = next(self.iterator)
                for proxy in self.proxies:
                    proxy.cache.put(val)
                return val

        # Make list of iterator wrappers
        iters = [CachingIterator(data_iterator)]
        while len(iters) < len(self.model):
            iters.append(iters[0].make_proxy())
        return iters  # TODO @tmoon: Remove
        # TODO @tmoon: Use once available in Megatron-LM
        # return DataIteratorList(iters)

    def get_batch(self, data_iterator, tuning):
        """Generate a batch."""

        # Broadcast data.
        if data_iterator is not None:
            # If tuple, 1st element in it is the batch since dataloader_iter returns batch, batch_idx, dataloader_idx
            data = next(data_iterator)
            if isinstance(data, tuple):
                data = data[0]
        else:
            data = None

        # return batch for GPT SFT
        if tuning:
            return data

        batch = {
            'tokens': data["tokens"],
            'labels': data["labels"],
            'loss_mask': data["loss_mask"],
            'attention_mask': data["attention_mask"],
            'position_ids': data["position_ids"],
        }

        return batch
    
    def initialize_ub_func(self):
        ub_cfgs = self.cfg.get('ub_tp_comm_overlap_cfg', None)
        if ub_cfgs is None:
            warnings.warn(
                "Couldn't find TP config. Please check the path correctness. Initializing TP comm overlap with the default config."
            )

        input_shape = [
            self.cfg.get('encoder_seq_length')
            * self.cfg.get('micro_batch_size')
            // self.cfg.get('context_parallel_size', 1),
            self.cfg.get('hidden_size'),
        ]

        te_module.base.initialize_ub(
            shape=input_shape,
            tp_size=self.cfg.get('tensor_model_parallel_size'),
            use_fp8=self.cfg.get('fp8'),
            ub_cfgs=ub_cfgs,
        )
        self.initialize_ub = False

    def training_step_fwd_bwd_step_call(self, dataloader_iter, forward_only):
        """
        This method is called from the training_step method.
        It is separated out to allow for overriding in the MegatronGPTEmbeddingModel
        """
        loss_mean = self.fwd_bwd_step(dataloader_iter, forward_only)
        return loss_mean


    def training_step(self, dataloader_iter):
        """
            We pass the dataloader iterator function to the micro-batch scheduler.
            The input batch to each micro-batch is fetched using the dataloader function
            in the micro-batch fwd function.
        """
        # Initialize userbuffer communicators.
        if self.initialize_ub:
            self.initialize_ub_func()

        if self.rampup_batch_size:
            num_microbatch_calculator = apex.transformer.pipeline_parallel.utils._GLOBAL_NUM_MICROBATCHES_CALCULATOR
            current_global_batch_size = num_microbatch_calculator.current_global_batch_size
            # do validation and save the checkpoint when gbs is changed
            if self.prev_global_batch_size != current_global_batch_size and self.prev_global_batch_size:
                self.trainer.should_stop = True

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
                if isinstance(module, (Float16Module, MCoreFloat16Module)):
                    module = module.module
                if not self.mcore_gpt:
                    module = module.language_model
                if hasattr(module, 'embedding'):
                    for param in module.embedding.parameters():
                        param.data_ptr()

        loss_mean = self.training_step_fwd_bwd_step_call(dataloader_iter, forward_only=False)

        if self.cfg.get('fp8', False):
            self.prev_step_training = self.training

        # when using sequence parallelism, the sequence parallel layernorm grads must be all-reduced
        if self.cfg.get('tensor_model_parallel_size', 1) > 1 and self.cfg.get('sequence_parallel', False):
            self.megatron_timer_start('allreduce_sequence_parallel_gradients', log_level=1)
            self.allreduce_sequence_parallel_gradients()
            self.megatron_timer_stop('allreduce_sequence_parallel_gradients')

        self.megatron_timer_start('gradient_allreduce', log_level=1)
        if self.use_fsdp:
            # Reduce the gradients omitted from FSDP-sharding
            self.allreduce_fsdp_sharding_omitted_gradients()
        elif self.with_distributed_adam:
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
        self.megatron_timer_stop('gradient_allreduce')

        if self.cfg.get('pipeline_model_parallel_size', 1) > 1 and self.cfg.get(
            'share_embeddings_and_output_weights', True
        ):
            self.megatron_timer_start('allreduce_first_last_embeddings', log_level=1)
            # when using pipeline parallelism the first and last stage must keep embeddings in sync
            self.allreduce_first_last_embeddings()
            self.megatron_timer_stop('allreduce_first_last_embeddings')

        if self.log_memory_usage:
            mem_reserved = torch.cuda.max_memory_reserved()
            self.log(
                'peak_memory_usage', mem_reserved, prog_bar=True, rank_zero_only=True, batch_size=1,
            )

        ## logging
        if self.log_train_loss:
            # When using pipeline parallelism, loss is calculated only in the last pipeline stage and
            # it should be casted to other pipeline stages for logging.
            # we can avoid this broadcast by updating the PTL log function to accept specific ranks
            if parallel_state.get_pipeline_model_parallel_world_size() > 1:
                if torch.distributed.get_rank() == get_last_rank():
                    torch.distributed.send(loss_mean, 0)
                elif torch.distributed.get_rank() == 0:
                    torch.distributed.recv(loss_mean, get_last_rank())
            self.log('reduced_train_loss', loss_mean, prog_bar=True, rank_zero_only=True, batch_size=1)

            # (@adithyare) we need to check for the _scaler attribute to enable pp>1 for adapter training
            if self.cfg.precision == 16 and hasattr(self.trainer.precision_plugin.scaler, "_scale"):
                loss_scale = self.trainer.precision_plugin.scaler._scale
                if loss_scale is not None:
                    self.log('loss_scale', loss_scale, batch_size=1)

        lr = self._optimizer.param_groups[0]['lr']
        self.log('lr', lr, rank_zero_only=True, batch_size=1)
        self.log(
            'global_step', self.trainer.global_step, prog_bar=True, rank_zero_only=True, batch_size=1,
        )

        consumed_samples = self._compute_consumed_samples_after_training_step()
        # TODO: make sure compute_consumed_samples works for pipeline parallelism
        self.log(
            'consumed_samples', consumed_samples, prog_bar=True, rank_zero_only=True, batch_size=1,
        )

        if self.rampup_batch_size:
            self.prev_global_batch_size = current_global_batch_size
            self.prev_consumed_samples = consumed_samples
            num_microbatch_calculator.update(
                consumed_samples=consumed_samples, consistency_check=False,
            )
            current_global_batch_size = num_microbatch_calculator.current_global_batch_size
            self.log('global_batch_size', current_global_batch_size, prog_bar=True, rank_zero_only=True, batch_size=1)
            self.if_first_step = 1

        return loss_mean

    def backward(self, *args, **kwargs):
        """ LightningModule hook to do backward.
            We want this to do nothing since we run backward in the fwd/bwd functions from megatron-core.
            No need to call it here.
        """
        return

    def optimizer_zero_grad(self, *args, **kwargs):
        """ LightningModule hook to zero grad.
            We want this to do nothing as we are zeroing grads during the training_step.
        """
        return

    def get_batch_on_this_context_parallel_rank(self, batch):
        num_valid_tokens_in_ub = None
        if 'loss_mask' in batch and batch['loss_mask'] is not None:
            num_valid_tokens_in_ub = batch['loss_mask'].sum()

        cp_size = parallel_state.get_context_parallel_world_size()
        if cp_size > 1:
            cp_rank = parallel_state.get_context_parallel_rank()
            for key, val in batch.items():
                if val is not None:
                    seq_dim = 1 if key != 'attention_mask' else 2
                    val = val.view(
                        *val.shape[0:seq_dim],
                        2 * cp_size,
                        val.shape[seq_dim] // (2 * cp_size),
                        *val.shape[(seq_dim + 1) :],
                    )
                    index = torch.tensor([cp_rank, (2 * cp_size - cp_rank - 1)], device="cpu", pin_memory=True).cuda(
                        non_blocking=True
                    )
                    val = val.index_select(seq_dim, index)
                    val = val.view(*val.shape[0:seq_dim], -1, *val.shape[(seq_dim + 2) :])
                    batch[key] = val

        batch['num_valid_tokens_in_ub'] = num_valid_tokens_in_ub

        return batch
    
    def get_forward_output_and_loss_func(self, validation_step=False, tuning=False):
        def fwd_output_and_loss_func(dataloader_iter, model, checkpoint_activations_all_layers=None):

            # Get data batch
            batch = self.get_batch(dataloader_iter, tuning)
            # Transfer needed data to GPU
            required_keys = set()
            max_seqlen = batch['max_seqlen'].squeeze() if 'max_seqlen' in batch else None
            cu_seqlens_argmin = batch['cu_seqlens_argmin'] if 'cu_seqlens_argmin' in batch else None
            if parallel_state.get_pipeline_model_parallel_world_size() == 1:
                required_keys.update(batch.keys())
            else:
                required_keys.add('attention_mask')
                if 'cu_seqlens' in batch:
                    required_keys.add('cu_seqlens')
                if parallel_state.is_pipeline_first_stage():
                    required_keys.update(('tokens', 'position_ids'))
                if parallel_state.is_pipeline_last_stage():
                    required_keys.update(('labels', 'loss_mask'))
            if self.get_attention_mask_from_fusion and 'attention_mask' in required_keys:
                required_keys.remove('attention_mask')
            batch = {key: val.cuda(non_blocking=True) if key in required_keys else None for key, val in batch.items()}

            # slice batch along sequence dimension for context parallelism
            batch = self.get_batch_on_this_context_parallel_rank(batch)

            # Model forward pass
            forward_args = {
                'input_ids': batch['tokens'],
                'position_ids': batch['position_ids'],
                'attention_mask': None if self.get_attention_mask_from_fusion else batch['attention_mask'],
                'labels': batch['labels'] if 'labels' in batch else None,
                'loss_mask': batch['loss_mask'],
            }

            if not self.mcore_gpt:
                forward_args['checkpoint_activations_all_layers'] = checkpoint_activations_all_layers
                if not self.use_loss_mask:
                    forward_args.pop('loss_mask')
            else:
                # TODO: @eharper can we add this to mcore?
                forward_args.pop('loss_mask')

                if 'cu_seqlens' in batch:  # packed sequence from GPTSFTPackedDataset
                    # these args are passed eventually into TEDotProductAttention.forward()
                    cu_seqlens = batch['cu_seqlens'].squeeze()  # remove batch size dimension (mbs=1)
                    # remove -1 "paddings" added in collate_fn
                    if cu_seqlens_argmin is not None:
                        cu_seqlens = cu_seqlens[: cu_seqlens_argmin.item()]
                    else:
                        cu_seqlens = cu_seqlens[: torch.argmin(cu_seqlens)]

                    try:
                        from megatron.core.packed_seq_params import PackedSeqParams
                    except (ImportError, ModuleNotFoundError) as e:
                        mcore_version = packaging.version.Version(version('megatron-core'))
                        logging.error(
                            f"megatron-core v{mcore_version} does not support training with packed sequence. "
                            "Please use megatron-core >= 0.5.0, or set model.data.train_ds.packed_sequence=False"
                        )
                        raise e

                    forward_args['packed_seq_params'] = PackedSeqParams(
                        cu_seqlens_q=cu_seqlens,
                        cu_seqlens_kv=cu_seqlens,
                        max_seqlen_q=max_seqlen,
                        max_seqlen_kv=max_seqlen,
                        qkv_format='thd',
                    )
            print(forward_args['input_ids'].shape)
            print(forward_args['attention_mask'].shape)
            # output_tensor = model(**forward_args)
            output_tensor = model(forward_args['input_ids'], forward_args['attention_mask'])
            print(output_tensor.shape)
            import sys
            sys.exit()

            def loss_func(output_tensor):
                # Loss for a micro-batch (ub)
                loss_for_ub = self.loss_func(batch['loss_mask'], batch['num_valid_tokens_in_ub'], output_tensor)
                cp_size = self.cfg.get('context_parallel_size', 1)
                if self.cfg.data.get(
                    "return_output_tensors", False
                ):  # TODO: need a better way to check if loss_func is returning more stuff than just loss... (@adithyare)
                    loss_for_ub, q_hs, d_hs, pos_cs, neg_cs, diff_cs = loss_for_ub
                    reduced_loss = average_losses_across_data_parallel_group([loss_for_ub])
                    pos_cs = average_losses_across_data_parallel_group([pos_cs])
                    neg_cs = average_losses_across_data_parallel_group([neg_cs])
                    diff_cs = average_losses_across_data_parallel_group([diff_cs])
                    return (
                        loss_for_ub * cp_size,
                        {
                            'avg': reduced_loss,
                            'query_hs': q_hs,
                            'doc_hs': d_hs,
                            'avg_pos_cs': pos_cs,
                            'avg_neg_cs': neg_cs,
                            'diff_cs': diff_cs,
                        },
                    )
                elif validation_step and not self.cfg.data.get('validation_drop_last', True):
                    num_valid_tokens_in_ub = batch['num_valid_tokens_in_ub']
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
                    return loss_for_ub * cp_size, {'loss_sum_and_ub_size': loss_sum_and_ub_size_all_gpu}
                else:
                    reduced_loss = average_losses_across_data_parallel_group([loss_for_ub])
                    return loss_for_ub * cp_size, {'avg': reduced_loss}

            return output_tensor, loss_func

        return fwd_output_and_loss_func

    def build_train_valid_test_datasets(self):
        # Override limit_val_batches to be a multiple of num microbatches to prevent val_step from exiting in between a step
        self._reconfigure_val_batches()
        logging.info('Building Bert datasets.')
        if self.trainer.limit_val_batches > 1.0 and isinstance(self.trainer.limit_val_batches, float):
            raise ValueError("limit_val_batches must be an integer or float less than or equal to 1.0.")
        global_batch_size = self.cfg.global_batch_size
        # Compute trianing micro-batch steps: total_global_batch_steps x grad_acumms_per_global_batch
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

        self._train_ds, self._validation_ds, self._test_ds = dataset_utils.build_train_valid_test_datasets(
            cfg=self.cfg,
            trainer=self.trainer,
            data_prefix=self.cfg.data.data_prefix,
            data_impl=self.cfg.data.data_impl,
            splits_string=self.cfg.data.splits_string,
            train_valid_test_num_samples=train_valid_test_num_samples,
            max_seq_length=self.cfg.data.seq_length,
            masked_lm_prob=self.cfg.data.masked_lm_prob,
            short_seq_prob=self.cfg.data.short_seq_prob,
            seed=self.cfg.seed,
            skip_warmup=self.cfg.data.get('skip_warmup', True),
            binary_head=self.cfg.bert_binary_head,
            max_seq_length_dec=None,
            dataset_type='standard_bert',
            tokenizer=self.tokenizer.tokenizer,
        )

        if self._train_ds is not None:
            logging.info(f'Length of train dataset: {len(self._train_ds)}')
        if self._validation_ds is not None:
            logging.info(f'Length of val dataset: {len(self._validation_ds)}')
        if self._test_ds is not None:
            logging.info(f'Length of test dataset: {len(self._test_ds)}')
        logging.info(f'Finished building Bert datasets.')
        return self._train_ds, self._validation_ds, self._test_ds

    def backward(self, *args, **kwargs):
        """ LightningModule hook to do backward.
            We want this to do nothing since we run backward in the fwd/bwd functions from megatron-core.
            No need to call it here.
        """
        return

    def optimizer_zero_grad(self, *args, **kwargs):
        """ LightningModule hook to zero grad.
            We want this to do nothing as we are zeroing grads during the training_step.
        """
        return

    def _append_sequence_parallel_module_grads(self, module, grads):
        """ Helper method for allreduce_sequence_parallel_gradients"""

        for param in module.parameters():
            sequence_parallel_param = getattr(param, 'sequence_parallel', False)
            if sequence_parallel_param:
                if self.megatron_amp_O2:
                    grad = param.main_grad
                else:
                    grad = param.grad
                grads.append(grad.data)

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

        if stage == 'predict':
            return
        else:
            # TODO: consider adding a ModelPT guard to check if model is being restored.
            # allowing restored models to optionally setup datasets
            if self.cfg.data.dataloader_type == "LDDL":
                self.build_LDDL_data(self.cfg.data)
                torch.distributed.barrier()
            else:
                self.build_train_valid_test_datasets()
                self.setup_training_data(self.cfg.data)
                self.setup_validation_data(self.cfg.data)
                self.setup_test_data(self.cfg.data)

        # when using pipeline model parallel the final stage need to initialize word embeddings
        if parallel_state.get_pipeline_model_parallel_world_size() > 1:
            for index, module in enumerate(self.get_model_module_list()):
                if parallel_state.get_virtual_pipeline_model_parallel_world_size() is not None:
                    parallel_state.set_virtual_pipeline_model_parallel_rank(index)
                    sync_embeddings = (
                        module.sync_initial_word_embeddings
                    )
                    sync_embeddings()
                    if parallel_state.get_virtual_pipeline_model_parallel_world_size() is not None:
                        parallel_state.set_virtual_pipeline_model_parallel_rank(0)

        self.setup_transformer_engine_tp_groups()


    def build_pretraining_data_loader(self, dataset, consumed_samples):
        """Buld dataloader given an input dataset."""

        if dataset is None:
            return None
        # Megatron sampler
        if hasattr(self.cfg.data, 'dataloader_type') and self.cfg.data.dataloader_type is not None:
            if self.cfg.data.dataloader_type == 'single':
                batch_sampler = MegatronPretrainingSampler(
                    total_samples=len(dataset),
                    consumed_samples=consumed_samples,
                    micro_batch_size=self.cfg.micro_batch_size,
                    global_batch_size=self.cfg.global_batch_size,
                    data_parallel_rank=parallel_state.get_data_parallel_rank(),
                    data_parallel_size=parallel_state.get_data_parallel_world_size(),
                    drop_last=self.cfg.get('drop_last', True),
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

        # Torch dataloader.
        return torch.utils.data.DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            num_workers=self.cfg.data.num_workers,
            pin_memory=True,
            persistent_workers=True if self.cfg.data.num_workers > 0 else False,
        )

    def setup_training_data(self, cfg):
        if hasattr(self, '_train_ds'):
            consumed_samples = self.compute_consumed_samples(0)
            logging.info(
                f'Setting up train dataloader with len(len(self._train_ds)): {len(self._train_ds)} and consumed samples: {consumed_samples}'
            )
            self._train_dl = self.build_pretraining_data_loader(self._train_ds, consumed_samples)

    def setup_validation_data(self, cfg):
        if hasattr(self, '_validation_ds'):
            consumed_samples = 0
            logging.info(
                f'Setting up validation dataloader with len(len(self._validation_ds)): {len(self._validation_ds)} and consumed samples: {consumed_samples}'
            )
            self._validation_dl = self.build_pretraining_data_loader(self._validation_ds, consumed_samples)

    def setup_test_data(self, cfg):
        if hasattr(self, '_test_ds'):
            consumed_samples = 0
            logging.info(
                f'Setting up test dataloader with len(len(self._test_ds)): {len(self._test_ds)} and consumed samples: {consumed_samples}'
            )
            self._test_dl = self.build_pretraining_data_loader(self._test_ds, consumed_samples)

    def transfer_batch_to_device(self, batch: Any, device: torch.device, dataloader_idx: int) -> Any:
        """ PTL hook: https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#transfer-batch-to-device
            When using pipeline parallelism, we need the global batch to remain on the CPU,
            since the memory overhead will be too high when using a large number of microbatches.
            Microbatches are transferred from CPU to GPU inside the pipeline.
        """
        return batch

    def parameters(self):
        if isinstance(self.model, list):
            return itertools.chain.from_iterable(module.parameters() for module in self.model)
        else:
            return self.model.parameters()

    @classmethod
    def list_available_models(cls) -> Optional[PretrainedModelInfo]:
        """
        This method returns a list of pre-trained model which can be instantiated directly from NVIDIA's NGC cloud.
        Returns:
            List of available pre-trained models.
        """
        result = []
        for vocab in ['cased', 'uncased']:
            result.append(
                PretrainedModelInfo(
                    pretrained_model_name=f"megatron_bert_345m_{vocab}",
                    location=f"https://api.ngc.nvidia.com/v2/models/nvidia/nemo/megatron_bert_345m_{vocab}/versions/1/files/megatron_bert_345m_{vocab}.nemo",
                    description=f"345M parameter BERT Megatron model with {vocab} vocab.",
                )
            )
        for vocab_size in ['50k', '30k']:
            for vocab in ['cased', 'uncased']:
                result.append(
                    PretrainedModelInfo(
                        pretrained_model_name=f"biomegatron345m_biovocab_{vocab_size}_{vocab}",
                        location=f"https://api.ngc.nvidia.com/v2/models/nvidia/nemo/biomegatron345m_biovocab_{vocab_size}_{vocab}/versions/1/files/BioMegatron345m-biovocab-{vocab_size}-{vocab}.nemo",
                        description="Megatron 345m parameters model with biomedical vocabulary ({vocab_size} size) {vocab}, pre-trained on PubMed biomedical text corpus.",
                    )
                )
        for vocab in ['cased', 'uncased']:
            result.append(
                PretrainedModelInfo(
                    pretrained_model_name=f"biomegatron-bert-345m-{vocab}",
                    location=f"https://api.ngc.nvidia.com/v2/models/nvidia/nemo/biomegatron345m{vocab}/versions/1/files/BioMegatron345m{vocab.capitalize()}.nemo",
                    description=f"Megatron pretrained on {vocab} biomedical dataset PubMed with 345 million parameters.",
                )
            )
        return result

    def setup_optimizer_param_groups(self):
        """ModelPT override. Optimizer will get self._optimizer_param_groups"""
        self._optimizer_param_groups = get_params_for_weight_decay_optimization(self.model)

    def configure_optimizers(self):

        if self.with_distributed_adam:

            # Disable overlapped grad sync for embedding grad when
            # pipeline parallelism is enabled
            if parallel_state.get_pipeline_model_parallel_world_size() > 1:
                modules = self.get_model_module_list()
                if parallel_state.is_pipeline_first_stage(ignore_virtual=True):
                    module = modules[0]  # only the first virtual rank has the embeddings
                    if self.cfg.get('share_embeddings_and_output_weights', True):
                        param = (
                             module.word_embeddings_weight()
                        )
                        param._disable_greedy_grad_copy = not self.megatron_amp_O2
                        param._disable_overlap_grad_sync = True
                if parallel_state.is_pipeline_last_stage(ignore_virtual=True):
                    if len(modules) > 1:
                        module = modules[-1]  # only the last virtual rank has the embeddings
                    else:
                        module = modules[0]
                    if self.cfg.get('share_embeddings_and_output_weights', True):
                        param = (
                             module.word_embeddings_weight()
                        )
                        param._disable_greedy_grad_copy = not self.megatron_amp_O2
                        param._disable_overlap_grad_sync = True

            # Disable overlapped grad sync for layer norm grads when
            # sequence parallelism is enabled
            for param in self.parameters():
                if getattr(param, 'sequence_parallel', False):
                    param._disable_greedy_grad_copy = not self.megatron_amp_O2
                    param._disable_overlap_grad_sync = True

            # sequence parallelism is enabled
            for param in self.parameters():
                if getattr(param, 'sequence_parallel', False):
                    param._disable_greedy_grad_copy = not self.megatron_amp_O2
                    param._disable_overlap_grad_sync = True

            # Initialize parameter buckets for overlapped grad and param syncs
            # Note: Params with disabled overlapping are put in the
            # last param bucket
            buckets = []
            if self.cfg.get('virtual_pipeline_model_parallel_size', None) is not None:
                # Initialize a bucket for each virtual pipeline stage
                for module in self.model:
                    if isinstance(module, (Float16Module, MCoreFloat16Module)):
                        module = module.module
                    stage_bucket = []
                    layers = module.language_model.encoder.layers
                    for layer in layers:
                        stage_bucket.extend(
                            p for p in layer.parameters() if not getattr(p, '_disable_overlap_grad_sync', False)
                        )
                    buckets.append(stage_bucket)
            else:
                # Initialize a bucket for each Transformer layer
                modules = self.model if isinstance(self.model, list) else [self.model]
                for module in modules:
                    if isinstance(module, (Float16Module, MCoreFloat16Module)):
                        module = module.module
                    layers = module.language_model.encoder.layers
                    for layer in layers:
                        buckets.append(
                            [p for p in layer.parameters() if not getattr(p, '_disable_overlap_grad_sync', False)]
                        )
            buckets.reverse()
            used_params = set()
            for bucket in buckets:
                used_params.update(bucket)
            buckets[-1].extend(p for p in self.parameters() if p not in used_params)
            self.distributed_adam_buckets = buckets

        return super().configure_optimizers()

    # Required for ONNX export
    @property
    def input_types(self) -> Optional[Dict[str, NeuralType]]:
        return {
            "input_ids": NeuralType(('B', 'T'), ChannelType()),
            "attention_mask": NeuralType(('B', 'T'), MaskType(), optional=True),
            "token_type_ids": NeuralType(('B', 'T'), ChannelType(), optional=True),
        }

    # Required for ONNX export
    def input_example(self, max_batch=1, max_dim=256):
        """
        Generates input examples for tracing etc.
        Returns:
            A tuple of input examples.
        """
        sample = next(self.parameters())
        sz = (max_batch, max_dim)
        input_ids = torch.randint(low=0, high=2048, size=sz, device=sample.device)
        token_type_ids = torch.randint(low=0, high=1, size=sz, device=sample.device)
        attention_mask = torch.randint(low=0, high=1, size=sz, device=sample.device)
        input_dict = {"input_ids": input_ids, "attention_mask": attention_mask, "token_type_ids": token_type_ids}
        return tuple([input_dict])

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

    def build_transformer_config(self) -> TransformerConfig:
        """ Builds the megatron core gpt transformer config for the model.
            For attributes in the nemo model config that are the same
            as the megatron core TransformerConfig, we will use the value from the nemo model config.
            For attributes in TransformerConfig that are not in the nemo model config, we add custom logic.
        """
        activation = self.cfg.get('activation', 'gelu')
        assert activation == 'gelu', "Only gelu activation is support for BERT at the moment."

        normalization = self.cfg.get('normalization', 'layernorm')

        layernorm_zero_centered_gamma = self.cfg.get('normalization', 'layernorm') == 'layernorm1p'
        if normalization == 'layernorm':
            normalization = 'LayerNorm'
        elif normalization == 'rmsnorm':
            normalization = 'RMSNorm'
        elif normalization == 'layernorm1p':
            normalization = 'LayerNorm'
            layernorm_zero_centered_gamma = True
        else:
            logging.warning(
                f"The normalization type: {normalization} might not be supported in megatron core."
                f"Supported types are LayerNorm and RMSNorm."
            )

        # any configs that are not in the nemo model config will be added here
        model_specific_configs = {
            'layernorm_zero_centered_gamma': layernorm_zero_centered_gamma,
            'normalization': normalization,
        }

        transformer_config = super().build_transformer_config()

        for key, value in model_specific_configs.items():
            setattr(transformer_config, key, value)

        # pass mcore customization configs directly to mcore
        mcore_customization_config_dict = self.cfg.get('mcore_customization_config', {})
        for key, value in mcore_customization_config_dict.items():
            setattr(transformer_config, key, value)

        return transformer_config

    def initialize_last_rank_embeddings(self):
        if parallel_state.get_pipeline_model_parallel_world_size() > 1:
            if self.cfg.get('share_embeddings_and_output_weights', True):
                for index, module in enumerate(self.get_model_module_list()):
                    if parallel_state.get_virtual_pipeline_model_parallel_world_size() is not None:
                        parallel_state.set_virtual_pipeline_model_parallel_rank(index)
                    sync_embeddings = (
                        module.initialize_last_stage_with_word_embeddings
                        if self.mcore_gpt
                        else module.sync_initial_word_embeddings
                    )
                    sync_embeddings()
                if parallel_state.get_virtual_pipeline_model_parallel_world_size() is not None:
                    parallel_state.set_virtual_pipeline_model_parallel_rank(0)

