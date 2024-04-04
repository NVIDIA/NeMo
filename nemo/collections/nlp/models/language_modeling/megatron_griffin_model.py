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

from megatron.core.models.griffin.griffin_model import GriffinModel
from pkg_resources import packaging
from importlib.metadata import version
import torch
import torch.nn.functional as F
from omegaconf.dictconfig import DictConfig
from pytorch_lightning.trainer.trainer import Trainer
from pytorch_lightning.accelerators import CPUAccelerator
from nemo.collections.nlp.modules.common.megatron.build_model import build_model
from contextlib import nullcontext
import os
from nemo.collections.nlp.modules.common.megatron.utils import (
    ApexGuardDefaults,
    average_losses_across_data_parallel_group,
)
from nemo.collections.nlp.models.language_modeling.megatron_gpt_model import MegatronGPTModel
from nemo.collections.nlp.models.language_modeling.megatron_base_model import MegatronBaseModel

from nemo.utils import logging

try:

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

    HAVE_MEGATRON_CORE = True

except (ImportError, ModuleNotFoundError):
    TransformerConfig = ApexGuardDefaults
    HAVE_MEGATRON_CORE = False

try:
    import transformer_engine

    HAVE_TE = True

except (ImportError, ModuleNotFoundError):
    HAVE_TE = False

class MegatronGriffinModel(MegatronGPTModel):
    """
    Megatron Griffin pretraining.
    """

    def __init__(self, cfg: DictConfig, trainer: Trainer):
        if not HAVE_MEGATRON_CORE:
            raise ImportError(
                "megatron-core was not found. Please see the NeMo README for installation instructions: https://github.com/NVIDIA/NeMo#megatron-gpt."
            )

        # build the transformer config
        # TODO: add type hint once pip package is out
        
        self.vocab_size = cfg.get('vocab_size', 256_128)
        MegatronBaseModel.__init__(self, cfg=cfg, trainer=trainer)

        self.cfg = cfg
        self.transformer_config = self.build_transformer_config()
        self.transformer_config.gated_linear_unit = True
        

        self.megatron_amp_O2 = cfg.get('megatron_amp_O2', False)
        
        self.mcore_gpt=True

        self._validate_trainer()

        self.spec_name = cfg.get('name', '')
        if cfg.get('fp8', False):
            self.prev_step_training = True

        self.rampup_batch_size = self.cfg.get('rampup_batch_size', None)
        if self.rampup_batch_size:
            self.prev_consumed_samples = 0
            self.if_first_step = 0
            self.prev_global_batch_size = None

        if cfg.get('data', None) is not None:
            self.reset_position_ids = cfg.data.get('reset_position_ids', False)
            self.reset_attention_mask = cfg.data.get('reset_attention_mask', False)
            self.eod_mask_loss = cfg.data.get('eod_mask_loss', False)

        if not self.megatron_amp_O2 and self.cfg.get('virtual_pipeline_model_parallel_size', None):
            raise ValueError('Virtual pipeline model parallel is only supported when using megatron_amp_O2')

        if not self.megatron_amp_O2 and self.cfg.get('expert_model_parallel_size', 1) > 1:
            raise ValueError('Expert parallelism is only supported when using megatron_amp_O2')

        # TODO(akoumparouli): this is temporary and will be removed in the future.
        if self.cfg.get('expert_model_parallel_size', 1) > 1 and self.with_distributed_adam:
            raise ValueError('Expert parallelism is currently not supporting distributed optimizer')

        # build_model returns a list of modules which are used for interleaved pipeline parallelism
        if isinstance(self.trainer.accelerator, CPUAccelerator):
            self.model = build_model(
                model_provider_func=self.model_provider_func,
                wrap_with_ddp=False,
                on_cpu=True,
                virtual_pipeline_model_parallel_size=self.cfg.get('virtual_pipeline_model_parallel_size', None),
            )
        else:
            build_model_context = nullcontext
            if HAVE_TE and self.cfg.get('fp8', False) and self.cfg.get('fp8_params', False):
                build_model_context = transformer_engine.pytorch.fp8_model_init
            with build_model_context():
                self.model = build_model(
                    model_provider_func=self.model_provider_func,
                    wrap_with_ddp=False,
                    virtual_pipeline_model_parallel_size=self.cfg.get('virtual_pipeline_model_parallel_size', None),
                    on_cpu=cfg.get('fsdp', False) and cfg.get('use_cpu_initialization', False),
                )

        # if we're not using interleaved, then self.model is a module.
        if self.cfg.get('virtual_pipeline_model_parallel_size', None) is None:
            self.model = self.model[0]

        if self.megatron_amp_O2:

            if not self.with_distributed_adam and not self.cfg.get("use_cpu_initialization", False):
                # Pre-allocate the model on GPU to have master parameters allocated on the same device with matching data type
                if isinstance(self.model, list):
                    for module in self.model:
                        module.cuda(torch.cuda.current_device())
                else:
                    self.model.cuda(torch.cuda.current_device())

            self._wrap_model_for_O2()

        self.enable_autocast = (
            True if (not self.megatron_amp_O2) and (self.autocast_dtype in [torch.float16, torch.bfloat16]) else False
        )

        self.transformer_engine = cfg.get('transformer_engine', False)

        # configuration used for inference
        self._inference_config = None

        # Convert the global-batch-based profile index to micro-batch index
        if hasattr(self, '_nsys_profile_enabled'):
            mp_size = cfg.get('tensor_model_parallel_size', 1) * cfg.get('pipeline_model_parallel_size', 1)
            cp_size = cfg.get('context_parallel_size', 1)
            data_parallel_world_size = trainer.world_size // (mp_size * cp_size)
            grad_accum_steps = cfg.get('global_batch_size') // (cfg.get('micro_batch_size') * data_parallel_world_size)
            self._nsys_profile_start_step *= grad_accum_steps
            self._nsys_profile_end_step *= grad_accum_steps

        self.get_attention_mask_from_fusion = self.cfg.get('get_attention_mask_from_fusion', True)
        self.initialize_ub = self.cfg.get('ub_tp_comm_overlap', False)
        self.log_train_loss = bool(int(os.getenv("NEMO_LOG_TRAIN_LOSS", 1)))
        self.log_memory_usage = bool(int(os.getenv("NEMO_LOG_MEMORY_USAGE", 0)))
        self.loss_broadcast_src_rank = None

        self.inference_params = None

        # default to false since this doesn't work with sequence parallelism currently
        self.use_loss_mask = self.cfg.get('use_loss_mask', False)

        if self.use_loss_mask and self.transformer_config.sequence_parallel:
            raise ValueError('Loss mask is not supported with sequence parallelism.')

    def model_provider_func(self, pre_process, post_process):
        
        model = GriffinModel(
            config=self.transformer_config,
            vocab_size=self.vocab_size,
            )
    
        return model


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

    def on_validation_epoch_end(self):
        
        averaged_loss = torch.tensor(0.0, dtype=torch.float32).cuda()
        return averaged_loss
    
    def sharded_state_dict(self, prefix: str = ''):
            return None


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

            forward_args['input_ids'] = torch.randint(0, 100, (2, 128)).cuda()
            forward_args['labels'] = torch.randint(0, 100, (2, 128)).cuda()
            batch['loss_mask'] = torch.ones(256).cuda()
            batch['num_valid_tokens_in_ub'] = torch.tensor(4.).cuda()
            output_tensor = model(forward_args['input_ids'], None, labels=forward_args['labels'])

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

   