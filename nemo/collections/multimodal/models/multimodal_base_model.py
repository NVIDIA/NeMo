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

import copy
import hashlib
import json
import os
import re
from typing import Any, Dict, Optional, Union

import torch
from omegaconf import open_dict
from omegaconf.dictconfig import DictConfig
from pytorch_lightning.core.saving import _load_state as ptl_load_state
from pytorch_lightning.core.saving import load_hparams_from_tags_csv, load_hparams_from_yaml
from pytorch_lightning.plugins.precision.native_amp import NativeMixedPrecisionPlugin
from pytorch_lightning.trainer.connectors.logger_connector.fx_validator import _FxValidator
from pytorch_lightning.trainer.trainer import Trainer
from pytorch_lightning.utilities import rank_zero_only
from pytorch_lightning.utilities.cloud_io import load as pl_load
from pytorch_lightning.utilities.migration import pl_legacy_patch
from transformers import TRANSFORMERS_CACHE

from nemo.collections.nlp.modules.common.megatron.clip_grads import (
    clip_grad_norm_distributed_optimizer,
    clip_grad_norm_fp32,
)
from nemo.collections.nlp.modules.common.megatron.megatron_init import initialize_model_parallel_for_nemo
from nemo.collections.nlp.modules.common.tokenizer_utils import get_nmt_tokenizer, get_tokenizer
from nemo.collections.nlp.parts import utils_funcs
from nemo.collections.nlp.parts.nlp_overrides import (
    NEMO_MEGATRON_MODEL_PARALLEL_APPSTATE_OVERRIDE,
    GradScaler,
    NLPSaveRestoreConnector,
)
from nemo.core.classes import ModelPT
from nemo.core.classes.exportable import Exportable
from nemo.core.optim import MainParamsOptimizerWrapper, prepare_lr_scheduler
from nemo.utils import AppState, logging
from nemo.utils.get_rank import is_global_rank_zero

try:
    from apex.transformer.pipeline_parallel.utils import get_num_microbatches

    HAVE_APEX = True
except (ImportError, ModuleNotFoundError):
    HAVE_APEX = False

try:
    from megatron.core import parallel_state

    HAVE_MEGATRON_CORE = True
except (ImportError, ModuleNotFoundError):
    HAVE_MEGATRON_CORE = False


__all__ = ['MultimodalModel', 'MegatronMultimodalModel']

NEMO_MULTIMODAL_TMP = os.path.join(os.path.dirname(str(TRANSFORMERS_CACHE)), "nemo_multimodal_tmp")

os.makedirs(NEMO_MULTIMODAL_TMP, exist_ok=True)


class MultimodalModel(ModelPT, Exportable):
    """Base class for Multimodal Models.
    """

    def __init__(self, cfg: DictConfig, trainer: Trainer = None):

        super().__init__(cfg, trainer)

        # handles model parallel save and restore logic
        self._save_restore_connector = NLPSaveRestoreConnector()

    # TODO(yuya): check below
    @property
    def input_module(self):
        return self

    @property
    def output_module(self):
        return self

    @property
    def is_model_parallel_initialized(self):
        app_state = AppState()
        if app_state.model_parallel_group is not None:
            return True
        else:
            return False

    @classmethod
    def load_from_checkpoint(
        cls,
        checkpoint_path: str,
        map_location: Any = None,
        hparams_file: Optional[str] = None,
        strict: bool = True,
        **kwargs,
    ):
        """
        Loads ModelPT from checkpoint, with some maintenance of restoration.
        For documentation, please refer to LightningModule.load_from_checkpoin() documentation.
        """
        checkpoint = None
        try:
            cls._set_model_restore_state(is_being_restored=True)
            # TODO: replace with proper PTL API
            with pl_legacy_patch():
                if map_location is not None:
                    checkpoint = pl_load(checkpoint_path, map_location=map_location)
                else:
                    checkpoint = pl_load(checkpoint_path, map_location=lambda storage, loc: storage)

            if hparams_file is not None:
                extension = hparams_file.split(".")[-1]
                if extension.lower() == "csv":
                    hparams = load_hparams_from_tags_csv(hparams_file)
                elif extension.lower() in ("yml", "yaml"):
                    hparams = load_hparams_from_yaml(hparams_file)
                else:
                    raise ValueError(".csv, .yml or .yaml is required for `hparams_file`")

                hparams["on_gpu"] = False

                # overwrite hparams by the given file
                checkpoint[cls.CHECKPOINT_HYPER_PARAMS_KEY] = hparams

            # for past checkpoint need to add the new key
            if cls.CHECKPOINT_HYPER_PARAMS_KEY not in checkpoint:
                checkpoint[cls.CHECKPOINT_HYPER_PARAMS_KEY] = {}
            # override the hparams with values that were passed in
            cfg = checkpoint[cls.CHECKPOINT_HYPER_PARAMS_KEY].get('cfg', checkpoint[cls.CHECKPOINT_HYPER_PARAMS_KEY])
            # TODO: can we do this without overriding?
            config_kwargs = kwargs.copy()
            if 'trainer' in config_kwargs:
                config_kwargs.pop('trainer')
            cfg.update(config_kwargs)

            # Disable individual unet/vae weights loading otherwise the model will look for these partial ckpts and raise error
            if cfg:
                if cfg.get('unet_config') and cfg.get('unet_config').get('from_pretrained'):
                    cfg.unet_config.from_pretrained = None
                if cfg.get('first_stage_config') and cfg.get('first_stage_config').get('from_pretrained'):
                    cfg.first_stage_config.from_pretrained = None
                ## Append some dummy configs that DB didn't support
                if not cfg.get('channels_last'):
                    cfg.channels_last = True
                if not cfg.get('capture_cudagraph_iters'):
                    cfg.capture_cudagraph_iters = -1

            # compatibility for stable diffusion old checkpoint tweaks
            first_key = list(checkpoint['state_dict'].keys())[0]
            if first_key == "betas":
                # insert "model." into for megatron wrapper
                new_state_dict = {}
                for key in checkpoint['state_dict'].keys():
                    new_key = "model." + key
                    new_state_dict[new_key] = checkpoint['state_dict'][key]
                checkpoint['state_dict'] = new_state_dict
            elif (
                first_key == 'model.text_encoder.transformer.text_model.embeddings.position_ids'
                or first_key == 'model.text_encoder.model.language_model.embedding.position_embeddings'
            ):
                # remap state keys from dreambooth when using HF clip
                new_state_dict = {}
                for key in checkpoint['state_dict'].keys():
                    new_key = key.replace('._orig_mod', "")
                    new_key = new_key.replace('unet', 'model.diffusion_model')
                    new_key = new_key.replace('vae', 'first_stage_model')
                    new_key = new_key.replace('text_encoder', 'cond_stage_model')
                    new_key = new_key.replace('.noise_scheduler', '')
                    new_state_dict[new_key] = checkpoint['state_dict'][key]
                checkpoint['state_dict'] = new_state_dict

            # compatibility for inductor in inference
            if not cfg.get('inductor', False):
                new_state_dict = {}
                for key in checkpoint['state_dict'].keys():
                    new_key = key.replace('._orig_mod', '', 1)
                    new_state_dict[new_key] = checkpoint['state_dict'][key]
                checkpoint['state_dict'] = new_state_dict

            # compatibility for inductor in inference
            if not cfg.get('inductor', False):
                new_state_dict = {}
                for key in checkpoint['state_dict'].keys():
                    new_key = key.replace('._orig_mod', '', 1)
                    new_state_dict[new_key] = checkpoint['state_dict'][key]
                checkpoint['state_dict'] = new_state_dict

            if cfg.get('megatron_amp_O2', False):
                new_state_dict = {}
                for key in checkpoint['state_dict'].keys():
                    new_key = key.replace('model.', 'model.module.', 1)
                    new_state_dict[new_key] = checkpoint['state_dict'][key]
                checkpoint['state_dict'] = new_state_dict

            if 'cfg' in kwargs:
                model = ptl_load_state(cls, checkpoint, strict=strict, **kwargs)
            else:
                model = ptl_load_state(cls, checkpoint, strict=strict, cfg=cfg, **kwargs)
                # cfg = checkpoint[cls.CHECKPOINT_HYPER_PARAMS_KEY].cfg

            checkpoint = model

        finally:
            cls._set_model_restore_state(is_being_restored=False)
        return checkpoint


class MegatronMultimodalModel(MultimodalModel):
    """
    Megatron multimodal base class
    It does the following things:
    1. Initialize the model parallel for nemo given the model parallel parameters.
    2. Turn on all the nvidia optimizations.
    3. If using distributed optimizer, configure to be compatible with
       O2-level optimizations and/or model parallelism.
    4. Perform gradient clipping: `grad_clip_pl_default` triggers the
       PyTorch Lightning default implementation, `with_distributed_adam`
       triggers the distributed optimizer's implementation,
       `megatron_amp_O2` triggers gradient clipping on the main grads,
       and otherwise gradient clipping is performed on the model grads.
    """

    def __init__(self, cfg: DictConfig, trainer: Trainer):

        if not HAVE_MEGATRON_CORE:
            raise ImportError(
                "megatron-core was not found. Please see the NeMo README for installation instructions: https://github.com/NVIDIA/NeMo#megatron-gpt."
            )

        if trainer is None:
            raise ValueError(f"Trainer cannot be None for Megatron-based models. Please provide a PTL trainer object.")

        # this prevents base constructor from initializing tokenizer
        self.tokenizer = None
        super().__init__(cfg, trainer=trainer)

        self._validate_config()

        self.with_distributed_adam = cfg.optim.get('name') == 'distributed_fused_adam'

        # used in NVIDIA NGC PyTorch containers
        self._enable_nvidia_optimizations()

        if self._cfg.get('use_cpu_initialization', False) is False:
            torch.cuda.set_device(trainer.local_rank)

        # buffer used during train_step for logging average loss over gradient accumulation steps
        self._reduced_loss_buffer = []

        # Overrides used when converting checkpoints
        if os.environ.get(NEMO_MEGATRON_MODEL_PARALLEL_APPSTATE_OVERRIDE, "false").lower() == "true":
            app_state = AppState()
            init_world_size = app_state.tensor_model_parallel_size * app_state.pipeline_model_parallel_size
            init_global_rank = app_state.global_rank
            init_local_rank = app_state.local_rank
        else:
            init_world_size = trainer.world_size
            init_global_rank = trainer.global_rank
            init_local_rank = trainer.local_rank

        initialize_model_parallel_for_nemo(
            world_size=init_world_size,
            global_rank=init_global_rank,
            local_rank=init_local_rank,
            tensor_model_parallel_size=cfg.get('tensor_model_parallel_size', 1),
            pipeline_model_parallel_size=cfg.get('pipeline_model_parallel_size', 1),
            virtual_pipeline_model_parallel_size=cfg.get('virtual_pipeline_model_parallel_size', None),
            pipeline_model_parallel_split_rank=cfg.get('pipeline_model_parallel_split_rank', 0),
            micro_batch_size=cfg.get('micro_batch_size'),
            global_batch_size=cfg.get('global_batch_size'),
            rampup_batch_size=cfg.get('rampup_batch_size'),
            use_fp8=cfg.get('fp8', False),
            seed=self.cfg.get('seed', 1234),
            apex_transformer_log_level=self.cfg.get('apex_transformer_log_level', 30),
        )

        self.grad_clip_pl_default = False  # use pytorch default for gradient clipping. Default False

        if hasattr(self._cfg, "tokenizer") or (
            hasattr(self._cfg, "encoder_tokenizer") and hasattr(self._cfg, "decoder_tokenizer")
        ):
            # build tokenizer (defaults to nemo supported tokenizers)
            self._build_tokenizer()

            # manipulate vocabulary (e.g., pad vocabulary for better efficiency)
            self._build_vocab()

        # TODO: remove this when PTL 1.7.3 is released
        _FxValidator.functions["configure_gradient_clipping"] = {
            "allowed_on_step": (False, True),
            "allowed_on_epoch": (False, True),
            "default_on_step": True,
            "default_on_epoch": False,
        }

        self.use_fsdp = cfg.get('fsdp', False)

    def _enable_nvidia_optimizations(self):
        "These optimizations are present in NVIDIA NGC PyTorch Containers"

        # NVIDIA container version check
        nvidia_torch_version = os.getenv('NVIDIA_PYTORCH_VERSION', None)
        if nvidia_torch_version is not None:
            try:
                NVIDIA_TORCH_MAJOR = int(nvidia_torch_version.split('.')[0])
            except Exception:
                NVIDIA_TORCH_MAJOR = 0
            try:
                NVIDIA_TORCH_MINOR = int(nvidia_torch_version.split('.')[1])
            except Exception:
                NVIDIA_TORCH_MINOR = 0

            # Apex Persistent layer norm is supported from Nvidia PyTorch container v21.11
            if NVIDIA_TORCH_MAJOR < 21 or (NVIDIA_TORCH_MAJOR == 21 and NVIDIA_TORCH_MINOR < 11):
                self.cfg.persist_layer_norm = False

            if NVIDIA_TORCH_MAJOR >= 21 or (NVIDIA_TORCH_MAJOR == 21 and NVIDIA_TORCH_MINOR >= 11):
                # NVFUSER
                torch._C._jit_set_profiling_executor(True)
                torch._C._jit_set_profiling_mode(True)
                torch._C._jit_override_can_fuse_on_cpu(False)
                torch._C._jit_override_can_fuse_on_gpu(False)
                torch._C._jit_set_texpr_fuser_enabled(False)
                torch._C._jit_set_nvfuser_enabled(True)
                torch._C._debug_set_autodiff_subgraph_inlining(False)

        else:
            # Not a Nvidia container. NVFUSER Dependency check is on users
            pass

    def _build_tokenizer(self):
        """
        Default tokenizer is based on available nemo tokenizers.
        Override this method to use an external tokenizer.
        All tokenizers are expected to provide compatible interface.
        Override default Encoder-decoder tokenizer to use legacy=True for sentencepiece.
        """
        if hasattr(self._cfg.tokenizer, "sentencepiece_legacy"):
            legacy = self._cfg.tokenizer.sentencepiece_legacy
        else:
            legacy = True if self._cfg.tokenizer.library == 'sentencepiece' else False
        self.tokenizer = get_nmt_tokenizer(
            library=self._cfg.tokenizer.library,
            model_name=self._cfg.tokenizer.type,
            tokenizer_model=self.register_artifact("tokenizer.model", self._cfg.tokenizer.model),
            vocab_file=self.register_artifact("tokenizer.vocab_file", self._cfg.tokenizer.vocab_file),
            merges_file=self.register_artifact("tokenizer.merge_file", self._cfg.tokenizer.merge_file),
            delimiter=self.cfg.tokenizer.get('delimiter', None),
            legacy=legacy,
        )

    def _build_vocab(self):
        """
        Manipulate vocabulary (e.g., pad vocabulary for increased performance)/
        """
        # TODO: add config to allow to disable it?
        self.padded_vocab_size = self._vocab_size_with_padding(
            orig_vocab_size=self.tokenizer.vocab_size,
            make_vocab_size_divisible_by=self._cfg.get('make_vocab_size_divisible_by', 128),
            tensor_model_parallel_size=self._cfg.get('tensor_model_parallel_size', 1),
        )

    def _vocab_size_with_padding(self, orig_vocab_size, make_vocab_size_divisible_by, tensor_model_parallel_size):
        """Pad vocab size so it is divisible by model parallel size and
        still having GPU friendly size."""

        after = orig_vocab_size
        multiple = make_vocab_size_divisible_by * tensor_model_parallel_size
        while (after % multiple) != 0:
            after += 1
        logging.info(
            f'Padded vocab_size: {after}, original vocab_size: {orig_vocab_size}, dummy tokens: {after - orig_vocab_size}.'
        )
        return after

    def on_train_start(self) -> None:
        super().on_train_start()
        self.init_global_step = self.trainer.global_step

    def _get_parameters(self):
        """
        private method to load all the trainable parameters from optimizer param groups
        """
        params = []
        for param_group in self._optimizer_param_groups:
            for param in param_group['params']:
                params.append(param)
        return params

    def configure_gradient_clipping(self, *args, **kwargs):
        """PTL hook to configure gradients.
           We use gradient clipping implementation from megatron-lm.
        """
        clip_val = self.trainer.gradient_clip_val
        if clip_val is None:
            return

        clip_val = float(clip_val)
        if clip_val <= 0:
            return

        if self.grad_clip_pl_default:
            # use the default behavior
            return super().configure_gradient_clipping(*args, **kwargs)

        if self.with_distributed_adam:
            grad_norm = clip_grad_norm_distributed_optimizer(self._optimizer, clip_val)
        else:
            if self.megatron_amp_O2:
                # grep fp32 master parameters for gradient clipping
                parameters = self._optimizer.get_parameters()
            else:
                parameters = self._get_parameters()
            grad_norm = clip_grad_norm_fp32(parameters=parameters, max_norm=clip_val)

        self.log('grad_norm', grad_norm, rank_zero_only=True, batch_size=1)

    def allreduce_gradients(self):
        """Reduce gradients across data parallel ranks.
           Modified from megatron-lm: https://github.com/NVIDIA/Megatron-LM/blob/d41696840ed0a7edb7e0499eb82a48ae112d9bb3/megatron/model/distributed.py#L188
        """
        # Bucketize and all-reduce
        buckets = {}
        for param in self.parameters():
            if param.requires_grad and param.grad is not None:
                tp = param.data.type()
                if tp not in buckets:
                    buckets[tp] = []
                buckets[tp].append(param)
                # param.main_grad = param.grad

        # For each bucket, all-reduce and copy all-reduced grads.
        for tp in buckets:
            bucket = buckets[tp]
            grads = [param.grad.data for param in bucket]
            coalesced = torch._utils._flatten_dense_tensors(grads)
            coalesced /= parallel_state.get_data_parallel_world_size()
            torch.distributed.all_reduce(coalesced, group=parallel_state.get_data_parallel_group())
            for buf, synced in zip(grads, torch._utils._unflatten_dense_tensors(coalesced, grads)):
                buf.copy_(synced)

    def reduce_overlap_gradients(self):
        """Reduce grads if overlapped grad sync is enabled

        Used for pipeline parallelism with the distributed Adam
        optimizer. In the first pipeline stage, the grad sync is
        overlapped with the final backward pass. In other pipeline
        stages, the grad sync is deferred until the bubble overhead.

        """
        if self.with_distributed_adam and self._optimizer.overlap_grad_sync:
            if params is None:
                params = self._optimizer.parameters()
            self._optimizer.try_grad_sync(params)

    def sync_overlap_parameters(self, params=None):
        if self.with_distributed_adam:
            self._optimizer._try_start_bucket_param_sync(params)

    def on_train_batch_end(self, outputs, dataloader_iter: Any, batch_idx: int, unused: Optional[int] = 0) -> None:
        super().on_train_batch_end(outputs, dataloader_iter, batch_idx)

        # TODO: Replace with newer override for scheduler.step() instead of
        # search for plugins for fp16 GradScalar
        if self.trainer.precision_plugin is not None and isinstance(
            self.trainer.precision_plugin, NativeMixedPrecisionPlugin
        ):
            precision_plugin = self.trainer.precision_plugin

            if (
                hasattr(precision_plugin, 'scaler')
                and precision_plugin.scaler is not None
                and isinstance(precision_plugin.scaler, GradScaler)
            ):
                grad_scaler = precision_plugin.scaler

                # If the grad scaler skipped its optimizer step due to infs/nans,
                # decrement the step of all schedulers.
                if grad_scaler.optimizer_update_skipped is not None and grad_scaler.optimizer_update_skipped is True:
                    scheduler_cfgs = self.trainer.lr_scheduler_configs

                    if not scheduler_cfgs or not self.trainer.lightning_module.automatic_optimization:
                        return

                    for scheduler_cfg in scheduler_cfgs:
                        # Decrement the counter by 2, then perform a scheduler.step() to perform a no-up
                        # as well as update the optimizer lr in all param groups
                        scheduler_cfg.scheduler.last_epoch -= 2
                        scheduler_cfg.scheduler.step()

                    # Increase the max step count by 1

                    # Reset the optimizer update skipped to `None` - this is to prevent scheduler no-ops during
                    # accumulated gradient updates.
                    grad_scaler.optimizer_update_skipped = None

    def setup_optimization(
        self, optim_config: Optional[Union[DictConfig, Dict]] = None, optim_kwargs: Optional[Dict[str, Any]] = None,
    ):
        optim_kwargs = {} if optim_kwargs is None else optim_kwargs.copy()
        if self.with_distributed_adam:

            # Allocate contiguous buffers to avoid extra copies
            optim_kwargs['contiguous_grad_buffer'] = True
            optim_kwargs['contiguous_param_buffer'] = True

            # Make sure optimizer state is in FP32
            optim_dtype = torch.float32
            optim_kwargs['dtype'] = optim_dtype

            # Make sure embedding grad reductions are in FP32
            for name, param in self.named_parameters():
                if 'word_embedding' in name or 'position_embedding' in name:
                    param._with_fp32_optimizer = True

            # Match param allgather with model dtype
            model_dtype = torch.float32
            if self.megatron_amp_O2 and hasattr(self, 'autocast_dtype'):
                model_dtype = self.autocast_dtype
            optim_kwargs['param_sync_dtype'] = model_dtype

            # Determine whether to store master params in optimizer
            if optim_dtype == model_dtype:
                optim_kwargs['store_params'] = False
            elif optim_dtype == torch.float32 and model_dtype == torch.bfloat16:
                optim_kwargs['store_params'] = False
                optim_kwargs['store_param_remainders'] = True
            else:
                optim_kwargs['store_params'] = True

        return super().setup_optimization(optim_config=optim_config, optim_kwargs=optim_kwargs)

    def configure_optimizers(self):
        self.setup_optimization()

        # Wrap the baseline optimizer with the optimizer class with master parameters
        if self.megatron_amp_O2 and not self.with_distributed_adam and self._optimizer is not None:
            if self.cfg.precision == 'bf16':
                fp32_grad_accum = True
                contiguous_grad_bucket = True
            elif self.cfg.precision == 16:
                fp32_grad_accum = False
                # TODO: contiguous grad bucket for fp16 is also planned to be supported
                contiguous_grad_bucket = False
                raise ValueError(
                    "fp16 training is not yet supported with O2. Please set megatron_amp_O2 to False in the model config."
                )

            # if using tensor parallel only, we automatically use async grad all-reduce
            # if using pipeline parallel or sequence parallel or gradient accumulation fusion, then we disable it
            # if self.cfg.get('pipeline_model_parallel_size', 1) == 1 and not (
            #     self.cfg.get('sequence_parallel', False) or self.cfg.get('gradient_accumulation_fusion', False)
            # ):
            #     async_grad_allreduce = True
            # else:
            #     async_grad_allreduce = False

            async_grad_allreduce = False

            if async_grad_allreduce:
                # we need this to be configurable until make_nccl_premul_sum is in public PyTorch.
                # currently cannot be imported in PyTorch 1.12.0
                grad_div_ar_fusion = self.cfg.get('grad_div_ar_fusion', False)
            else:
                grad_div_ar_fusion = False

            self._optimizer = MainParamsOptimizerWrapper(
                self._optimizer,
                fp32_grad_accum=fp32_grad_accum,
                contiguous_grad_bucket=contiguous_grad_bucket,
                async_grad_allreduce=async_grad_allreduce,
                grad_div_ar_fusion=grad_div_ar_fusion,
                grad_allreduce_chunk_size_mb=self.cfg.get('grad_allreduce_chunk_size_mb', 125),
            )

            assert self._trainer.max_steps is not None, "'max_steps' is missing in trainer config."
            if hasattr(self._cfg.optim, 'sched'):
                sched_config = self._cfg.optim.sched
                sched_config['max_steps'] = self._trainer.max_steps
                self._scheduler = prepare_lr_scheduler(
                    optimizer=self._optimizer, scheduler_config=sched_config, train_dataloader=self._train_dl
                )

        # Configure distributed optimizer
        if self.with_distributed_adam:

            # Initialize param buckets if explicitly provided
            if hasattr(self, 'distributed_adam_buckets'):
                for bucket in self.distributed_adam_buckets:
                    self._optimizer.init_params_bucket(bucket)
                del self.distributed_adam_buckets

            # Make sure all params are initialized so main grads are
            # available
            # Note: Consolidate grads without overlap
            overlap_params = []
            no_overlap_params = []
            for p in self.parameters():
                if getattr(p, '_disable_overlap_grad_sync', False):
                    no_overlap_params.append(p)
                else:
                    overlap_params.append(p)
            self._optimizer.init_params(reversed(overlap_params))
            self._optimizer.init_params(reversed(no_overlap_params))

            # Initialize contiguous parameter buffer
            self._optimizer.init_param_buffer()

        if self._scheduler is None:
            return self._optimizer
        else:
            return [self._optimizer], [self._scheduler]

    def compute_consumed_samples(self, steps_since_resume=0):
        app_state = AppState()
        consumed_samples = (
            self.init_consumed_samples
            + steps_since_resume * app_state.data_parallel_size * self.cfg.micro_batch_size * get_num_microbatches()
        )
        return int(consumed_samples)

    def _extract_consumed_samples_from_ckpt(self, ckpt_path):
        try:
            init_consumed_samples = int(float(re.findall(r"consumed_samples\=([0-9]+.[0-9]+)", ckpt_path)[0]))
        except (ValueError, TypeError, IndexError):
            logging.warning("Cannot parse the checkpoint file to get the consumed samples. assume it is zero.")
            init_consumed_samples = 0

        return init_consumed_samples

    def _validate_config(self):
        """ Certain configurations might be incompatible or discouraged. We can check for them here."""

        if self.cfg.get('sequence_parallel', False) and self.cfg.get('tensor_model_parallel_size', 1) == 1:
            logging.info(
                "Sequence parallel should only be used with tensor parallel size > 1. Setting sequence parallel to False"
            )
            with open_dict(self.cfg):
                self.cfg.sequence_parallel = False

        if (
            self.cfg.get('gradient_accumulation_fusion', False)
            and self.cfg.get('pipeline_model_parallel_size', 1) == 1
        ):
            logging.info("Gradient accumulation fusion can only be used with pipeline parallel size > 1.")
            with open_dict(self.cfg):
                self.cfg.gradient_accumulation_fusion = False

        if self.cfg.get('gradient_accumulation_fusion', False) and not self.cfg.get('megatron_amp_O2', False):
            logging.info("Gradient accumulation fusion can only be used with megatron amp O2 mixed precision.")
            with open_dict(self.cfg):
                self.cfg.gradient_accumulation_fusion = False

    def is_data_parallel_rank_zero(self):
        if is_global_rank_zero():
            return True
        else:
            try:
                data_parallel_rank = parallel_state.get_data_parallel_rank()
            except:
                data_parallel_rank = None

            if data_parallel_rank is not None and data_parallel_rank == 0:
                return True
            else:
                return False

    def configure_sharded_model(self):
        if self.use_fsdp:
            """ Top-evel FSDP model sharding """
            # Cast the full model to initialization precision to match the precision among parameters.
            params_dtype = utils_funcs.dtype_from_precision(self.cfg.precision, None)
            self.model = self.model.to(params_dtype)
            # Shard the top-level model with FSDP. We shard the strategy-unwrapped model not
            # to lose the structure of non-FSDP wrapped parameters (e.g, embedding)
            self.model = self.trainer.strategy._setup_model(self.model)
            # Keep the master parameters in fp32.
            # The prameters can be initialized in half-precision to save memory before sharding
            self.model = self.model.float()
            # Move model from CPU to GPU, which is to avoid out-of-memory carash before sharding.
            # FSDP with `use_cpu_initialization` has the model initialized on CPU then move GPU after sharding.
            # In case of GPU-initialized model, this is no-op.
            self.model = self.model.cuda(torch.cuda.current_device())
