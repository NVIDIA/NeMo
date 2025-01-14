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

import types
from abc import ABCMeta
from argparse import ArgumentParser
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from typing import Any, Callable, Dict, Optional, Tuple

import modelopt.torch.distill as mtd
import modelopt.torch.opt as mto
import torch
import torch.nn.functional as F
from megatron.core import parallel_state
from megatron.core.dist_checkpointing.mapping import ShardedStateDict
from megatron.core.models.gpt.gpt_model import GPTModel as MCoreGPTModel
from megatron.core.optimizer import OptimizerConfig
from megatron.core.transformer.module import Float16Module as MCoreFloat16Module
from megatron.core.transformer.transformer_config import TransformerConfig
from torch import Tensor
from torch.nn.modules.loss import _Loss

from nemo import lightning as nl
from nemo.collections import llm
from nemo.collections.llm.gpt.model.base import get_batch_on_this_context_parallel_rank

# from nemo.collections.nlp.models.language_modeling.megatron.gpt_layer_modelopt_spec import get_gpt_layer_modelopt_spec
from nemo.collections.nlp.modules.common.megatron.module import Float16Module
from nemo.collections.nlp.modules.common.megatron.utils import average_losses_across_data_parallel_group
from nemo.lightning import io
from nemo.lightning.ckpt_utils import ckpt_to_context_subdir
from nemo.lightning.megatron_parallel import DDP, MaskedTokenLossReduction
from nemo.lightning.pytorch.callbacks import ModelCheckpoint
from nemo.lightning.pytorch.optim import CosineAnnealingScheduler
from nemo.utils import logging
from nemo.utils.model_utils import unwrap_model


def get_args():
    """
    Parse the command line arguments.
    """
    parser = ArgumentParser(description="""Run Knowledge Distillation from a teacher model to a student.""")

    parser.add_argument("--name", type=str, required=True, help="""Experiment name""")
    parser.add_argument("--teacher_path", type=str, required=True, help="""Path to NeMo 2 checkpoint""")
    parser.add_argument("--student_path", type=str, required=True, help="""Path to NeMo 2 checkpoint""")
    parser.add_argument("--tp", type=int, default=1, help="""Tensor parallel size""")
    parser.add_argument("--cp", type=int, default=1, help="""Context parallel size""")
    parser.add_argument("--pp", type=int, default=1, help="""Pipeline parallel size""")
    parser.add_argument("--enable_sp", action="store_true", help="""Enable Sequence parallelism""")
    parser.add_argument("--precision", type=str, default="bf16-mixed", help="""Datatype for models and optimizer""")
    parser.add_argument("--devices", type=int, default=1, help="""Number of GPUs to use per node""")
    parser.add_argument("--nodes", type=int, default=1, help="""Number of nodes to use""")
    parser.add_argument("--log_dir", type=str, required=True, help="""Folder for logging and checkpoint saving""")
    parser.add_argument("--steps", type=int, required=True, help="""Number of global batches to process""")
    parser.add_argument("--global_batch_size", type=int, required=True, help="""Data samples per optimizer step""")
    parser.add_argument("--micro_batch_size", type=int, required=True, help="""Data samples per forward pass""")
    parser.add_argument("--data_paths", nargs='+', required=True, help="""List of tokenized data paths to load from""")
    parser.add_argument("--split", type=str, default="99,1,0", help="""""")
    parser.add_argument("--index_mapping_dir", type=str, default=None, help="""""")
    parser.add_argument("--sequence_length", type=int, required=True, help="""Number of tokens per input sample""")
    parser.add_argument("--lr", type=float, default=3e-5, help="""""")
    parser.add_argument("--min_lr", type=float, default=2e-7, help="""""")
    parser.add_argument("--warmup_steps", type=int, default=50, help="""""")
    parser.add_argument("--val_check_interval", type=int, default=100, help="""""")
    parser.add_argument("--limit_val_batches", type=int, default=32, help="""""")
    parser.add_argument("--limit_test_batches", type=int, default=32, help="""""")
    parser.add_argument("--log_interval", type=int, default=10, help="""""")

    args = parser.parse_args()
    return args


def gpt_distillation_data_step(dataloader_iter, attn_mask_cpu=False) -> Dict[str, torch.Tensor]:
    batch = next(dataloader_iter)

    _batch: dict
    if isinstance(batch, tuple) and len(batch) == 3:
        _batch = batch[0]
    else:
        _batch = batch

    required_device_keys = set()
    required_host_keys = set()

    if attn_mask_cpu:
        # [ModelOpt]: We cache data for PP distillation, and save GPU mem by storing masks on CPU mem.
        required_host_keys.add("attention_mask")
    else:
        required_device_keys.add("attention_mask")

    if 'cu_seqlens' in _batch:
        required_device_keys.add('cu_seqlens')
        required_host_keys.add('cu_seqlens_argmin')
        required_host_keys.add('max_seqlen')

    if parallel_state.is_pipeline_first_stage():
        required_device_keys.update(("tokens", "position_ids"))
    if parallel_state.is_pipeline_last_stage():
        required_device_keys.update(("labels", "loss_mask"))

    _batch_required_keys = {}
    for key, val in _batch.items():
        if key in required_device_keys:
            _batch_required_keys[key] = val.cuda(non_blocking=True)
        elif key in required_host_keys:
            _batch_required_keys[key] = val.cpu()
        else:
            _batch_required_keys[key] = None

    # slice batch along sequence dimension for context parallelism
    output = get_batch_on_this_context_parallel_rank(_batch_required_keys)

    return output


@dataclass
class DistillationGPTConfig(llm.GPTConfig):
    kd_teacher_restore_from_path: str = ""  # default set only for dataclass inheritance

    data_step_fn: Callable = gpt_distillation_data_step

    def configure_model(self, *args, **kwargs) -> MCoreGPTModel:
        if not self.kd_teacher_restore_from_path:
            raise ValueError("Config attribute `kd_teacher_restore_from_path` must be set.")
        if self.virtual_pipeline_model_parallel_size is not None:
            raise ValueError("ModelOpt Distillation incompatible with interleaved pipeline schedule.")

        model = super().configure_model(*args, **kwargs)

        # [ModelOpt] Intialize DistillationModel.
        distill_cfg = load_distillation_config(self)
        kd_config = {
            "teacher_model": (_teacher_provider, [], {"cfg": self}),
            "criterion": distill_cfg["criterion"],
            "loss_balancer": distill_cfg["loss_balancer"],
        }
        model = mtd.convert(model, mode=[("kd_loss", kd_config)])

        # Additional MCore-specific tweaks needed.
        adjust_distillation_model_for_mcore(model, model_cfg=self, distill_cfg=distill_cfg)

        return model


class _DistillationLossReduction(MaskedTokenLossReduction):
    """Custom masking and reduction callable used only in training mode."""

    def __init__(self, distillation_loss_fn, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._distillation_loss_fn = distillation_loss_fn
        self._cp_size = parallel_state.get_context_parallel_world_size()

    def forward(self, batch: Dict[str, Tensor], forward_out: Tensor) -> Tuple[Tensor, Dict[str, Tensor]]:
        if isinstance(forward_out, tuple):
            # neva returns (logits, loss_mask)
            forward_out, batch["loss_mask"] = forward_out

        # [ModelOpt]: KD loss calculation.
        loss_for_ub = self._distillation_loss_fn(
            loss_reduction_fn=lambda x: self._masked_token_loss(
                x, batch["loss_mask"], batch.get("num_valid_tokens_in_ub")
            )
        )

        reduced_loss = average_losses_across_data_parallel_group([loss_for_ub])
        return loss_for_ub * self._cp_size, {"avg": reduced_loss}

    def _masked_token_loss(self, loss_output: Tensor, mask: Tensor, num_valid_tokens_in_ub: Optional[int] = None):
        """
        The function takes as input per-token loss and masks non-required values.
        """
        if isinstance(loss_output, tuple):
            # [ModelOpt]: Losses can return extra flag to indicate additional TP-reduction (often required)
            loss_output, tp_reduce = loss_output
        losses = loss_output.float()
        loss_mask = mask.view(-1).float()

        if self._cp_size > 1:
            if num_valid_tokens_in_ub is None:
                num_valid_tokens_in_ub = loss_mask.sum()
            if num_valid_tokens_in_ub < 0.5:  # no valid tokens
                num_valid_tokens_in_ub += 1.0
            loss = torch.sum(losses.view(-1) * loss_mask) / num_valid_tokens_in_ub  # sequence level nll
            torch.distributed.all_reduce(loss, group=parallel_state.get_context_parallel_group())
        else:
            loss = torch.sum(losses.view(-1) * loss_mask) / loss_mask.sum()  # sequence level nll

        if tp_reduce is True:
            torch.distributed.all_reduce(loss, group=parallel_state.get_tensor_model_parallel_group())

        return loss


class DistillationGPTModel(llm.GPTModel):
    """Custom GPT subclass for distillation-related modifications."""

    def data_step(self, dataloader_iter, cache_num_batches: Optional[int] = None) -> Dict[str, torch.Tensor]:
        if cache_num_batches:
            batches = [self.config.data_step_fn(dataloader_iter, attn_mask_cpu=True) for _ in range(cache_num_batches)]
            return _LoopingCachedDataIterator(batches)
        elif isinstance(dataloader_iter, _LoopingCachedDataIterator):
            batch = next(dataloader_iter)
            if "attention_mask" in batch:
                batch["attention_mask"] = batch["attention_mask"].cuda(non_blocking=True)  # move back to GPU
            return batch
        else:
            return self.config.data_step_fn(dataloader_iter)

    @property
    def training_loss_reduction(self) -> _DistillationLossReduction:
        if not self._training_loss_reduction:
            core_module = unwrap_model(self.module, (DDP, Float16Module, MCoreFloat16Module))
            self._training_loss_reduction = _DistillationLossReduction(
                distillation_loss_fn=core_module.compute_kd_loss
            )

        return self._training_loss_reduction


########################################################


class BaseLoss(_Loss, metaclass=ABCMeta):
    """Abstract base class for Megatron distillation losses."""

    def __init__(self, model_config: TransformerConfig):
        """
        Constructor.

        Args:
            model_config: MCore transformer config.
        """
        super().__init__()
        self._config = model_config

    def pre_forward(self, predictions: Tensor, targets: Tensor) -> Tuple[Tensor, Tensor]:
        """Prepares inputs safely for loss computation."""
        if isinstance(predictions, tuple):
            # `ColumnParallelLinear` returns bias too
            predictions, targets = predictions[0], targets[0]
        targets = targets.detach()

        return predictions, targets

    def post_forward(self, loss: Tensor, tp_reduce: bool = False) -> Tensor:
        """Reshapes tensor from [s, b] to [b, s] for upcoming loss masking."""
        loss = loss.transpose(0, 1).contiguous()
        return loss, tp_reduce


class LogitsKLLoss(BaseLoss):
    """Calculates KL-Divergence loss between two logits tensors without reducing the sequence dim."""

    def __init__(self, model_config: TransformerConfig, temperature: float = 1.0, reverse: bool = False):
        """
        Constructor.

        Args:
            model_config: MCore transformer config.
            temperature: Divide tensors by this value prior to calculating loss.
            reverse: Whether to reverse the loss as KLD(teacher, student) instead of KLD(student, teacher)
        """
        super().__init__(model_config)
        self._temperature = temperature
        self._reverse = reverse

    def forward(self, predictions: Tensor, targets: Tensor) -> Tensor:
        """
        Forward function.

        Args:
            predictions: Student model tensors (size [s, b, h])
            targets: Teacher model tensors (size [s, b, h])

        Returns:
            KLD loss of tensors (size [b, s])
        """
        predictions, targets = self.pre_forward(predictions, targets)

        # Division by temp should happen prior to finding max for both student and teacher.
        # Currently we don't use temperature in any of ours runs (temp=1.0)
        output_teacher = targets.float() / self._temperature
        output_student = predictions.float() / self._temperature

        # Compute local softmax, and the reweight to compute global softmax.
        if self._config.tensor_model_parallel_size > 1:

            # Maximum value along vocab dimension across all GPUs.
            teacher_logits_max, _ = torch.max(output_teacher, dim=-1)
            torch.distributed.all_reduce(
                teacher_logits_max,
                op=torch.distributed.ReduceOp.MAX,
                group=parallel_state.get_tensor_model_parallel_group(),
            )
            output_teacher = output_teacher - teacher_logits_max.unsqueeze(dim=-1)

            denom_teacher = torch.sum(torch.exp(output_teacher), dim=-1)
            # We can't use standard reduction function here since the computation
            # that follows it isn't identical across TP ranks.
            denom_teacher = all_reduce_autograd(denom_teacher, group=parallel_state.get_tensor_model_parallel_group())

            # Maximum value along vocab dimension across all GPUs.
            student_logits_max, _ = torch.max(output_student, dim=-1)
            torch.distributed.all_reduce(
                student_logits_max,
                op=torch.distributed.ReduceOp.MAX,
                group=parallel_state.get_tensor_model_parallel_group(),
            )
            output_student = output_student - student_logits_max.unsqueeze(dim=-1).detach()

            denom_student = torch.sum(torch.exp(output_student), dim=-1)
            denom_student = all_reduce_autograd(denom_student, group=parallel_state.get_tensor_model_parallel_group())

            slen, bsz, sharded_vocab_size = output_student.shape
            student_log_prob = output_student - torch.log(denom_student).view(slen, bsz, 1).expand(
                slen, bsz, sharded_vocab_size
            )
            teacher_log_prob = output_teacher - torch.log(denom_teacher).view(slen, bsz, 1).expand(
                slen, bsz, sharded_vocab_size
            )

            if self._reverse:
                loss = torch.sum(
                    F.kl_div(teacher_log_prob, student_log_prob, reduction="none", log_target=True),
                    dim=-1,
                )
            else:
                loss = torch.sum(
                    F.kl_div(student_log_prob, teacher_log_prob, reduction="none", log_target=True),
                    dim=-1,
                )

        else:
            if self._reverse:
                loss = torch.sum(
                    F.kl_div(
                        F.log_softmax(output_teacher, dim=-1),
                        F.softmax(output_student, dim=-1),
                        reduction="none",
                    ),
                    dim=-1,
                )
            else:
                loss = torch.sum(
                    F.kl_div(
                        F.log_softmax(output_student, dim=-1),
                        F.softmax(output_teacher, dim=-1),
                        reduction="none",
                    ),
                    dim=-1,
                )

        return self.post_forward(loss, tp_reduce=True)


class _AllReduce(torch.autograd.Function):
    """Implementation from old PyTorch `torch.distributed.nn.parallel`."""

    @staticmethod
    def forward(ctx, op, group, tensor):
        ctx.group, ctx.op = group, op
        tensor = tensor.clone()
        torch.distributed.all_reduce(tensor, op=op, group=group)
        return tensor

    @staticmethod
    def backward(ctx, grad_output):
        return (None, None, _AllReduce.apply(ctx.op, ctx.group, grad_output))


def all_reduce_autograd(tensor, op=torch.distributed.ReduceOp.SUM, group=torch.distributed.group.WORLD):
    """Custom all-reduce function.

    Needed instead of other all-reduce functions available when the computation following
    the all-reduce call differs per rank. In KL loss, this corresponds to the different numerators.
    """
    return _AllReduce.apply(op, group, tensor)


########################################################


def load_distillation_config(cfg: DistillationGPTConfig) -> Dict[str, Any]:
    """Create a default distillation config for MCore GPT Models.

    Args:
        student_cfg: Model config for student model.
    """
    logit_pair = ("output_layer", "output_layer")  # logit module names for MCoreGPTModel
    distill_cfg = {
        "criterion": {},
        "loss_balancer": None,
        "skip_lm_loss": True,
    }
    if cfg.pipeline_model_parallel_size == 1 or parallel_state.is_pipeline_last_stage():
        distill_cfg["criterion"][logit_pair] = LogitsKLLoss(cfg)

    return distill_cfg


def _teacher_provider(cfg: DistillationGPTConfig) -> MCoreGPTModel:
    """Teacher model factory (must be a non-local function to pickle)."""

    logging.info("Distillation: Loading teacher weights...")
    strategy = nl.MegatronStrategy(
        tensor_model_parallel_size=cfg.tensor_model_parallel_size,
        context_parallel_size=cfg.context_parallel_size,
        pipeline_model_parallel_size=cfg.pipeline_model_parallel_size,
        ckpt_load_optimizer=False,
        ckpt_parallel_save_optim=False,
        setup_optimizers=False,
        ddp="pytorch",
    )
    trainer = nl.Trainer(
        devices=cfg.tensor_model_parallel_size,
        num_nodes=cfg.context_parallel_size * cfg.pipeline_model_parallel_size,
        strategy=strategy,
        plugins=nl.MegatronMixedPrecision(precision="bf16-mixed"),
    )

    # TODO(aanoosheh): Replace spec with modelopt one
    model, _ = io.ModelConnector().nemo_load(cfg.kd_teacher_restore_from_path, trainer, cpu=False)
    model = unwrap_model(model.module, (DDP, Float16Module, MCoreFloat16Module))

    logging.info("Distillation: ... teacher weights loaded.")
    return model


class _LoopingCachedDataIterator:
    def __init__(self, data):
        self.data = data
        self.it = iter(self.data)

    def __next__(self):
        try:
            return next(self.it)
        except StopIteration:
            self.it = iter(self.data)
            return next(self.it)


def adjust_distillation_model_for_mcore(
    model: mtd.DistillationModel, model_cfg: TransformerConfig, distill_cfg: Dict[str, Any]
):
    """Extra modifcations to ``mtd.DistillationModel`` requried for Megatron-Core."""

    # HACK: Get rid of ModelOpt Distillation state
    # NOTE: If re-placed, above losses need modifcation as `TransformerConfig` has non-pickleable elements.
    mto.ModeloptStateManager(model)._state.pop()

    # HACK: Hide teacher during `sharded_state_dict` method.
    def _sharded_state_dict(self, *args, **kwargs) -> ShardedStateDict:
        with self.hide_teacher_model():
            return self._sharded_state_dict(*args, **kwargs)

    model._sharded_state_dict = model.sharded_state_dict
    model.sharded_state_dict = types.MethodType(_sharded_state_dict, model)

    # HACK: Skip `lm_loss` bypassing it when training if not needed for backprop.
    def _compute_language_model_loss(self, labels, logits) -> Tensor:
        if self.training:
            return torch.zeros_like(labels)
        return self._compute_language_model_loss(labels, logits)

    if distill_cfg["skip_lm_loss"]:
        model._compute_language_model_loss = model.compute_language_model_loss
        model.compute_language_model_loss = types.MethodType(_compute_language_model_loss, model)

    # HACK: Skip `lm_loss` always for teacher.
    def _compute_language_model_loss(self, labels, logits) -> Tensor:
        return torch.zeros_like(labels)

    model.teacher_model.compute_language_model_loss = types.MethodType(
        _compute_language_model_loss, model.teacher_model
    )

    if model_cfg.pipeline_model_parallel_size > 1:

        def _set_input_tensor(self, input_tensor: Tensor):
            obj = self.teacher_model if self._only_teacher_fwd else self
            return type(self).set_input_tensor(obj, input_tensor)

        # HACK: Pipeline-parallel Distillation requires a way to cache input batches for subsequent
        # forward calls, as well as a way to pass through output tensors to teacher model.
        model.set_input_tensor = types.MethodType(_set_input_tensor, model)

        @contextmanager
        def _swap_teacher_config(self, model_wrapper):
            try:
                if hasattr(model_wrapper, "config"):
                    model_wrapper._config = model_wrapper.config
                model_wrapper.config = self.teacher_model.config
                yield
            finally:
                del model_wrapper.config
                if hasattr(model_wrapper, "_config"):
                    model_wrapper.config = model_wrapper._config
                    del model_wrapper._config

        # HACK: Pipeline-parallel forward function relies on the config in the model to know what
        # hidden size of tensor to communicate to next stage.
        model.swap_teacher_config = types.MethodType(_swap_teacher_config, model)


########################################################


if __name__ == "__main__":
    logging.info("Distillation enabled.")

    args = get_args()

    ## initialize the strategy and trainer
    strategy = nl.MegatronStrategy(
        tensor_model_parallel_size=args.tp,
        pipeline_model_parallel_size=args.pp,
        context_parallel_size=args.cp,
        sequence_parallel=args.enable_sp,
    )
    trainer = nl.Trainer(
        devices=args.devices,
        num_nodes=args.nodes,
        max_steps=args.steps,
        log_every_n_steps=args.log_interval,
        val_check_interval=args.val_check_interval,
        limit_val_batches=args.limit_val_batches,
        limit_test_batches=args.limit_test_batches,
        strategy=strategy,
        accelerator="gpu",
        plugins=nl.MegatronMixedPrecision(precision=args.precision),
    )

    ## load student model
    model = nl.io.load_context(path=ckpt_to_context_subdir(args.student_path), subpath="model")
    assert hasattr(model, "tokenizer"), "Please provide a model checkpoint with tokenizer included."
    model_config, tokenizer = model.config, model.tokenizer

    model_config = DistillationGPTConfig(
        **asdict(model_config),
        # transformer_layer_spec=get_gpt_layer_modelopt_spec(),  # TODO(aanoosheh)
        kd_teacher_restore_from_path=args.teacher_path,
    )
    model = DistillationGPTModel(model_config, tokenizer=tokenizer)

    # setup the dataset
    data = llm.PreTrainingDataModule(
        paths=args.data_paths,
        seq_length=args.sequence_length,
        micro_batch_size=args.micro_batch_size,
        global_batch_size=args.global_batch_size,
        split=args.split,
        index_mapping_dir=args.index_mapping_dir,
        tokenizer=tokenizer,
    )

    # auto-resume setup
    # resume = nl.AutoResume(
    #     resume_if_exists=True,
    #     resume_ignore_no_checkpoint=True,
    #     resume_from_directory=LOG_DIR,
    #     restore_config=nl.RestoreConfig(path=STUDENT_PATH) if STUDENT_PATH else None,
    # )

    ## setup the optimizer
    opt_config = OptimizerConfig(
        optimizer="adam",
        lr=args.lr,
        bf16=("bf16" in args.precision),
        use_distributed_optimizer=True,
    )
    sched = CosineAnnealingScheduler(
        max_steps=args.steps,
        warmup_steps=args.warmup_steps,
        constant_steps=0,
        min_lr=args.min_lr,
    )
    opt = nl.MegatronOptimizerModule(opt_config, sched)

    # checkpointing and logging
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        save_top_k=1,
        every_n_train_steps=args.val_check_interval,
    )
    logger = nl.NeMoLogger(
        name=args.name,
        log_dir=args.log_dir,
        ckpt=checkpoint_callback,
    )

    # run
    llm.train(
        model=model,
        data=data,
        optim=opt,
        tokenizer='model',
        trainer=trainer,
        log=logger,
    )
