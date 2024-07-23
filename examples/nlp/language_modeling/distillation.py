# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.

"""Distillation loss function(s)."""

import logging
import types
from abc import ABCMeta
from typing import Any, Dict, Tuple

import modelopt.torch.distill as mtd
import torch
import torch.nn.functional as F
from megatron.core.dist_checkpointing.mapping import ShardedStateDict
from megatron.core.parallel_state import get_tensor_model_parallel_group
from megatron.core.transformer import TransformerConfig
from torch import Tensor
from torch.nn.modules.loss import _Loss

logger = logging.getLogger(__name__)


def load_distillation_config(student_cfg: TransformerConfig) -> Dict[str, Any]:
    """Create a default distillation config for MCore GPT Models.

    Args:
        student_cfg: Model config for student model.
    """
    logit_pair = ("output_layer", "output_layer")  # logit module names for MCoreGPTModel
    tp_enabled = student_cfg.tensor_model_parallel_size > 1

    cfg = {
        "criterion": {tuple(logit_pair): LogitsKLLoss(tensor_parallel=tp_enabled)},
        "loss_balancer": None,
        "skip_lm_loss": True,
    }
    return cfg


########################################################


class BaseLoss(_Loss, metaclass=ABCMeta):
    """Abstract base class for Megatron distillation losses."""

    def __init__(self, tensor_parallel: bool = False):
        """Constructor.

        Args:
            tensor_parallel: Whether tensor parallelism is enabled or not.
        """
        super().__init__()
        self._tensor_parallel = tensor_parallel

    def pre_forward(self, predictions: Tensor, targets: Tensor) -> Tuple[Tensor, Tensor]:
        """Performs projection of student tensor to match teacher's size if necessary."""
        if isinstance(predictions, tuple):
            # `ColumnParallelLinear` returns bias too
            predictions, targets = predictions[0], targets[0]

        targets = targets.detach()

        return predictions, targets

    def post_forward(self, loss: Tensor) -> Tensor:
        """Reshapes tensor from [s, b] to [b, s] for upcoming loss masking."""
        loss = loss.transpose(0, 1).contiguous()
        return loss


class LogitsKLLoss(BaseLoss):
    """Calculates KL-Divergence loss between two logits tensors without reducing the sequence dim."""

    def __init__(
        self,
        tensor_parallel: bool = False,
        temperature: float = 1.0,
        reverse: bool = False,
    ):
        """
        Constructor.

        Args:
            tensor_parallel: Whether tensor parallelism is enabled or not.
            temperature: Divide tensors by this value prior to calculating loss.
            reverse: Whether to reverse the loss as KLD(teacher, student) instead of KLD(student, teacher)
        """
        super().__init__(tensor_parallel)
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
        if self._tensor_parallel:

            # Maximum value along vocab dimension across all GPUs.
            teacher_logits_max, _ = torch.max(output_teacher, dim=-1)
            torch.distributed.all_reduce(
                teacher_logits_max,
                op=torch.distributed.ReduceOp.MAX,
                group=get_tensor_model_parallel_group(),
            )
            output_teacher = output_teacher - teacher_logits_max.unsqueeze(dim=-1)

            denom_teacher = torch.sum(torch.exp(output_teacher), dim=-1)
            # We can't use `gather_from_tensor_model_parallel_region` here since it discards
            # gradients from other ranks - we need to all_reduce the gradients as well.
            denom_teacher = all_reduce_autograd(
                denom_teacher, group=get_tensor_model_parallel_group()
            )

            # Maximum value along vocab dimension across all GPUs.
            student_logits_max, _ = torch.max(output_student, dim=-1)
            torch.distributed.all_reduce(
                student_logits_max,
                op=torch.distributed.ReduceOp.MAX,
                group=get_tensor_model_parallel_group(),
            )
            output_student = output_student - student_logits_max.unsqueeze(dim=-1).detach()

            denom_student = torch.sum(torch.exp(output_student), dim=-1)
            denom_student = all_reduce_autograd(denom_student, group=get_tensor_model_parallel_group())

            slen, bsz, sharded_vocab_size = output_student.shape
            student_log_prob = output_student - torch.log(denom_student).view(slen, bsz, 1).expand(slen, bsz, sharded_vocab_size)
            teacher_log_prob = output_teacher - torch.log(denom_teacher).view(slen, bsz, 1).expand(slen, bsz, sharded_vocab_size)

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
                    F.kl_div(F.log_softmax(output_teacher, dim=-1), F.softmax(output_student, dim=-1), reduction="none"),
                    dim=-1,
                )
            else:
                loss = torch.sum(
                    F.kl_div(F.log_softmax(output_student, dim=-1), F.softmax(output_teacher, dim=-1), reduction="none"),
                    dim=-1,
                )

        return self.post_forward(loss)


########################################################


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
    return _AllReduce.apply(op, group, tensor)


########################################################


def adjust_distillation_model_for_mcore(model: mtd.DistillationModel, distill_cfg: Dict[str, Any]):
    """Extra modifcations to ``mtd.DistillationModel`` requried for Megatron-Core."""

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
