from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict

import modelopt.torch.distill as mtd
import torch.multiprocessing as mp
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig
from pytorch_lightning.trainer.trainer import Trainer

try:
    from megatron.core import parallel_state, tensor_parallel
    from megatron.core.models.gpt import GPTModel as MCoreGPTModel
    from megatron.core.transformer.module import Float16Module as MCoreFloat16Module
except ImportError:
    raise AssertionError("ModelOpt only supports Megatron-Core.")
import types
from abc import ABCMeta
from importlib.metadata import version
from typing import Tuple

import torch
import torch.nn.functional as F
from megatron.core.dist_checkpointing.mapping import ShardedStateDict
from megatron.core.parallel_state import get_tensor_model_parallel_group
from megatron.core.transformer import TransformerConfig
from pkg_resources import packaging
from torch import Tensor
from torch.nn.modules.loss import _Loss

from nemo.collections.common.parts.utils import extend_instance
from nemo.collections.nlp.models.language_modeling.megatron_gpt_model import (
    EmbeddingScalingMixin,
    MegatronGPTModel,
    get_specs,
)
from nemo.collections.nlp.modules.common.megatron.module import Float16Module
from nemo.collections.nlp.modules.common.megatron.utils import (
    average_losses_across_data_parallel_group,
)
from nemo.collections.nlp.parts.megatron_trainer_builder import MegatronTrainerBuilder
from nemo.collections.nlp.parts.nlp_overrides import NLPSaveRestoreConnector
from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.model_utils import load_config, unwrap_model


class DistillationMegatronGPTModel(MegatronGPTModel):
    """..."""

    def __init__(self, cfg: DictConfig, trainer: Trainer):
        """Constructor."""
        super().__init__(cfg, trainer)

        self.log("Distillation: Enabled.")
        assert self.cfg.kd_teacher_restore_from_path is not None

        # [ModelOpt]: Hack to load teacher configs properly.
        teacher_cfg = load_config(self.cfg.kd_teacher_restore_from_path)
        with self._temp_set_attr("cfg", teacher_cfg):
            teacher_transformer_config = self.build_transformer_config()

        self._teacher_cfg = teacher_cfg
        self._teacher_transformer_config = teacher_transformer_config

    @contextmanager
    def _temp_set_attr(self, attr, value):
        original_value = getattr(self, attr, None)
        setattr(self, attr, value)
        try:
            yield
        finally:
            setattr(self, attr, original_value)

    def _teacher_provider(self, model_kwargs: Dict[str, Any]) -> MCoreGPTModel:
        """Teacher model factory (must be a non-local function to pickle)."""
        teacher_model = MCoreGPTModel(config=self._teacher_transformer_config, **model_kwargs)

        self.log("Loading teacher checkpoint...")
        with self._temp_set_attr("model", teacher_model):
            self.trainer.strategy.load_checkpoint(self.cfg.kd_teacher_restore_from_path)

        return teacher_model

    def model_provider_func(self, pre_process, post_process):
        """Model depends on pipeline paralellism."""
        if not self.mcore_gpt:
            raise AssertionError("ModelOpt Distillation only supports M-Core model edition.")

        model_kwargs = dict(
            transformer_layer_spec=get_specs(
                self.spec_name,
                self.transformer_config.num_moe_experts,
                self.transformer_config.moe_grouped_gemm,
                self.transformer_engine,
                self.cfg.get('hyena', None),
            ),
            vocab_size=self.cfg.get('override_vocab_size', self.padded_vocab_size),
            max_sequence_length=self.cfg.get('encoder_seq_length', 512),
            pre_process=pre_process,
            post_process=post_process,
            parallel_output=True,
            share_embeddings_and_output_weights=self.cfg.get('share_embeddings_and_output_weights', True),
            position_embedding_type=self.cfg.get('position_embedding_type', 'learned_absolute'),
            rotary_percent=self.cfg.get('rotary_percentage', 1.0),
            seq_len_interpolation_factor=self.cfg.get('seq_len_interpolation_factor', None),
            rotary_base=self.cfg.get('rotary_base', 10000),
        )
        model = MCoreGPTModel(config=self.transformer_config, **model_kwargs)
        if self.cfg.get("apply_embedding_scaling", False) and parallel_state.is_pipeline_first_stage():
            extend_instance(model.embedding, EmbeddingScalingMixin)

        # [ModelOpt] Distillation mode.
        distill_cfg = distillation.load_distillation_config(self.transformer_config)
        # Intialize DistillationModel.
        kd_config = {
            "teacher_model": (self._teacher_provider, [model_kwargs], {}),
            "criterion": distill_cfg["criterion"],
            "loss_balancer": distill_cfg["loss_balancer"],
        }
        model = mtd.convert(model, mode=[("kd_loss", kd_config)])

        # Additional tweaks needed for MCore/Nemo.
        distillation.adjust_distillation_model_for_mcore(model, distill_cfg)

        return model

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
            batch = {
                key: val.cuda(non_blocking=True) if key in required_keys and isinstance(val, torch.Tensor) else None
                for key, val in batch.items()
            }

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

            output_tensor = model(**forward_args)

            def loss_func(output_tensor):
                # [ModelOpt] KD Loss for a micro-batch (ub)
                # loss_for_ub = self.loss_func(batch['loss_mask'], batch['num_valid_tokens_in_ub'], output_tensor)
                unwrapped_model = unwrap_model(model, (Float16Module, MCoreFloat16Module))
                loss_for_ub = unwrapped_model.compute_kd_loss(
                    loss_reduction_fn=lambda x: self.loss_func(batch['loss_mask'], batch['num_valid_tokens_in_ub'], x)
                )
                cp_size = parallel_state.get_context_parallel_world_size()
                if self.return_output_tensors:
                    # TODO: need a better way to check if loss_func is returning more stuff than just loss... (@adithyare)
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
                elif validation_step and not self.validation_drop_last:
                    num_valid_tokens_in_ub = batch['num_valid_tokens_in_ub']
                    if loss_for_ub.isnan():
                        assert batch['loss_mask'].count_nonzero() == 0, 'Got NaN loss with non-empty input'
                        loss_sum_for_ub = torch.zeros_like(loss_for_ub)
                        num_valid_tokens_in_ub = 0
                    else:
                        if self.sample_weight == 'constant':
                            num_valid_tokens_in_ub = 1
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

    def loss_func(self, loss_mask, num_valid_tokens_in_ub, output_tensor):
        loss = super().loss_func(loss_mask, num_valid_tokens_in_ub, output_tensor)
        # [ModelOpt] KD loss requires extra all-reduce to ensure same values across MP-TP partitions.
        if self.cfg.tensor_model_parallel_size > 1:
            loss = torch.sum(tensor_parallel.gather_from_tensor_model_parallel_region(loss.reshape(1)))
        return loss


########################################################


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


########################################################


@hydra_runner(config_path="conf", config_name="megatron_gpt_distill")
def main(cfg) -> None:
    logging.info("\n\n************** Experiment configuration ***********")
    logging.info(f'\n{OmegaConf.to_yaml(cfg)}')

    trainer = MegatronTrainerBuilder(cfg).create_trainer()

    # Merge model config with the one from the checkpoint
    model_cfg = load_config(cfg.model.restore_from_path)
    model_cfg.update(cfg.model)

    # Continual training
    if model_cfg.get("restore_from_path") is not None:
        # Option 1: Restore only the model weights from a .nemo file
        logging.info(f"Continual training: loading weights from {model_cfg.restore_from_path}")
        model = DistillationMegatronGPTModel.restore_from(
            restore_path=model_cfg.restore_from_path,
            override_config_path=model_cfg,
            trainer=trainer,
            save_restore_connector=NLPSaveRestoreConnector(),
        )
    elif model_cfg.get("restore_from_ckpt") is not None:
        # Option 2: Restore both model weights and optimizer states from a PTL checkpoint
        logging.info(f"Continual training: loading weights and optimizer states from {model_cfg.restore_from_ckpt}")
        trainer.ckpt_path = Path(model_cfg.restore_from_ckpt)
        model = DistillationMegatronGPTModel(model_cfg, trainer)

    # Start new pretraining or resume from a checkpoint if it exists
    else:
        model = DistillationMegatronGPTModel(model_cfg, trainer)

    trainer.fit(model)


if __name__ == '__main__':
    mp.set_start_method("spawn", force=True)
    main()
