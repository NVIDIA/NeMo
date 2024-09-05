from dataclasses import dataclass, fields
from typing import List, Optional

import pytorch_lightning as pl
import torch
from megatron.core import ModelParallelConfig
from megatron.core.distributed import DistributedDataParallelConfig
from megatron.core.optimizer import OptimizerConfig
from pytorch_lightning.callbacks.callback import Callback

from nemo.lightning.pytorch.strategies.megatron_strategy import MegatronStrategy, ParallelismConfig
from nemo.utils import logging
from nemo.utils.get_rank import get_rank

HAVE_TE = True
try:
    import transformer_engine
except (ImportError, ModuleNotFoundError):
    HAVE_TE = False


@dataclass
class TPOverlapCfg:
    pass


@dataclass
class PipelineOverlapCfg(TPOverlapCfg):
    """Tensor parallel overlap via gemm, communication pipelineing."""

    num_sm: int
    cga_size: int
    num_splits: int
    set_sm_margin: bool
    fp8_buf: bool = False
    method: str = 'pipeline'


@dataclass
class RingExchangeOverlapCfg(TPOverlapCfg):
    """Tensor parallel overlap via ring exchange."""

    aggregate: bool = False
    method: str = 'ring_exchange'


@dataclass
class BulkOverlapCfg(TPOverlapCfg):
    """Tensor parallel overlap for comm, gemms without serial dependencies."""

    num_sm: int
    cga_size: int
    set_sm_margin: bool
    method: str = 'bulk'


@dataclass
class TransformerLayerTPOverlapCfg:
    """Each tensor parallel gemm in the transformer layer has
    a specific tensor parallel overlap scheme."""

    qkv_dgrad: TPOverlapCfg
    qkv_wgrad: TPOverlapCfg
    fc1_dgrad: TPOverlapCfg
    fc1_wgrad: TPOverlapCfg
    qkv_fprop: TPOverlapCfg
    proj_dgrad: TPOverlapCfg
    fc1_fprop: TPOverlapCfg
    fc2_dgrad: TPOverlapCfg
    proj_fprop: TPOverlapCfg
    fc2_fprop: TPOverlapCfg


@dataclass
class _CommOverlapConfig:
    # Tensor parallel communication overlap (experimental)
    tp_comm_overlap: bool = None
    tp_comm_overlap_cfg: dict = None
    # Pipeline parallel communication overlap
    overlap_p2p_comm: bool = None
    batch_p2p_comm: bool = None
    # Data parallel communication overlap
    overlap_grad_reduce: bool = None
    overlap_param_gather: bool = None
    overlap_param_gather_with_optimizer_step: bool = None
    align_param_gather: bool = None
    bucket_size: int = None
    # Pipeline bubble overlap
    defer_embedding_wgrad_compute: bool = None
    wgrad_deferral_limit: int = None


class MegatronCommOverlapCallback(Callback):
    """
    A PyTorch Lightning callback to enable communication compute overlap.
    This callback enables the following:
        - tensor parallel communication overlap
        - pipeline parallel communication overlap
        - data parallel communication overlap
        - pipeline bubble overlap

    Args:
        tp_comm_overlap (bool): Enable tensor parallel overlap
        tp_comm_overlap_cfg (TransformerLayerTPOverlapCfg):
        overlap_p2p_comm (bool): Enable pipeline parallel overlap
        batch_p2p_comm (bool):
        overlap_grad_reduce (bool):
        overlap_param_gather (bool):
        overlap_param_gather_with_optimizer_step (bool):
        align_param_gather (bool):
        bucket_size (int):
        defer_embedding_wgrad_compute (bool):
        wgrad_deferral_limit (int):

    Example:
        >>> callback = MegatronCommOverlapCallback()
        >>> trainer = Trainer(callbacks=[callback])
    """

    def __init__(
        self,
        # Tensor parallel communication overlap (experimental)
        tp_comm_overlap: bool = None,
        tp_comm_overlap_cfg: TransformerLayerTPOverlapCfg = None,
        # Pipeline parallel communication overlap
        overlap_p2p_comm: bool = None,
        batch_p2p_comm: bool = None,
        # Data parallel communication overlap
        overlap_grad_reduce: bool = None,
        overlap_param_gather: bool = None,
        overlap_param_gather_with_optimizer_step: bool = None,
        align_param_gather: bool = None,
        bucket_size: int = None,
        # Pipeline bubble overlap
        defer_embedding_wgrad_compute: bool = None,
        wgrad_deferral_limit: int = None,
    ):

        self.overlap_cfg = _CommOverlapConfig(
            tp_comm_overlap=tp_comm_overlap,
            tp_comm_overlap_cfg=tp_comm_overlap_cfg,
            overlap_p2p_comm=overlap_p2p_comm,
            batch_p2p_comm=batch_p2p_comm,
            overlap_grad_reduce=overlap_grad_reduce,
            overlap_param_gather=overlap_param_gather,
            overlap_param_gather_with_optimizer_step=overlap_param_gather_with_optimizer_step,
            align_param_gather=align_param_gather,
            bucket_size=bucket_size,
            defer_embedding_wgrad_compute=defer_embedding_wgrad_compute,
            wgrad_deferral_limit=wgrad_deferral_limit,
        )

    def _apply_model_comm_overlap_cfgs(
        self,
        parallelism_cfg: ParallelismConfig,
        model_parallel_cfg: ModelParallelConfig,
    ) -> ModelParallelConfig:
        comm_overlap_cfg = _CommOverlapConfig()

        vp_size = parallelism_cfg.virtual_pipeline_model_parallel_size
        if vp_size is None:
            vp_size = 1

        # TP overlap is disabled by default, can be overriden by user
        comm_overlap_cfg.tp_comm_overlap = False
        comm_overlap_cfg.tp_comm_overlap_cfg = None

        defer_embedding_wgrad_compute = False
        wgrad_deferral_limit = 0

        # PP overlap
        if parallelism_cfg.pipeline_model_parallel_size > 1:
            if vp_size > 1:
                comm_overlap_cfg.overlap_p2p_comm = True
                comm_overlap_cfg.batch_p2p_comm = False
            else:
                comm_overlap_cfg.overlap_p2p_comm = False
                comm_overlap_cfg.batch_p2p_comm = True
        else:
            comm_overlap_cfg.overlap_p2p_comm = False
            comm_overlap_cfg.batch_p2p_comm = False

        model_parallel_cfg = self._apply_overlap_configs(comm_overlap_cfg, model_parallel_cfg)

        # Check if TP overlap can be safely enabled
        if hasattr(model_parallel_cfg, "tp_comm_overlap") and model_parallel_cfg.tp_comm_overlap is True:
            if parallelism_cfg.tensor_model_parallel_size < 2:
                logging.warning("Disabling tensor parallel communication overlap due to TP size < 2.")
                model_parallel_cfg.tp_comm_overlap = False
            elif not parallelism_cfg.sequence_parallel:
                logging.warning("Disabling tensor parallel communication overlap due to sequence_parallel=False.")
                model_parallel_cfg.tp_comm_overlap = False
            elif not HAVE_TE:
                logging.warning(
                    "Disabling tensor parallel communication overlap due to Tranformer Engine not detected."
                )
                model_parallel_cfg.tp_comm_overlap = False
        return model_parallel_cfg

    def _apply_optimizer_overlap_cfgs(
        self,
        parallelism_cfg: ParallelismConfig,
        optim_cfg: OptimizerConfig,
        ddp_cfg: DistributedDataParallelConfig,
    ) -> OptimizerConfig:
        # Data parallel overlap is only available with the Megatron DDP and Distributed optimizer
        if not ddp_cfg.use_distributed_optimizer:
            return

        from nemo.utils import AppState

        app_state = AppState()
        data_parallel_size = app_state.data_parallel_size

        vp_size = parallelism_cfg.virtual_pipeline_model_parallel_size
        if vp_size is None:
            vp_size = 1

        comm_overlap_cfg = _CommOverlapConfig()
        comm_overlap_cfg.bucket_size = None
        comm_overlap_cfg.overlap_grad_reduce = False
        comm_overlap_cfg.overlap_param_gather = False
        comm_overlap_cfg.overlap_param_gather_with_optimizer_step = False
        comm_overlap_cfg.align_param_gather = False

        if data_parallel_size > 1:
            comm_overlap_cfg.bucket_size = 128 * 1024 * 1024
            comm_overlap_cfg.overlap_grad_reduce = True
            comm_overlap_cfg.overlap_param_gather = True
            if parallelism_cfg.pipeline_model_parallel_size > 1 and vp_size > 1:
                comm_overlap_cfg.overlap_param_gather_with_optimizer_step = True
                comm_overlap_cfg.align_param_gather = True

        self._apply_overlap_configs(comm_overlap_cfg, optim_cfg)
        self._apply_overlap_configs(comm_overlap_cfg, ddp_cfg)

    def _apply_overlap_configs(self, comm_overlap_cfg: _CommOverlapConfig, cfg):
        # override default configs with any user provided configs
        user_comm_overlap_cfg = self.overlap_cfg
        if isinstance(user_comm_overlap_cfg, _CommOverlapConfig):
            for field in fields(user_comm_overlap_cfg):
                user_value = getattr(user_comm_overlap_cfg, field.name)
                if user_value is not None:
                    setattr(comm_overlap_cfg, field.name, user_value)
        # apply optimizations into target cfg
        for field in fields(comm_overlap_cfg):
            if hasattr(cfg, field.name):
                setattr(cfg, field.name, getattr(comm_overlap_cfg, field.name))
        return cfg

    def setup(self, trainer: pl.Trainer, pl_module: pl.LightningModule, stage: str) -> None:
        assert isinstance(trainer.strategy, MegatronStrategy), "MegatronCommOverlapCallback requires MegatronStrategy"
        parallelism_cfg = trainer.strategy.parallelism

        if hasattr(trainer.model, "config") and isinstance(trainer.model.config, ModelParallelConfig):
            self._apply_model_comm_overlap_cfgs(
                parallelism_cfg=parallelism_cfg, model_parallel_cfg=trainer.model.config
            )
            if trainer.model.config.tp_comm_overlap:
                trainer.strategy.tp_comm_overlap_need_init = True

        if (
            hasattr(trainer.model.optim, "config")
            and isinstance(trainer.model.optim.config, OptimizerConfig)
            and isinstance(trainer.strategy.ddp_config, DistributedDataParallelConfig)
        ):
            self._apply_optimizer_overlap_cfgs(
                parallelism_cfg=parallelism_cfg,
                optim_cfg=trainer.model.optim.config,
                ddp_cfg=trainer.strategy.ddp_config,
            )
