import os
from typing import Any
import lightning.pytorch as pl
from lightning.pytorch.callbacks import Callback
from megatron.core.parallel_state import get_tensor_and_data_parallel_group
import nvdlfw_inspect.api as nvinspect_api
from nemo.utils import logging

class TensorInspectConfig:
    """Python-based configuration for tensor inspection."""
    def __init__(self, multi_tensor_stat_collection: dict, transformer_engine: dict, log_dir: str, feature_dirs: list):
        self.multi_tensor_stat_collection = multi_tensor_stat_collection
        self.transformer_engine = transformer_engine
        self.log_dir = log_dir
        self.feature_dirs = feature_dirs

class TensorInspectCallback(Callback):
    def __init__(self, config: TensorInspectConfig):
        super().__init__()
        self.config = config
        self.debug_setup_done = False

        nvinspect_api.initialize(
            multi_tensor_stat_collection=self.config.multi_tensor_stat_collection,
            transformer_engine=self.config.transformer_engine,
            log_dir=self.config.log_dir,
            feature_dirs=self.config.feature_dirs,
        )
        logging.info("nvinspect_api initialized with Python configuration.")

    def on_train_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if not self.debug_setup_done:
            nvinspect_api.infer_and_assign_layer_names(pl_module)
            nvinspect_api.set_tensor_reduction_group(get_tensor_and_data_parallel_group())
            self.debug_setup_done = True
            logging.info("nvinspect_api: Inferred layer names and set tensor reduction group.")

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: pl.utilities.types.STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
    ) -> None:
        nvinspect_api.step()