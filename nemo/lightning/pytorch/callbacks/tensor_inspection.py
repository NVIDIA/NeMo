import os
from typing import Any, Dict

import lightning.pytorch as pl
import nvdlfw_inspect.api as nvinspect_api
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.loggers import WandbLogger as PLWandbLogger
from megatron.core.parallel_state import get_tensor_and_data_parallel_group

from nemo.utils import logging
from typing import Dict, Optional

class TensorInspectConfig:
    """Python-based configuration for tensor inspection tool.

    Args:
        features (Dict[str, Dict]): Dictionary of feature configurations.
            Each feature should contain:
            - enabled (bool): Whether the feature is enabled
            - layers (dict): Layer selection criteria (regex patterns, layer types)
            - feature-specific configurations (LogTensorStats, custom namespace features, etc.)
        log_dir (str): Directory path for storing logs
        feature_dirs (list): List of directories containing feature implementations
    """

    def __init__(self, features: Dict[str, Dict], log_dir: str, feature_dirs: list):
        self.features = features
        self.log_dir = log_dir
        self.feature_dirs = feature_dirs

        # Validate that each feature has required base fields
        for feature_name, feature_config in features.items():
            if not isinstance(feature_config, dict):
                raise ValueError(f"Feature {feature_name} configuration must be a dictionary")

            if 'enabled' not in feature_config:
                raise ValueError(f"Feature {feature_name} must have 'enabled' field")

            if 'layers' not in feature_config:
                raise ValueError(f"Feature {feature_name} must have 'layers' configuration")

    @property
    def multi_tensor_stat_collection(self):
        return self.features.get('multi_tensor_stat_collection', {})

    @property
    def transformer_engine(self):
        return self.features.get('transformer_engine', {})


class TensorInspectCallback(Callback):
    def __init__(self, config: TensorInspectConfig):
        super().__init__()
        self.config = config
        self.debug_setup_done = False
        self.statistics_logger = None

        nvinspect_api.initialize(
            config_file=self.config.features,
            feature_dirs=self.config.feature_dirs,
            log_dir=self.config.log_dir,
            statistics_logger=self.statistics_logger,
        )
        logging.info("nvinspect_api initialized with Python configuration.")

    def on_train_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if not self.debug_setup_done:
            # Configure additional loggers from available trainer loggers
            for logger in trainer.loggers:
                if isinstance(logger, PLWandbLogger):
                    from nvdlfw_inspect.logging import wrap_wandb_logger, MetricLogger
                    wandb_logger = wrap_wandb_logger(logger)
                    MetricLogger.add_logger(wandb_logger)
                    logging.info("Added WandB logger to nvdlfw_inspect for tensor statistics")
                
                # Also check for TensorBoard loggers
                elif hasattr(logger, 'experiment') and 'SummaryWriter' in str(type(logger.experiment)):
                    from nvdlfw_inspect.logging import wrap_tensorboard_writer, MetricLogger
                    tb_logger = wrap_tensorboard_writer(logger.experiment)
                    MetricLogger.add_logger(tb_logger)
                    logging.info("Added TensorBoard logger to nvdlfw_inspect for tensor statistics")
            
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
