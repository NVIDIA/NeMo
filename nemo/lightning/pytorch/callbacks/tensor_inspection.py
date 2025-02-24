import os
from typing import Any, Dict

import lightning.pytorch as pl
import nvdlfw_inspect.api as nvinspect_api
from lightning.pytorch.callbacks import Callback
from megatron.core.parallel_state import get_tensor_and_data_parallel_group

from nemo.utils import logging


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

        nvinspect_api.initialize(
            config_file=self.config.features,
            feature_dirs=self.config.feature_dirs,
            log_dir=self.config.log_dir,
            statistics_logger=None,
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
