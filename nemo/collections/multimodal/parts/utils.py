# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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
import os
from typing import Any, Callable, Dict, Tuple

import torch
from omegaconf import DictConfig, OmegaConf, open_dict
from PIL import Image
from pytorch_lightning import Trainer
from pytorch_lightning.plugins.environments import TorchElasticEnvironment

from nemo.collections.nlp.parts.nlp_overrides import NLPDDPStrategy, NLPSaveRestoreConnector
from nemo.utils import AppState, logging
from nemo.utils.distributed import initialize_distributed

try:
    from megatron.core import parallel_state

    HAVE_MEGATRON_CORE = True

except (ImportError, ModuleNotFoundError):

    HAVE_MEGATRON_CORE = False


def numpy_to_pil(images):
    """
    Convert a numpy image or a batch of images to a PIL image.
    """
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]

    return pil_images


def randn_like(x, generator=None):
    return torch.randn(x.shape, dtype=x.dtype, device=x.device, generator=generator)


def setup_trainer_and_model_for_inference(
    model_provider: Any, cfg: DictConfig, model_cfg_modifier: Callable,
) -> Tuple[Trainer, Any]:
    """
    Set up a trainer and NeMo model for inference.

    Args:
        model_provider (Any): An object that provides the NeMo model.
        cfg (DictConfig): The configuration dictionary, containing the
            necessary settings for the trainer and the model.
        model_cfg_modifier (Callable): A function that modifies the model
            configuration for inference.

    Returns:
        Tuple[Trainer, Any]: A tuple containing the trainer and the model.
    """

    # Check if we need to use the TorchElasticEnvironment plugin for the trainer.
    plugins = []
    if cfg.get('cluster_type', None) == 'BCP':
        plugins.append(TorchElasticEnvironment())

    # Use the NLPDDPStrategy for the distributed data parallel strategy.
    # We don't use DDP for async grad allreduce and don't find unused parameters.
    strategy = NLPDDPStrategy(no_ddp_communication_hook=True, find_unused_parameters=False,)

    # Set up the trainer with the specified plugins and strategy.
    trainer = Trainer(plugins=plugins, strategy=strategy, **cfg.trainer)

    # Create the NLPSaveRestoreConnector object for model saving and restoring.
    save_restore_connector = NLPSaveRestoreConnector()

    if cfg.model.restore_from_path.endswith(".nemo"):
        # Set the model_extracted_dir attribute if the restore path is a directory.
        if os.path.isdir(cfg.model.restore_from_path):
            save_restore_connector.model_extracted_dir = cfg.model.restore_from_path

        # Restore the model configuration from the specified path and modify it for inference.
        model_cfg = model_provider.restore_from(
            restore_path=cfg.model.restore_from_path,
            trainer=trainer,
            save_restore_connector=save_restore_connector,
            return_config=True,
        )
        with open_dict(model_cfg):
            model_cfg_modifier(model_cfg)  # modify the configuration for inference

        # assert (
        #         cfg.trainer.devices * cfg.trainer.num_nodes
        #         == model_cfg.get("tensor_model_parallel_size", 1) * model_cfg.get("pipeline_model_parallel_size", 1)
        # ), "devices * num_nodes should equal tensor_model_parallel_size * pipeline_model_parallel_size"

        # Restore the model from the specified path and configuration, and set it up for inference.
        model = model_provider.restore_from(
            restore_path=cfg.model.restore_from_path,
            trainer=trainer,
            override_config_path=model_cfg,
            save_restore_connector=save_restore_connector,
            strict=True,
        )

    elif cfg.model.restore_from_path.endswith(".ckpt"):
        logging.warning(
            "Loading from .ckpt checkpoint for inference is experimental! It doesn't support models with model parallelism!"
        )

        model = model_provider.load_from_checkpoint(
            cfg.model.restore_from_path, hparams_file=cfg.model.get("hparams_file"), trainer=trainer, strict=False,
        )

    else:
        raise ValueError(f"Unrecognized checkpoint type: {cfg.model.restore_from_path}")

    # initialize apex DDP strategy
    def dummy():
        return

    if trainer.strategy.launcher is not None:
        trainer.strategy.launcher.launch(dummy, trainer=trainer)
    trainer.strategy.setup_environment()

    model = model.cuda()  # move the model to the GPU
    model.eval().requires_grad_(False)  # set the model to evaluation mode and disable gradients

    # Return the trainer and model objects.
    return trainer, model
