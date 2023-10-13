# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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

import glob
import os

import torch
from omegaconf.omegaconf import OmegaConf, open_dict
from PIL import Image
from pytorch_lightning import Trainer
from pytorch_lightning.plugins.environments import TorchElasticEnvironment
from torch.utils.data import DataLoader, Dataset

from nemo.collections.nlp.parts.nlp_overrides import NLPDDPStrategy, NLPSaveRestoreConnector
from nemo.collections.vision.data.imagenet_classnames import imagenet_classnames
from nemo.collections.vision.data.megatron.vit_dataset import ClassificationTransform
from nemo.collections.vision.models.megatron_vit_classification_models import MegatronVitClassificationModel
from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.get_rank import is_global_rank_zero
from nemo.utils.trt_utils import build_engine


@hydra_runner(config_path="conf", config_name="megatron_vit_classification_export")
def main(cfg) -> None:
    logging.info("\n\n************** Experiment configuration ***********")
    logging.info(f'\n{OmegaConf.to_yaml(cfg)}')

    output_dir = cfg.infer.out_path
    max_batch_size = cfg.infer.max_batch_size
    max_dim = cfg.infer.max_dim
    plugins = []
    strategy = NLPDDPStrategy(
        no_ddp_communication_hook=True, find_unused_parameters=False,  # we don't use DDP for async grad allreduce
    )
    print(type(cfg.trainer.precision))
    if cfg.get('cluster_type', None) == 'BCP':
        plugins.append(TorchElasticEnvironment())
    trt_precision = cfg.trainer.precision
    cfg.trainer.precision = 32
    # trainer required for restoring model parallel models
    trainer = Trainer(plugins=plugins, strategy=strategy, **cfg.trainer)

    save_restore_connector = NLPSaveRestoreConnector()
    if os.path.isdir(cfg.model.restore_from_path):
        save_restore_connector.model_extracted_dir = cfg.model.restore_from_path

    model_cfg = MegatronVitClassificationModel.restore_from(
        restore_path=cfg.model.restore_from_path,
        trainer=trainer,
        save_restore_connector=save_restore_connector,
        return_config=True,
    )

    assert (
        cfg.trainer.devices * cfg.trainer.num_nodes
        == model_cfg.tensor_model_parallel_size * model_cfg.pipeline_model_parallel_size
    ), "devices * num_nodes should equal tensor_model_parallel_size * pipeline_model_parallel_size"

    # These configs are required to be off during inference.
    with open_dict(model_cfg):
        model_cfg.precision = int(trainer.precision) if trainer.precision.isdigit() else trainer.precision
        print(type(model_cfg.precision))
        if trainer.precision != "bf16":
            model_cfg.megatron_amp_O2 = False
        model_cfg.sequence_parallel = False
        model_cfg.activations_checkpoint_granularity = None
        model_cfg.activations_checkpoint_method = None

    model = MegatronVitClassificationModel.restore_from(
        restore_path=cfg.model.restore_from_path,
        trainer=trainer,
        override_config_path=model_cfg,
        save_restore_connector=save_restore_connector,
        strict=True,
    )

    model.eval()

    # initialize apex DDP strategy
    def dummy():
        return

    if trainer.strategy.launcher is not None:
        trainer.strategy.launcher.launch(dummy, trainer=trainer)
    trainer.strategy.setup_environment()

    os.makedirs(f"{output_dir}/onnx/", exist_ok=True)
    os.makedirs(f"{output_dir}/plan/", exist_ok=True)

    model.export(f"{output_dir}/onnx/vit.onnx", dynamic_axes={'tokens': {0: 'B'}})

    input_profile = {}
    bs1_example = model.input_example(max_batch=1, max_dim=max_dim)[0]
    bsmax_example = model.input_example(max_batch=max_batch_size, max_dim=max_dim)[0]
    input_profile['tokens'] = [tuple(bs1_example.shape), tuple(bsmax_example.shape), tuple(bsmax_example.shape)]
    build_engine(
        f"{output_dir}/onnx/vit.onnx",
        f"{output_dir}/plan/vit.plan",
        fp16=(trt_precision in [16, '16', '16-mixed']),
        input_profile=input_profile,
        timing_cache=None,
        workspace_size=0,
    )


if __name__ == '__main__':
    main()
