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


import torch.multiprocessing as mp
import torch.autograd.profiler
from omegaconf.omegaconf import OmegaConf, open_dict
from pytorch_lightning import Trainer
from pytorch_lightning.plugins.environments import TorchElasticEnvironment
from pytorch_lightning.trainer.connectors.checkpoint_connector import CheckpointConnector
# import nvidia_dlprof_pytorch_nvtx as nvtx

from nemo.collections.nlp.models.language_modeling.megatron_gpt_model import MegatronGPTModel
from nemo.collections.nlp.parts.nlp_overrides import (
    GradScaler,
    MegatronHalfPrecisionPlugin,
    NLPDDPStrategy,
    LayerUnitTestStrategy,
    PipelineMixedPrecisionPlugin,
)
from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.exp_manager import exp_manager

mp.set_start_method("spawn", force=True)



@hydra_runner(config_path="conf", config_name="megatron_gpt_config")
def main(cfg) -> None:
    logging.info("\n\n************** Experiment configuration ***********")
    logging.info(f'\n{OmegaConf.to_yaml(cfg)}')

    megatron_amp_o2 = cfg.model.get('megatron_amp_O2', False)
    with_distributed_adam = cfg.model.optim.get('name') == 'distributed_fused_adam'

    # nvtx.init(enable_function_stack=True)
    with torch.autograd.profiler.emit_nvtx():

        plugins = []
        if cfg.model.get("parallelization_specs", None) is not None:
            strategy = LayerUnitTestStrategy(
                no_ddp_communication_hook=True,  # we don't use DDP for async grad allreduce
                gradient_as_bucket_view=cfg.model.gradient_as_bucket_view,
                find_unused_parameters=False,
                parallelization_specs=cfg.model.parallelization_specs,
                micro_batch_size=cfg.model.micro_batch_size,
            )
            logging.info("**Using LayerUnitTestStrategy**")
        else:
            strategy = NLPDDPStrategy(
                no_ddp_communication_hook=True,  # we don't use DDP for async grad allreduce
                gradient_as_bucket_view=cfg.model.gradient_as_bucket_view,
                find_unused_parameters=False,
            )
            logging.info("**Using NLPDDPStrategy**")

        if cfg.trainer.precision in [16, 'bf16']:
            scaler = None
            if cfg.trainer.precision == 16:
                scaler = GradScaler(
                    init_scale=cfg.model.get('native_amp_init_scale', 2 ** 32),
                    growth_interval=cfg.model.get('native_amp_growth_interval', 1000),
                    hysteresis=cfg.model.get('hysteresis', 2),
                )
            if megatron_amp_o2 and not with_distributed_adam:
                plugins.append(MegatronHalfPrecisionPlugin(precision=cfg.trainer.precision, device='cuda', scaler=scaler))
            else:
                plugins.append(PipelineMixedPrecisionPlugin(precision=cfg.trainer.precision, device='cuda', scaler=scaler))

        if cfg.get('cluster_type', None) == 'BCP':
            plugins.append(TorchElasticEnvironment())

        trainer = Trainer(plugins=plugins, strategy=strategy, **cfg.trainer)

        exp_manager(trainer, cfg.exp_manager)

        # update resume from checkpoint found by exp_manager
        if cfg.model.resume_from_checkpoint is not None:
            resume_from_checkpoint = cfg.model.resume_from_checkpoint
        else:
            resume_from_checkpoint = trainer._checkpoint_connector.resume_from_checkpoint_fit_path

        logging.info(f'Resuming training from checkpoint: {resume_from_checkpoint}')

        trainer._checkpoint_connector = CheckpointConnector(trainer, resume_from_checkpoint=resume_from_checkpoint)

        # hydra interpolation does not work here as the interpolation key is lost when PTL saves hparams
        with open_dict(cfg):
            cfg.model.precision = cfg.trainer.precision

        model = MegatronGPTModel(cfg.model, trainer)
        # i = 0
        # # for module in model.model:
        # #     logging.info(f"XXXXXXXXXXXX module {i}: {module}")
        # j = 0
        # logging.info(f"ZZZZZZZZZZZZ {model.model._language_model_key}")
        # logging.info(f"XXXXXXXXXXXX {model.model.language_model}")
        # for layer in model.model.language_model.encoder.layers:
        #     logging.info(f"YYYYYYYYYYYYY layer {j}: {layer}, layer_type: {layer.layer_type}")
        #     j += 1
        #     # i += 1

        import time
        s = time.time()
        trainer.fit(model)
        e = time.time()
        print(f'Train time: {(e-s):.2f}')

if __name__ == '__main__':
    main()
