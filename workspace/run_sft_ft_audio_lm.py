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

import torch.multiprocessing as mp
from omegaconf.omegaconf import OmegaConf, open_dict
from pytorch_lightning import Trainer
from pytorch_lightning.plugins.environments import TorchElasticEnvironment

from nemo.collections.multimodal.speechllm.models.speechllm_models import ModularAudioGPTLoRAModel
from nemo.collections.nlp.models.language_modeling.megatron_gpt_peft_models import MegatronGPTPEFTModel
from nemo.collections.nlp.models.language_modeling.megatron_gpt_sft_model import MegatronGPTSFTModel
from nemo.collections.nlp.parts.nlp_overrides import (
    GradScaler,
    MegatronHalfPrecisionPlugin,
    NLPDDPStrategy,
    NLPSaveRestoreConnector,
    PEFTSaveRestoreConnector,
    PipelineMixedPrecisionPlugin,
)
from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.exp_manager import exp_manager

mp.set_start_method("spawn", force=True)


@hydra_runner(config_path="../examples/multimodel/conf/speechllm", config_name="modularized_speech_gpt_config_tuning")
def main(cfg) -> None:
    logging.info("\n\n************** Experiment configuration ***********")
    logging.info(f"\n{OmegaConf.to_yaml(cfg)}")
    assert cfg.model.restore_from_path is not None
    megatron_amp_o2 = cfg.model.get("megatron_amp_O2", False)
    with_distributed_adam = False

    plugins = []
    strategy = NLPDDPStrategy(
        no_ddp_communication_hook=True,  # we don't use DDP for async grad allreduce
        gradient_as_bucket_view=cfg.model.gradient_as_bucket_view,
        find_unused_parameters=False,
    )
    if cfg.trainer.precision in [16, "bf16"]:
        scaler = None
        if cfg.trainer.precision == 16:
            scaler = GradScaler(
                init_scale=cfg.model.get("native_amp_init_scale", 2 ** 32),
                growth_interval=cfg.model.get("native_amp_growth_interval", 1000),
                hysteresis=cfg.model.get("hysteresis", 2),
                enabled=False
                if cfg.model.pipeline_model_parallel_size > 1
                else True,  # turn off the grad scale for pipeline parallel LM model
            )
        if megatron_amp_o2 and not with_distributed_adam:
            plugins.append(MegatronHalfPrecisionPlugin(precision=cfg.trainer.precision, device="cuda", scaler=scaler))
        else:
            plugins.append(PipelineMixedPrecisionPlugin(precision=cfg.trainer.precision, device="cuda", scaler=scaler))

    if cfg.get("cluster_type", None) == "BCP":
        plugins.append(TorchElasticEnvironment())

    trainer = Trainer(plugins=plugins, strategy=strategy, **cfg.trainer)
    exp_manager(trainer, cfg.exp_manager)

    if cfg.model.peft.restore_from_path:
        if cfg.model.peft.restore_from_path.endswith(".nemo"):
            peft_model_cfg = MegatronGPTPEFTModel.restore_from(
                restore_path=cfg.model.peft.restore_from_path, trainer=trainer, return_config=True,
            )
        elif cfg.model.peft.restore_from_hparams_path:  # not a .nemo model we expect a hparams.yaml file
            peft_model_cfg = OmegaConf.to_container(OmegaConf.load(cfg.model.peft.restore_from_hparams_path).cfg)
            peft_model_cfg = OmegaConf.create(peft_model_cfg)
            # extract dict inside cfg key and convert it to DictConfig
            # this allows interpolation to work the same way as config from the .restore_from method
        else:
            raise RuntimeError("This script requires a .nemo peft model or path to hparams.yaml (and a ckpt path).")
    else:
        peft_model_cfg = MegatronGPTSFTModel.restore_from(
            restore_path=cfg.model.restore_from_path, trainer=trainer, return_config=True,
        )

    # hydra interpolation does not work here as the interpolation key is lost when PTL saves hparams
    with open_dict(peft_model_cfg):
        # update the model config of the trained model with params we want to set at inference time.
        peft_model_cfg.precision = cfg.trainer.precision
        peft_model_cfg.global_batch_size = cfg.model.global_batch_size
        peft_model_cfg.micro_batch_size = cfg.model.micro_batch_size
        peft_model_cfg.data = cfg.model.data
        peft_model_cfg.optim = cfg.model.optim
        if peft_model_cfg.get("use_flash_attention", False):
            peft_model_cfg.use_flash_attention = cfg.model.use_flash_attention
        if cfg.model.get("seq_len_interpolation_factor", None) is not None:
            peft_model_cfg["seq_len_interpolation_factor"] = cfg.model.seq_len_interpolation_factor

    if cfg.model.peft.restore_from_path:
        if '\\' in cfg.model.peft.restore_from_path:
            cfg.model.peft.restore_from_path = cfg.model.peft.restore_from_path.replace('\\', '')

        if cfg.model.peft.restore_from_path.endswith(".nemo"):
            save_restore_connector = PEFTSaveRestoreConnector(
                peft_model_nemo_path=cfg.model.peft.restore_from_path, peft_model_ckpt_path=None,
            )
        else:
            # attempting to load a ckpt peft model.
            if cfg.model.peft.restore_from_ckpt_name:
                ckpt_name = cfg.model.peft.restore_from_ckpt_name
            else:
                ckpt_name = "model_weights.ckpt"
            save_restore_connector = PEFTSaveRestoreConnector(
                peft_model_nemo_path=None,
                peft_model_ckpt_path=cfg.model.peft.restore_from_path,
                peft_model_ckpt_name=ckpt_name,
            )
    else:
        save_restore_connector = NLPSaveRestoreConnector()

    if os.path.isdir(cfg.model.restore_from_path):
        save_restore_connector.model_extracted_dir = cfg.model.restore_from_path

    model = ModularAudioGPTLoRAModel.restore_from(
        restore_path=cfg.model.restore_from_path,
        trainer=trainer,
        override_config_path=peft_model_cfg,
        save_restore_connector=save_restore_connector,
    )

    trainer.fit(model)


if __name__ == '__main__':
    main()
