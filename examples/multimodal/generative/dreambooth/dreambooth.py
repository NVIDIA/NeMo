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
import pytorch_lightning as pl
import torch

from omegaconf import DictConfig, OmegaConf, open_dict
from pytorch_lightning import Trainer
from pytorch_lightning.plugins.environments import TorchElasticEnvironment
from pytorch_lightning.trainer.connectors.checkpoint_connector import CheckpointConnector

from nemo.collections.multimodal.models.dreambooth.dreambooth import MegatronDreamBooth
from nemo.collections.multimodal.models.multimodal_base_model import MegatronMultimodalModel
from nemo.collections.multimodal.models.stable_diffusion.ldm.ddpm import MegatronLatentDiffusion
from nemo.collections.multimodal.parts.stable_diffusion.pipeline import pipeline
from nemo.collections.multimodal.parts.utils import setup_trainer_and_model_for_inference
from nemo.collections.nlp.parts.nlp_overrides import (
    GradScaler,
    MegatronHalfPrecisionPlugin,
    NLPDDPStrategy,
    PipelineMixedPrecisionPlugin,
)
from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.exp_manager import exp_manager

try:
    from megatron.core import parallel_state

    HAVE_MEGATRON_CORE = True

except (ImportError, ModuleNotFoundError):

    HAVE_MEGATRON_CORE = False


def prepare_reg_data(cfg):
    reg_dir = cfg.model.data.regularization_dir
    num_reg_images = cfg.model.data.num_reg_images
    num_images_per_prompt = cfg.model.data.num_images_per_prompt
    reg_prompt = cfg.model.data.regularization_prompt
    os.makedirs(reg_dir, exist_ok=True)
    NUM_REG_IMAGES = len(os.listdir(reg_dir))
    if NUM_REG_IMAGES < num_reg_images:

        def model_cfg_modifier(model_cfg):
            model_cfg.precision = cfg.trainer.precision
            model_cfg.ckpt_path = None
            model_cfg.inductor = False
            model_cfg.unet_config.use_flash_attention = False
            model_cfg.micro_batch_size = cfg.model.micro_batch_size
            model_cfg.global_batch_size = cfg.model.global_batch_size
            model_cfg.unet_config.from_pretrained = None
            model_cfg.first_stage_config.from_pretrained = None
            model_cfg.target = 'nemo.collections.multimodal.models.stable_diffusion.ldm.ddpm.MegatronLatentDiffusion'

        trainer, megatron_diffusion_model = setup_trainer_and_model_for_inference(
            model_provider=MegatronLatentDiffusion, cfg=cfg, model_cfg_modifier=model_cfg_modifier
        )
        model = megatron_diffusion_model.model
        rng = torch.Generator()
        rng.manual_seed(trainer.global_rank * 100 + cfg.model.seed)
        images_to_generate = cfg.model.data.num_reg_images - NUM_REG_IMAGES
        images_to_generate = images_to_generate // trainer.world_size

        logging.info(f"No enough images in regularization folder, generating {images_to_generate} from provided ckpt")

        for i in range(images_to_generate // num_images_per_prompt + 1):
            output = pipeline(model, cfg, verbose=False, rng=rng)
            for text_prompt, pils in zip(reg_prompt, output):
                for idx, image in enumerate(pils):
                    image.save(
                        os.path.join(
                            cfg.infer.out_path,
                            f'{reg_prompt}_{trainer.global_rank}_{NUM_REG_IMAGES + i * num_images_per_prompt + idx}.png',
                        )
                    )
        del model
        del trainer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


@hydra_runner(config_path='conf', config_name='dreambooth.yaml')
def main(cfg):
    logging.info("\n\n************** Experiment configuration ***********")
    logging.info(f'\n{OmegaConf.to_yaml(cfg)}')

    megatron_amp_O2 = cfg.model.get('megatron_amp_O2', False)
    with_distributed_adam = cfg.model.optim.get('name') == 'distributed_fused_adam'

    torch.backends.cuda.matmul.allow_tf32 = True

    plugins = []
    strategy = NLPDDPStrategy(
        no_ddp_communication_hook=True,  # we don't use DDP for async grad allreduce
        gradient_as_bucket_view=cfg.model.gradient_as_bucket_view,
        find_unused_parameters=False,
    )

    if cfg.trainer.precision in [16, 'bf16']:
        scaler = None
        if cfg.trainer.precision == 16:
            scaler = GradScaler(
                init_scale=cfg.model.get('native_amp_init_scale', 2 ** 32),
                growth_interval=cfg.model.get('native_amp_growth_interval', 1000),
                hysteresis=cfg.model.get('hysteresis', 2),
            )
        if megatron_amp_O2 and not with_distributed_adam:
            plugins.append(MegatronHalfPrecisionPlugin(precision=cfg.trainer.precision, device='cuda', scaler=scaler))
        else:
            plugins.append(PipelineMixedPrecisionPlugin(precision=cfg.trainer.precision, device='cuda', scaler=scaler))

    if cfg.get('cluster_type', None) == 'BCP':
        plugins.append(TorchElasticEnvironment())

    if cfg.model.with_prior_preservation:
        prepare_reg_data(cfg)
    parallel_state.destroy_model_parallel()

    callbacks = []
    trainer = Trainer(plugins=plugins, strategy=strategy, callbacks=callbacks, **cfg.trainer)

    exp_manager(trainer, cfg.exp_manager)
    # update resume from checkpoint found by exp_manager
    if cfg.model.get("resume_from_checkpoint") is not None:
        resume_from_checkpoint = cfg.model.resume_from_checkpoint
    else:
        resume_from_checkpoint = trainer._checkpoint_connector.resume_from_checkpoint_fit_path

    logging.info(f'Resuming training from checkpoint: {resume_from_checkpoint}')

    trainer._checkpoint_connector = CheckpointConnector(trainer, resume_from_checkpoint=resume_from_checkpoint)

    # hydra interpolation does not work here as the interpolation key is lost when PTL saves hparams
    with open_dict(cfg):
        cfg.model.precision = cfg.trainer.precision

    model = MegatronDreamBooth(cfg.model, trainer)

    trainer.fit(model)


if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    main()
