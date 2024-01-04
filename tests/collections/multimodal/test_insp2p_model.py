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

import pytest
import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer

from nemo.collections.multimodal.data.stable_diffusion.stable_diffusion_dataset import build_train_valid_datasets
from nemo.collections.multimodal.models.instruct_pix2pix.ldm.ddpm_edit import MegatronLatentDiffusionEdit
from nemo.collections.nlp.parts.nlp_overrides import NLPDDPStrategy

DEVICE_CAPABILITY = None
if torch.cuda.is_available():
    DEVICE_CAPABILITY = torch.cuda.get_device_capability()


@pytest.fixture()
def model_cfg():
    model_cfg_string = """
    precision: 16
    # specify micro_batch_size, global_batch_size, and model parallelism
    # gradient accumulation will be done automatically based on data_parallel_size
    ckpt_path: null # load checkpoint weights from previous stages for fine-tuning
    micro_batch_size: 1
    global_batch_size: 1 # `= micro_batch_size * total_devices` fake global batch size for sampler

    linear_start: 0.00085
    linear_end: 0.012
    num_timesteps_cond: 1
    log_every_t: 200
    timesteps: 1000
    first_stage_key: edited
    cond_stage_key: edit # txt for cifar, caption for pbss
    image_size: 32
    channels: 4
    cond_stage_trainable: false
    conditioning_key: hybrid
    monitor: val/loss_simple_ema
    scale_factor: 0.18215
    use_ema: False
    scale_by_std: False

    ignore_keys: []
    parameterization: eps
    clip_denoised: True
    load_only_unet: False
    cosine_s: 8e-3
    given_betas:
    original_elbo_weight: 0
    v_posterior: 0
    l_simple_weight: 1
    use_positional_encodings: False
    learn_logvar: False
    logvar_init: 0
    beta_schedule: linear
    loss_type: l2

    concat_mode: True
    cond_stage_forward:
    text_embedding_dropout_rate: 0
    fused_opt: True
    inductor: False
    inductor_cudagraphs: False

    unet_config:
        _target_: nemo.collections.multimodal.modules.stable_diffusion.diffusionmodules.openaimodel.UNetModel
        from_pretrained: 
        image_size: 32 # unused
        in_channels: 8
        out_channels: 4
        model_channels: 320
        attention_resolutions:
        - 4
        - 2
        - 1
        num_res_blocks: 2
        channel_mult:
        - 1
        - 2
        - 4
        - 4
        num_heads: 8
        use_spatial_transformer: true
        transformer_depth: 1
        context_dim: 768
        use_checkpoint: False
        legacy: False
        use_flash_attention: False

    first_stage_config:
        _target_: nemo.collections.multimodal.models.stable_diffusion.ldm.autoencoder.AutoencoderKL
        from_pretrained: 
        embed_dim: 4
        monitor: val/rec_loss
        ddconfig:
            double_z: true
            z_channels: 4
            resolution: 256
            in_channels: 3
            out_ch: 3
            ch: 128
            ch_mult:
            - 1
            - 2
            - 4
            - 4
            num_res_blocks: 2
            attn_resolutions: []
            dropout: 0.0
        lossconfig:
            target: torch.nn.Identity

    cond_stage_config:
        _target_: nemo.collections.multimodal.modules.stable_diffusion.encoders.modules.FrozenCLIPEmbedder
        version: openai/clip-vit-large-patch14
        device: cuda
        max_length: 77


    # miscellaneous
    seed: 1234
    resume_from_checkpoint: null # manually set the checkpoint file to load from
    apex_transformer_log_level: 30 # Python logging level displays logs with severity greater than or equal to this
    gradient_as_bucket_view: True # PyTorch DDP argument. Allocate gradients in a contiguous bucket to save memory (less fragmentation and buffer memory)

    optim:
        name: fused_adam
        lr: 1e-4
        weight_decay: 0.
        betas:
        - 0.9
        - 0.999
        sched:
            name: WarmupHoldPolicy
            warmup_steps: 10000
            hold_steps: 10000000000000 # Incredibly large value to hold the lr as constant
            capturable: True
            master_weights: True
            max_norm: 1

    # Nsys profiling options
    nsys_profile:
        enabled: False
        start_step: 10  # Global batch to start profiling
        end_step: 10 # Global batch to end profiling
        ranks: [ 0 ] # Global rank IDs to profile
        gen_shape: False # Generate model and kernel details including input shapes

    data:
        # Path to instruct-pix2pix dataset must be specified by the user.
        # https://github.com/timothybrooks/instruct-pix2pix#generated-dataset
        data_path: /lustre/fsw/joc/yuya/stable_diffusion/instruct-pix2pix/data/tiny-ip2p
        num_workers: 2
        dataloader_type: cyclic # cyclic
        validation_drop_last: True # Set to false if the last partial validation samples is to be consumed

    """
    model_cfg = OmegaConf.create(model_cfg_string)
    return model_cfg


@pytest.fixture()
def trainer_cfg():
    trainer_cfg_string = """
        devices: 1 # number of GPUs (0 for CPU), or list of the GPUs to use e.g. [0, 1]
        num_nodes: 1
        max_epochs: -1
        max_steps: 10 # precedence over max_epochs
        logger: False  # Provided by exp_manager 
        precision: 16 # Should be set to 16 for O1 and O2 to enable the AMP.
        accelerator: gpu
        limit_val_batches: 0
        log_every_n_steps: 5  # Interval of logging.
        num_sanity_val_steps: 10 # number of steps to perform validation steps for sanity check the validation process before starting the training, setting to 0 disables it
        enable_checkpointing: False # Provided by exp_manager
        accumulate_grad_batches: 1 # do not modify, grad acc is automatic for training megatron models
        gradient_clip_val: 1.0
        benchmark: False
        enable_model_summary: True
    """
    trainer_cfg = OmegaConf.create(trainer_cfg_string)

    return trainer_cfg


@pytest.fixture()
def exp_manager_cfg():
    exp_manager_cfg_string = """
        explicit_log_dir: null
        exp_dir: null
        name: megatron_clip
        create_wandb_logger: False
        wandb_logger_kwargs:
          project: null
          name: null
        resume_if_exists: False
        resume_ignore_no_checkpoint: True
        create_checkpoint_callback: False
        checkpoint_callback_params:
            every_n_train_steps: 1000
            every_n_epochs: 0
            monitor: reduced_train_loss
            filename: '${name}--{reduced_train_loss:.2f}-{step}-{consumed_samples}'
        ema:
          enable: False
          decay: 0.9999
          validate_original_weights: False
          every_n_steps: 1
          cpu_offload: False
    """
    exp_manager_cfg = OmegaConf.create(exp_manager_cfg_string)

    return exp_manager_cfg


@pytest.fixture()
def precision():
    return 16


@pytest.fixture()
def sd_trainer_and_model(model_cfg, trainer_cfg, precision):
    model_cfg['precision'] = precision
    trainer_cfg['precision'] = precision

    strategy = NLPDDPStrategy()

    trainer = Trainer(strategy=strategy, **trainer_cfg)

    cfg = DictConfig(model_cfg)

    model = MegatronLatentDiffusionEdit(cfg=cfg, trainer=trainer)

    def dummy():
        return

    if model.trainer.strategy.launcher is not None:
        model.trainer.strategy.launcher.launch(dummy, trainer=model.trainer)
    model.trainer.strategy.setup_environment()

    return trainer, model


@pytest.mark.run_only_on('GPU')
class TestMegatronSDModel:
    @pytest.mark.unit
    def test_constructor(self, sd_trainer_and_model):
        sd_model = sd_trainer_and_model[1]
        assert isinstance(sd_model, MegatronLatentDiffusionEdit)

        num_weights = sd_model.num_weights
        assert num_weights == 859532484

    @pytest.mark.unit
    def test_build_dataset(self, sd_trainer_and_model, test_data_dir):
        sd_model = sd_trainer_and_model[1]
        sd_model.cfg.data.data_path = os.path.join(test_data_dir, "multimodal/tiny-ip2p")
        sd_model.build_train_valid_test_datasets()
        train_ds, validation_ds = sd_model._train_ds, sd_model._validation_ds
        print(len(train_ds), len(validation_ds))
        assert len(train_ds) == 205
        assert len(validation_ds) == 8
        sample = next(iter(train_ds))
        print(sample.keys(), sample['edit'].keys())
        assert "edit" in sample and "edited" in sample
        assert "c_concat" in sample["edit"] and "c_crossattn" in sample["edit"]

    @pytest.mark.unit
    def test_forward(self, sd_trainer_and_model, test_data_dir, precision=None):
        trainer, sd_model = sd_trainer_and_model

        dtype = None
        if sd_model.cfg['precision'] in [32, '32', '32-true']:
            dtype = torch.float
        elif sd_model.cfg['precision'] in [16, '16', '16-mixed']:
            dtype = torch.float16
        elif sd_model.cfg['precision'] in ['bf16', 'bf16-mixed']:
            dtype = torch.bfloat16
        else:
            raise ValueError(f"precision: {sd_model.cfg['precision']} is not supported.")

        sd_model.cfg.data.data_path = os.path.join(test_data_dir, "multimodal/tiny-ip2p")
        sd_model.build_train_valid_test_datasets()
        train_ds, validation_ds = sd_model._train_ds, sd_model._validation_ds

        train_loader = torch.utils.data.DataLoader(train_ds, batch_size=2)
        batch = next(iter(train_loader))

        sd_model = sd_model.to('cuda')
        batch['edited'] = batch['edited'].cuda()
        x, c = sd_model.model.get_input(batch, 'edited')
        with torch.no_grad():
            with torch.autocast('cuda', dtype=dtype):
                loss = sd_model(x, c)
                assert len(loss) == 2
