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
from nemo.collections.nlp.parts.nlp_overrides import NLPDDPStrategy
from nemo.collections.multimodal.models.generative.stable_diffusion.ldm.ddpm import MegatronLatentDiffusion
from nemo.collections.multimodal.data.stable_diffusion.stable_diffusion_dataset import build_train_valid_datasets

DEVICE_CAPABILITY = None
if torch.cuda.is_available():
    DEVICE_CAPABILITY = torch.cuda.get_device_capability()

@pytest.fixture()
def model_cfg():
    
    model_cfg_string = """
    precision: 16
    # specify micro_batch_size, global_batch_size, and model parallelism
    # gradient accumulation will be done automatically based on data_parallel_size
    micro_batch_size: 1 # limited by GPU memory
    global_batch_size: 1 # will use more micro batches to reach global batch size
    native_amp_init_scale: 65536.0 # Init scale for grad scaler used at fp16


    linear_start: 0.00085
    linear_end: 0.012
    num_timesteps_cond: 1
    log_every_t: 200
    timesteps: 1000
    first_stage_key: images
    cond_stage_key: captions # txt for cifar, caption for pbss
    image_size: 64
    channels: 4
    cond_stage_trainable: false
    conditioning_key: crossattn # check
    monitor: val/loss_simple_ema
    scale_factor: 0.18215
    use_ema: False
    scale_by_std: False
    ckpt_path:
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
    text_embedding_dropout_rate: 0.1
    fused_opt: True
    inductor: False
    inductor_cudagraphs: False
    capture_cudagraph_iters: -1 # -1 to disable
    channels_last: True

    unet_config:
        _target_: nemo.collections.multimodal.modules.stable_diffusion.diffusionmodules.openaimodel.UNetModel
        from_pretrained: #/ckpts/nemo-v1-2.ckpt
        from_NeMo: True #Must be specified when from pretrained is not None, False means loading unet from HF ckpt
        image_size: 32 # unused
        in_channels: 4
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
        use_flash_attention: True
        enable_amp_o2_fp16: True
        resblock_gn_groups: 32

    first_stage_config:
        _target_: nemo.collections.multimodal.models.stable_diffusion.ldm.autoencoder.AutoencoderKL
        from_pretrained: /ckpts/vae.bin
        embed_dim: 4
        monitor: val/rec_loss
        ddconfig:
            double_z: true
            z_channels: 4
            resolution: 256  #Never used
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
        capture_cudagraph_iters: -1

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
    ddp_overlap: True # True for using PyTorch DDP overlap.

    optim:
        name: megatron_fused_adam
        lr: null
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
        num_workers: 16
        synthetic_data: True # dataset_path and local_root_path can be empty when using synthetic data
        synthetic_data_length: 10000
        train:
            dataset_path:
                - /datasets/coyo/wdinfo.pkl
            augmentations:
                resize_smallest_side: 512
                center_crop_h_w: 512, 512
                horizontal_flip: False
            filterings:

        webdataset:
            infinite_sampler: False
            local_root_path: /datasets/coyo
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
    model_cfg['precision'] = precision
    trainer_cfg['precision'] = precision

    strategy = NLPDDPStrategy()

    trainer = Trainer(strategy=strategy, **trainer_cfg)

    cfg = DictConfig(model_cfg)

    model = MegatronLatentDiffusion(cfg=cfg, trainer=trainer)

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
        assert isinstance(sd_model, MegatronLatentDiffusion)

        num_weights = sd_model.num_weights
        assert num_weights == 859520964
    
    @pytest.mark.unit
    def test_build_dataset(self, sd_trainer_and_model, test_data_dir):
        sd_model = sd_trainer_and_model[1]
        train_ds, validation_ds = build_train_valid_datasets(
            model_cfg=sd_model.cfg, consumed_samples=0,
        )
        assert len(train_ds) == 100000
        assert validation_ds is None
        sample = next(iter(train_ds))
        assert "captions" in sample
        assert "images" in sample
    

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

        trainer_ds, _ = build_train_valid_datasets(model_cfg=sd_model.cfg, consumed_samples=0)

        train_loader = torch.utils.data.DataLoader(trainer_ds, batch_size=4)
        batch = next(iter(train_loader))

        
        sd_model = sd_model.to('cuda')
        batch['images'] = batch['images'].cuda()
        x, c = sd_model.model.get_input(batch, 'images')
        with torch.no_grad():
            with torch.autocast('cuda', dtype=dtype):
                loss = sd_model(x, c)
                assert len(loss) == 2
