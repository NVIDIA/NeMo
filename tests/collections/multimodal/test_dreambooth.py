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

import pytest
import torch
from nemo.collections.multimodal.models.text_to_image.dreambooth.dreambooth import MegatronDreamBooth
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer

from nemo.collections.nlp.parts.nlp_overrides import NLPDDPStrategy

DEVICE_CAPABILITY = None
if torch.cuda.is_available():
    DEVICE_CAPABILITY = torch.cuda.get_device_capability()


@pytest.fixture()
def model_cfg():
    model_cfg_string = """
      precision: ${trainer.precision}
      # specify micro_batch_size, global_batch_size, and model parallelism
      # gradient accumulation will be done automatically based on data_parallel_size
      micro_batch_size: 1 # limited by GPU memory
      global_batch_size: 1 # will use more micro batches to reach global batch size

      with_prior_preservation: False
      use_cached_latents: False
      prior_loss_weight: 0.5
      train_text_encoder: False
      restore_from_path: /ckpts/nemo-v1-5-188000-ema.nemo #This ckpt is only used to generate regularization images, thus .nemo ckpt is needed




      linear_start: 0.00085
      linear_end: 0.012
      num_timesteps_cond: 1
      log_every_t: 200
      timesteps: 1000
      first_stage_key: images
      cond_stage_key: captions
      image_size: 64
      channels: 4
      cond_stage_trainable: false
      conditioning_key: crossattn # check
      monitor: val/loss_simple_ema
      scale_factor: 0.18215
      use_ema: False
      scale_by_std: False
      ckpt_path:
      ignore_keys: [ ]
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
      channels_last: False

      unet_config:
        _target_: nemo.collections.multimodal.modules.stable_diffusion.diffusionmodules.openaimodel.UNetModel
        from_pretrained: #load unet weights for finetuning, can use .ckpt ckpts from various sources
        from_NeMo: False #Must be specified when from pretrained is not None, False means loading unet from HF ckpt
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
        use_flash_attention: False

      first_stage_config:
        _target_: nemo.collections.multimodal.models.stable_diffusion.ldm.autoencoder.AutoencoderKL
        from_pretrained:
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
          attn_resolutions: [ ]
          dropout: 0.0
        lossconfig:
          target: torch.nn.Identity

      cond_stage_config:
        #    For compatibility of history version that uses HF clip model
        _target_: nemo.collections.multimodal.modules.stable_diffusion.encoders.modules.FrozenCLIPEmbedder
        version: openai/clip-vit-large-patch14
        device: cuda
        max_length: 77

      noise_scheduler:
        _target_: nemo.collections.multimodal.models.dreambooth.util.sd_noise_scheduler
        parameterization: eps
        v_posterior: 0
        given_betas:
        beta_schedule: linear
        timesteps: 1000
        linear_start: 0.00085
        linear_end: 0.012
        cosine_s: 8e-3

      # miscellaneous
      seed: 1234
      resume_from_checkpoint: null # manually set the checkpoint file to load from
      apex_transformer_log_level: 30 # Python logging level displays logs with severity greater than or equal to this
      gradient_as_bucket_view: True # PyTorch DDP argument. Allocate gradients in a contiguous bucket to save memory (less fragmentation and buffer memory)

      optim:
        name: fused_adam
        lr: 1e-6
        weight_decay: 0.
        betas:
          - 0.9
          - 0.999
        sched:
          name: WarmupHoldPolicy
          warmup_steps: 1
          hold_steps: 10000000000000 # Incredibly large value to hold the lr as constant

      # Nsys profiling options
      nsys_profile:
        enabled: False
        start_step: 10  # Global batch to start profiling
        end_step: 10 # Global batch to end profiling
        ranks: [ 0 ] # Global rank IDs to profile
        gen_shape: False # Generate model and kernel details including input shapes

      data:
        num_workers: 4
        instance_dir: /opt/NeMo/docs/source/tools/images
        instance_prompt: a photo of a sks dog
        regularization_dir: 
        regularization_prompt: 
        num_reg_images: 10
        num_images_per_prompt: 4
        resolution: 512
        center_crop: True
        cached_instance_dir: #/datasets/instance_dir_cached
        cached_reg_dir: #/datasets/nemo_dogs_cached
    """
    model_cfg = OmegaConf.create(model_cfg_string)
    return model_cfg


@pytest.fixture()
def trainer_cfg():
    trainer_cfg_string = """
      devices: 1
      num_nodes: 1
      accelerator: gpu
      precision: bf16-mixed
      logger: False # logger provided by exp_manager
      enable_checkpointing: False
      use_distributed_sampler: False
      max_epochs: -1 # PTL default. In practice, max_steps will be reached first.
      max_steps: 400 # consumed_samples = global_step * micro_batch_size * data_parallel_size * accumulate_grad_batches
      log_every_n_steps: 10
      accumulate_grad_batches: 1 # do not modify, grad acc is automatic for training megatron models
      gradient_clip_val: 1.0
      benchmark: False
      enable_model_summary: True
      limit_val_batches: 0
    """
    trainer_cfg = OmegaConf.create(trainer_cfg_string)

    return trainer_cfg


@pytest.fixture()
def exp_manager_cfg():

    exp_manager_cfg_string = """
      exp_dir: null
      name: ${name}
      create_checkpoint_callback: True
      create_tensorboard_logger: True
      checkpoint_callback_params:
        every_n_train_steps: 200
        every_n_epochs: 0
        monitor: reduced_train_loss
        save_on_train_epoch_end: False
        filename: '${name}-{step}'
        save_top_k: -1
      resume_if_exists: True
      resume_ignore_no_checkpoint: True
      resume_from_checkpoint: ${model.resume_from_checkpoint}
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
    return 32


@pytest.fixture()
def dreambooth_trainer_and_model(model_cfg, trainer_cfg, precision, test_data_dir):
    model_cfg['precision'] = precision
    trainer_cfg['precision'] = precision
    model_cfg['data']['instance_dir'] = os.path.join(test_data_dir, "multimodal/tiny-dreambooth")
    strategy = NLPDDPStrategy()

    trainer = Trainer(strategy=strategy, **trainer_cfg)

    cfg = DictConfig(model_cfg)

    model = MegatronDreamBooth(cfg=cfg, trainer=trainer)

    def dummy():
        return

    if model.trainer.strategy.launcher is not None:
        model.trainer.strategy.launcher.launch(dummy, trainer=model.trainer)

    model.trainer.strategy.setup_environment()

    return trainer, model


@pytest.mark.run_only_on('GPU')
class TestMegatronDreamBooth:
    @pytest.mark.unit
    def test_constructor(self, dreambooth_trainer_and_model):
        dreambooth_model = dreambooth_trainer_and_model[1]
        assert isinstance(dreambooth_model, MegatronDreamBooth)

        num_weights = dreambooth_model.num_weights
        assert num_weights == 859520964

    @pytest.mark.parametrize(
        "precision",
        [
            32,
            16,
            pytest.param(
                "bf16",
                marks=pytest.mark.skipif(
                    not DEVICE_CAPABILITY or DEVICE_CAPABILITY[0] < 8,
                    reason='bfloat16 is not supported on this device',
                ),
            ),
        ],
    )
    @pytest.mark.unit
    def test_forward(self, dreambooth_trainer_and_model, test_data_dir, precision=None):
        trainer, dreambooth_model = dreambooth_trainer_and_model

        dtype = None
        if dreambooth_model.cfg['precision'] in [32, '32', '32-true']:
            dtype = torch.float
        elif dreambooth_model.cfg['precision'] in [16, '16', '16-mixed']:
            dtype = torch.float16
        elif dreambooth_model.cfg['precision'] in ['bf16', 'bf16-mixed']:
            dtype = torch.bfloat16
        else:
            raise ValueError(f"precision: {dreambooth_model.cfg['precision']} is not supported.")

        dreambooth_model = dreambooth_model.cuda()
        dreambooth_model.eval()

        images = torch.randn(1, 3, 512, 512).cuda()
        caption = [f'This is meaningless fake text']
        batch = images, dreambooth_model.model.text_encoder(caption)
        with torch.no_grad():
            loss = dreambooth_model(batch)

        assert loss.dtype == torch.float
