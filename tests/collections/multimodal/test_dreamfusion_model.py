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
from hydra.utils import get_class, instantiate
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer

from nemo.collections.multimodal.models.text_to_3d.nerf import DreamFusion

DEVICE_CAPABILITY = None
if torch.cuda.is_available():
    DEVICE_CAPABILITY = torch.cuda.get_device_capability()


@pytest.fixture()
def model_cfg():
    model_cfg_string = """
        ### model options
        resume_from_checkpoint:
        prompt: 'a hamburger'
        negative_prompt: ''
        front_prompt: ', front view'
        side_prompt: ', side view'
        back_prompt: ', back view'
        update_extra_interval: 16
        guidance_scale: 100
        export_video: False

        iters: 1
        latent_iter_ratio: 0.2
        albedo_iter_ratio: 0.0
        min_ambient_ratio: 0.1
        textureless_ratio: 0.2

        data:
            _target_: examples.multimodal.generative.nerf.data.AggregatorDataModule

            train_batch_size: 1
            train_shuffle: false
            train_dataset:
                _target_: nemo.collections.multimodal.data.nerf.random_poses.RandomPosesDataset
                internal_batch_size: 100
                width: 64
                height: 64
                radius_range: [3.0, 3.5]
                theta_range: [45, 105]
                phi_range: [-180, 180]
                fovx_range: [10, 30]
                fovy_range: [10, 30]
                jitter: False
                jitter_center: 0.2
                jitter_target: 0.2
                jitter_up: 0.02
                uniform_sphere_rate: 0
                angle_overhead: 30
                angle_front: 60

            val_batch_size: 1
            val_shuffle: false
            val_dataset:
                _target_: nemo.collections.multimodal.data.nerf.circle_poses.CirclePosesDataset
                size: 5
                width: 800
                height: 800
                angle_overhead: 30
                angle_front: 60

            test_batch_size: 1
            test_shuffle: false
            test_dataset:
                _target_: nemo.collections.multimodal.data.nerf.circle_poses.CirclePosesDataset
                size: 100
                width: 800
                height: 800
                angle_overhead: 30
                angle_front: 60


        nerf:
            _target_: nemo.collections.multimodal.modules.nerf.geometry.torchngp_nerf.TorchNGPNerf
            num_input_dims: 3    # 3D space
            bound: 1
            density_activation: exp # softplus, exp
            blob_radius: 0.2
            blob_density: 5
            normal_type: central_finite_difference

            encoder_cfg:
                encoder_type: 'hashgrid'
                encoder_max_level:
                log2_hashmap_size: 19
                desired_resolution: 2048
                interpolation: smoothstep

            sigma_net_num_output_dims: 1    # density
            sigma_net_cfg:
                num_hidden_dims: 64
                num_layers: 3
                bias: True

            features_net_num_output_dims: 3   # rgb
            features_net_cfg:
                num_hidden_dims: 64
                num_layers: 3
                bias: True

        background:
            _target_: nemo.collections.multimodal.modules.nerf.background.static_background.StaticBackground
            background: [0, 0, 1] # rgb

        material:
            _target_: nemo.collections.multimodal.modules.nerf.materials.basic_shading.BasicShading

        renderer:
            _target_: nemo.collections.multimodal.modules.nerf.renderers.torchngp_volume_renderer.TorchNGPVolumeRenderer
            bound: 1
            update_interval: 16
            grid_resolution: 128
            density_thresh: 10
            max_steps: 1024
            dt_gamma: 0

        guidance:
            _target_: nemo.collections.multimodal.modules.nerf.guidance.stablediffusion_huggingface_pipeline.StableDiffusion
            precision: 16
            model_key: stabilityai/stable-diffusion-2-1-base
            t_range: [0.02, 0.98]

        optim:
            name: adan
            lr: 5e-3
            eps: 1e-8
            weight_decay: 2e-5
            max_grad_norm: 5.0
            foreach: False

        loss:
            lambda_sds: 1.0
            lambda_opacity: 0.0
            lambda_entropy: 1e-3
            lambda_orientation: 1e-2
            lambda_2d_normal_smooth: 0.0
            lambda_3d_normal_smooth: 0.0
            lambda_mesh_normal: 0.0
            lambda_mesh_laplacian: 0.0

    """
    model_cfg = OmegaConf.create(model_cfg_string)
    return model_cfg


@pytest.fixture()
def trainer_cfg():
    trainer_cfg_string = """
      devices: 1
      num_nodes: 1
      precision: 16
      max_steps: 10000 # example configs: dreamfuions=10000, dmtet=5000
      accelerator: gpu
      enable_checkpointing: False
      logger: False
      log_every_n_steps: 1
      val_check_interval: 100
      accumulate_grad_batches: 1
      benchmark: False
      enable_model_summary: True
    """
    trainer_cfg = OmegaConf.create(trainer_cfg_string)
    return trainer_cfg


@pytest.fixture()
def exp_manager_cfg():

    exp_manager_cfg_string = """
      name: dreamfusion-test
      exp_dir: /results
      create_tensorboard_logger: False
      create_wandb_logger: False
      wandb_logger_kwargs:
        project: dreamfusion
        group: nemo-df
        name: ${name}
        resume: True
      create_checkpoint_callback: True
      checkpoint_callback_params:
        every_n_epochs: 0
        every_n_train_steps: 1000
        monitor: loss
        filename: '${name}-{step}'
        save_top_k: -1
        always_save_nemo: False
      resume_if_exists: True
      resume_ignore_no_checkpoint: True

    """
    exp_manager_cfg = OmegaConf.create(exp_manager_cfg_string)
    return exp_manager_cfg


@pytest.fixture()
def precision():
    return 32


@pytest.fixture()
def dreamfusion_trainer_and_model(model_cfg, trainer_cfg, precision):
    # Trainer
    trainer_cfg['precision'] = precision
    trainer = Trainer(**trainer_cfg)

    # Model
    model_cfg = DictConfig(model_cfg)
    model_cfg['iters'] = trainer_cfg['max_steps']
    model_cfg['guidance']['precision'] = precision
    model = DreamFusion(cfg=model_cfg)

    datamodule = instantiate(model_cfg.data['val_dataset'])

    return trainer, model, datamodule


@pytest.mark.run_only_on('GPU')
class TestDreamFusion:
    @pytest.mark.unit
    def test_constructor(self, dreamfusion_trainer_and_model):
        trainer, model, datamodule = dreamfusion_trainer_and_model

        assert isinstance(model, DreamFusion)
        assert model.num_weights == 12209048

    @pytest.mark.unit
    def test_build_dataset(self, dreamfusion_trainer_and_model):
        trainer, model, datamodule = dreamfusion_trainer_and_model
        assert len(datamodule) == 5

    @pytest.mark.unit
    def test_forward(self, dreamfusion_trainer_and_model, test_data_dir, precision=None):
        trainer, model, datamodule = dreamfusion_trainer_and_model

        dtype = None
        if trainer.precision in [32, '32', '32-true']:
            dtype = torch.float
        elif trainer.precision in [16, '16', '16-mixed']:
            dtype = torch.float16
        elif trainer.precision in ['bf16', 'bf16-mixed']:
            dtype = torch.bfloat16
        else:
            raise ValueError(f"precision: {trainer.precision} is not supported.")

        model = model.cuda()
        batch = next(iter(datamodule))
        with torch.no_grad():
            with torch.autocast('cuda', dtype=dtype):
                outputs = model(
                    rays_o=batch['rays_o'].cuda(),
                    rays_d=batch['rays_d'].cuda(),
                    mvp=batch['mvp'].cuda(),
                    perturb=False,
                    ambient_ratio=batch['ambient_ratio'] if 'ambient_ratio' in batch else 1.0,
                    shading_type=batch['shading_type'] if 'shading_type' in batch else None,
                    binarize=False,
                    return_normal_image=False,
                    return_normal_perturb=False,
                    return_vertices=False,
                    return_faces=False,
                    return_faces_normals=False,
                )

        assert outputs['image'].dtype == dtype
        assert outputs['image'].shape == torch.Size([1, 800, 800, 3])
