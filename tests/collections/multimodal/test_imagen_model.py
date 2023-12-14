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

from nemo.collections.multimodal.data.imagen.imagen_dataset import build_train_valid_datasets
from nemo.collections.multimodal.models.text_to_image.imagen.imagen import DUMMY_TENSOR, MegatronImagen
from nemo.collections.nlp.parts.nlp_overrides import NLPDDPStrategy

DEVICE_CAPABILITY = None
if torch.cuda.is_available():
    DEVICE_CAPABILITY = torch.cuda.get_device_capability()


@pytest.fixture()
def model_cfg():

    model_cfg_string = """
        precision: 16
        micro_batch_size: 2 # limited by GPU memory
        global_batch_size: 2 # will use more micro batches to reach global batch size
        inductor: False
        inductor_cudagraphs: False
        unet_type: base
        channels_last: True

        unet:
            embed_dim: 256
            image_size: 64
            channels: 3
            num_res_blocks: 3
            channel_mult: [ 1, 2, 3, 4 ]
            num_attn_heads: 4
            per_head_channels: 64
            cond_dim: 512
            attention_type: fused
            feature_pooling_type: attention
            learned_sinu_pos_emb_dim: 0
            attention_resolutions: [ 8, 16, 32 ]
            dropout: False
            use_null_token: False
            init_conv_kernel_size: 3
            gradient_checkpointing: False
            scale_shift_norm: True
            stable_attention: False
            flash_attention: True
            resblock_updown: False
            resample_with_conv: True

        # miscellaneous
        seed: 1234
        resume_from_checkpoint: null # manually set the checkpoint file to load from
        apex_transformer_log_level: 30 # Python logging level displays logs with severity greater than or equal to this
        gradient_as_bucket_view: True # PyTorch DDP argument. Allocate gradients in a contiguous bucket to save memory (less fragmentation and buffer memory)
        ddp_overlap: False # True for using PyTorch default DDP overlap. False for using Megatron's default configuration for async grad allreduce

        preconditioning_type: EDM
        preconditioning:
            loss_type: l2
            sigma_data: 0.5
            p_mean: -1.2
            p_std: 1.2

        conditioning:
            embed_dim: 1024
            token_length: 128
            drop_rate: 0.1
            precached_key: embeddings_t5_xxl
            out_key: t5_text

        data:
            num_workers: 16
            synthetic_data: True
            synthetic_data_length: 800000
            train:
                augmentations:
                    resize_smallest_side: 64
                    center_crop_h_w: 64, 64
                    horizontal_flip: False
                filterings: null

            webdataset:
                use_webdataset: True
                object_store: False
                infinite_sampler: False
                local_root_path: /datasets
                verbose: False

        optim:
            # We need weight decay for large-scale odel
            name: fused_adam
            lr: 0.0001
            eps: 1e-8
            betas: [ 0.9, 0.999 ]
            weight_decay: 0.01
            sched:
                name: WarmupPolicy
                warmup_steps: 10000
                warmup_ratio: null
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
          monitor: reduced_train_loss
          save_top_k: 5
          mode: min
          always_save_nemo: False # saves nemo file during validation, not implemented for model parallel
          save_nemo_on_train_end: False # not recommended when training large models on clusters with short time limits
          filename: '${name}--{reduced_train_loss:.2f}-{step}-{consumed_samples}'
          model_parallel_size: 1
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
def imagen_trainer_and_model(model_cfg, trainer_cfg, precision):
    model_cfg['precision'] = precision
    model_cfg['precision'] = precision
    trainer_cfg['precision'] = precision

    strategy = NLPDDPStrategy()

    trainer = Trainer(strategy=strategy, **trainer_cfg)

    cfg = DictConfig(model_cfg)

    model = MegatronImagen(cfg=cfg, trainer=trainer)

    def dummy():
        return

    if model.trainer.strategy.launcher is not None:
        model.trainer.strategy.launcher.launch(dummy, trainer=model.trainer)
    model.trainer.strategy.setup_environment()

    return trainer, model


@pytest.mark.run_only_on('GPU')
class TestMegatronImagenModel:
    @pytest.mark.unit
    def test_constructor(self, imagen_trainer_and_model):
        imagen_model = imagen_trainer_and_model[1]
        assert isinstance(imagen_model, MegatronImagen)

        num_weights = imagen_model.num_weights
        assert num_weights == 524897540

    @pytest.mark.unit
    def test_build_dataset(self, imagen_trainer_and_model, test_data_dir):
        imagen_model = imagen_trainer_and_model[1]
        train_ds, validation_ds = build_train_valid_datasets(model_cfg=imagen_model.cfg, consumed_samples=0,)
        assert len(train_ds) == 800000
        assert validation_ds is None
        sample = next(iter(train_ds))
        assert "t5_text_embeddings" in sample
        assert "t5_text_mask" in sample
        assert "images" in sample

    @pytest.mark.unit
    def test_forward(self, imagen_trainer_and_model, test_data_dir, precision=None):
        trainer, imagen_model = imagen_trainer_and_model

        dtype = None
        if imagen_model.cfg['precision'] in [32, '32', '32-true']:
            dtype = torch.float
        elif imagen_model.cfg['precision'] in [16, '16', '16-mixed']:
            dtype = torch.float16
        elif imagen_model.cfg['precision'] in ['bf16', 'bf16-mixed']:
            dtype = torch.bfloat16
        else:
            raise ValueError(f"precision: {imagen_model.cfg['precision']} is not supported.")

        trainer_ds, _ = build_train_valid_datasets(model_cfg=imagen_model.cfg, consumed_samples=0)

        train_loader = torch.utils.data.DataLoader(trainer_ds, batch_size=4)
        batch = next(iter(train_loader))

        imagen_model = imagen_model.to('cuda')
        input_args = {
            'x_start': batch['images'].cuda(),
            'text_embed': batch['t5_text_embeddings'].cuda(),
            'text_mask': batch['t5_text_mask'].cuda(),
            'x_lowres': DUMMY_TENSOR.repeat(batch['images'].shape[0]).cuda(),
        }
        with torch.no_grad():
            with torch.autocast('cuda', dtype=dtype):
                loss = imagen_model.model(**input_args)
                assert len(loss) == 2
