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

import os

import pytest
import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer

from nemo.collections.multimodal.data.clip.clip_dataset import build_train_valid_datasets
from nemo.collections.multimodal.models.clip.megatron_clip_models import (
    CLIPModel,
    CLIPTextTransformer,
    CLIPVisionTransformer,
    MegatronCLIPModel,
)
from nemo.collections.nlp.modules.common.megatron.megatron_init import initialize_model_parallel_for_nemo
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
      tensor_model_parallel_size: 1 # intra-layer model parallelism
      pipeline_model_parallel_size: 1 # inter-layer model parallelism
      virtual_pipeline_model_parallel_size: null # interleaved pipeline
    
      restore_from_pretrained: null # used in fine-tuning
      # multimodal configs
      output_dim: 64
      local_loss: False # calculate loss w/ local features @ global (instead of realizing full global @ global matrix)
      gather_with_grad: True # enable full distributed gradient for feature gather, set this to False may cause convergence issue
    
      vision:
        precision: 16
        # vision configs
        patch_dim: 16
        img_h: 224
        img_w: 224
        image_mean: null
        image_std: null
        num_channels: 3
        drop_patch_rate: 0.0
        drop_path_rate: 0.0
        global_average_pool: False
        output_dim: 64
        class_token_length: 8
        preprocess_layernorm: True # apply layer norm to embedded tokens
    
        # model architecture
        encoder_seq_length: 196
        max_position_embeddings: 196
        position_embedding_type: learned_absolute
        num_layers: 2
        hidden_size: 768
        ffn_hidden_size: 3072 # Transformer FFN hidden size. Usually 4 * hidden_size.
        num_attention_heads: 12
        init_method_std: 0.02 # Standard deviation of the zero mean normal distribution used for weight initialization.')
        use_scaled_init_method: True # use scaled residuals initialization
        hidden_dropout: 0. # Dropout probability for hidden state transformer.
        attention_dropout: 0.
        kv_channels: null # Projection weights dimension in multi-head attention. Set to hidden_size // num_attention_heads if null
        apply_query_key_layer_scaling: True # scale Q * K^T by 1 / layer-number.
        normalization: layernorm # Type of normalization layers
        layernorm_epsilon: 1e-5
        do_layer_norm_weight_decay: False # True means weight decay on all params
        pre_process: True # add embedding
        post_process: True # add pooler
        persist_layer_norm: True # Use of persistent fused layer norm kernel.
    
        ## Activation Checkpointing
        activations_checkpoint_granularity: null # 'selective' or 'full'
        activations_checkpoint_method: null # 'uniform', 'block', not used with 'selective'
        activations_checkpoint_num_layers: null # not used with 'selective'
        sequence_parallel: False
    
        # precision
        native_amp_init_scale: 4294967296 # 2 ** 32
        native_amp_growth_interval: 1000
        hysteresis: 2 # Gradient scale hysteresis
        fp32_residual_connection: False # Move residual connections to fp32
        fp16_lm_cross_entropy: False # Move the cross entropy unreduced loss calculation for lm head to fp16
    
        # model fusions
        masked_softmax_fusion: True # Use a kernel that fuses the attention softmax with it's mask.
        bias_dropout_add_fusion: True # Use a kernel that fuses the bias addition, dropout and residual connection addition.
    
        use_cpu_initialization: False # Init weights on the CPU (slow for large models)
        onnx_safe: False # Use work-arounds for known problems with Torch ONNX exporter.
        gradient_accumulation_fusion: False # Fuse weight gradient accumulation to GEMMs. Only used with pipeline parallelism.
        openai_gelu: False
        bias_activation_fusion: False
        megatron_legacy: False
    
    
      text:
        precision: 16
        # text configs
        output_dim: 64
    
        # model architecture
        encoder_seq_length: 77
        max_position_embeddings: 77
        position_embedding_type: learned_absolute
        num_layers: 2
        hidden_size: 512
        ffn_hidden_size: 2048 # Transformer FFN hidden size. Usually 4 * hidden_size.
        num_attention_heads: 8
        init_method_std: 0.02 # Standard deviation of the zero mean normal distribution used for weight initialization.')
        use_scaled_init_method: True # use scaled residuals initialization
        hidden_dropout: 0. # Dropout probability for hidden state transformer.
        attention_dropout: 0.
        kv_channels: null # Projection weights dimension in multi-head attention. Set to hidden_size // num_attention_heads if null
        apply_query_key_layer_scaling: True # scale Q * K^T by 1 / layer-number.
        normalization: layernorm # Type of normalization layers
        layernorm_epsilon: 1e-5
        do_layer_norm_weight_decay: False # True means weight decay on all params
        pre_process: True # add embedding
        post_process: True # add pooler
        persist_layer_norm: True # Use of persistent fused layer norm kernel.
    
        ## Activation Checkpointing
        activations_checkpoint_granularity: null # 'selective' or 'full'
        activations_checkpoint_method: null # 'uniform', 'block', not used with 'selective'
        activations_checkpoint_num_layers: null # not used with 'selective'
        num_micro_batches_with_partial_activation_checkpoints: null
        activations_checkpoint_layers_per_pipeline: null
        sequence_parallel: False
    
        # precision
        native_amp_init_scale: 4294967296 # 2 ** 32
        native_amp_growth_interval: 1000
        hysteresis: 2 # Gradient scale hysteresis
        fp32_residual_connection: False # Move residual connections to fp32
        fp16_lm_cross_entropy: False # Move the cross entropy unreduced loss calculation for lm head to fp16
    
        # model fusions
        masked_softmax_fusion: True # Use a kernel that fuses the attention softmax with it's mask.
        bias_dropout_add_fusion: True # Use a kernel that fuses the bias addition, dropout and residual connection addition.
    
        use_cpu_initialization: False # Init weights on the CPU (slow for large models)
        onnx_safe: False # Use work-arounds for known problems with Torch ONNX exporter.
        gradient_accumulation_fusion: False # Fuse weight gradient accumulation to GEMMs. Only used with pipeline parallelism.
        openai_gelu: False
        bias_activation_fusion: False
        megatron_legacy: False
    
        transformer_engine: False
        fp8: False # enables fp8 in TransformerLayer forward
        fp8_e4m3: False # sets fp8_format = recipe.Format.E4M3
        fp8_hybrid: False # sets fp8_format = recipe.Format.HYBRID
        fp8_margin: 0 # scaling margin
        fp8_interval: 1 # scaling update interval
        fp8_amax_history_len: 1 # Number of steps for which amax history is recorded per tensor
        fp8_amax_compute_algo: most_recent # 'most_recent' or 'max'. Algorithm for computing amax from history
        use_emha: False # Use fused multi-head attention for large sequence-length. Note this is not yet supported. Please set to False.
    
      # Megatron O2-style half-precision
      megatron_amp_O2: False # Enable O2-level automatic mixed precision using main parameters
      grad_allreduce_chunk_size_mb: 125
      grad_div_ar_fusion: True # Fuse grad division into torch.distributed.all_reduce
    
      # miscellaneous
      seed: 1234
      resume_from_checkpoint: null # manually set the checkpoint file to load from
      apex_transformer_log_level: 30 # Python logging level displays logs with severity greater than or equal to this
      gradient_as_bucket_view: True # PyTorch DDP argument. Allocate gradients in a contiguous bucket to save memory (less fragmentation and buffer memory)
    
      tokenizer:
        library: 'huggingface'
        type: 'openai/clip-vit-large-patch14'
        model: null
        vocab_file: null
        merge_file: null
        delimiter: null # only used for tabular tokenizer
        sentencepiece_legacy: False # Legacy=True allows you to add special tokens to sentencepiece tokenizers.
      make_vocab_size_divisible_by: 128 # Pad the vocab size to be divisible by this value for computation efficiency.
    
      data:
        num_workers: 1
        dataset_type: webdataset
    
        train:
          data_path: # List of paths to pkl files or tar files
            - /lustre/fsw/joc/multimodal/datasets/cc3m/00000-00008_{000000..000001}.tar
          drop_last: True # drop_last = False is not implemented yet
        validation: # List of paths to pkl files or tar files
          data_path:
            - /lustre/fsw/joc/multimodal/datasets/cc3m/00000-00008_000002.tar
          drop_last: True # drop_last = False is not implemented yet
        webdataset:
          object_store: False
          bucket: datasets
          pbss_credentials_file: pbss_credential
          local_root_path: / # tar files local root path
          chunk_size: 1000 # if data path is list of tar files, chunk_size needs to be provided
    
        imagenet_val: null # Path to imagenet val set for conducting zero shot evaluation.
    
      # Nsys profiling options
      nsys_profile:
        enabled: False
        start_step: 10  # Global batch to start profiling
        end_step: 10 # Global batch to end profiling
        ranks: [ 0 ] # Global rank IDs to profile
        gen_shape: False # Generate model and kernel details including input shapes
    
      optim:
        name: fused_adam
        lr: 1e-3
        weight_decay: 0.2
        betas:
          - 0.9
          - 0.98
        sched:
          name: CosineAnnealing
          warmup_steps: 2000
          constant_steps: 0
          min_lr: 1e-5
    """
    model_cfg = OmegaConf.create(model_cfg_string)
    return model_cfg


@pytest.fixture()
def trainer_cfg():

    trainer_cfg_string = """
        devices: 1
        num_nodes: 1
        accelerator: gpu
        precision: 16
        logger: False
        enable_checkpointing: False
        use_distributed_sampler: False
        max_epochs: -1
        max_steps: 4
        log_every_n_steps: 1
        val_check_interval: 4
        limit_val_batches: 2
        limit_test_batches: 2
        accumulate_grad_batches: 1
        gradient_clip_val: 1.0
        benchmark: False
        enable_model_summary: False
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
          monitor: val_loss
          save_top_k: 10
          mode: min
          always_save_nemo: False # saves nemo file during validation, not implemented for model parallel
          save_nemo_on_train_end: False # not recommended when training large models on clusters with short time limits
          filename: 'megatron_vit_classify--{val_loss:.2f}-{step}-{consumed_samples}'
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
    return 32


@pytest.fixture()
def clip_trainer_and_model(model_cfg, trainer_cfg, precision):
    model_cfg['vision']['precision'] = precision
    model_cfg['text']['precision'] = precision
    trainer_cfg['precision'] = precision

    strategy = NLPDDPStrategy()

    trainer = Trainer(strategy=strategy, **trainer_cfg)

    cfg = DictConfig(model_cfg)

    model = MegatronCLIPModel(cfg=cfg, trainer=trainer)

    def dummy():
        return

    if model.trainer.strategy.launcher is not None:
        model.trainer.strategy.launcher.launch(dummy, trainer=model.trainer)
    model.trainer.strategy.setup_environment()

    return trainer, model


def build_datasets(cfg, tokenizer):
    return build_train_valid_datasets(model_cfg=cfg, consumed_samples=0, tokenizer=tokenizer,)


@pytest.mark.run_only_on('GPU')
class TestMegatronCLIPModel:
    @pytest.mark.unit
    def test_constructor(self, clip_trainer_and_model):
        clip_model = clip_trainer_and_model[1]
        assert isinstance(clip_model, MegatronCLIPModel)

        num_weights = clip_model.num_weights
        assert num_weights == 46643969

    @pytest.mark.unit
    def test_build_dataset(self, clip_trainer_and_model, test_data_dir):
        clip_model = clip_trainer_and_model[1]
        train_ds, validation_ds = build_train_valid_datasets(
            model_cfg=clip_model.cfg, consumed_samples=0, tokenizer=clip_model.tokenizer,
        )
        assert len(train_ds) == 2000
        assert len(validation_ds) == 1000
        sample = next(iter(train_ds))
        assert "captions" in sample
        assert "images" in sample

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
    def test_forward(self, clip_trainer_and_model, test_data_dir, precision=None):
        trainer, clip_model = clip_trainer_and_model

        dtype = None
        if clip_model.cfg['precision'] == 32:
            dtype = torch.float
        elif clip_model.cfg['precision'] == 16:
            dtype = torch.float16
        elif clip_model.cfg['precision'] == 'bf16':
            dtype = torch.bfloat16
        else:
            raise ValueError(f"precision: {clip_model.cfg['precision']} is not supported.")

        clip_model.eval()
        _, validation_ds = build_datasets(clip_model.cfg, clip_model.tokenizer)

        val_loader = torch.utils.data.DataLoader(validation_ds, batch_size=4)
        batch = next(iter(val_loader))

        tokens = batch["images"]
        texts = batch["captions"]
        with torch.no_grad():
            B, C, H, W = tokens.shape
            assert H == W
            with torch.autocast('cuda', dtype=dtype):
                output_tensor = clip_model(image=tokens.cuda(), text=texts.cuda(),)
            # output is (B, #classes)
            # assert output_tensor.shape == torch.Size([B, clip_model.cfg['num_classes']])
            # assert output_tensor.dtype == dtype

    # @pytest.mark.unit
    # def test_vit_backbone(self, model_cfg, trainer_cfg, precision):
    #     initialize_model_parallel_for_nemo(
    #         world_size=1,
    #         global_rank=0,
    #         local_rank=0,
    #         tensor_model_parallel_size=model_cfg.get('tensor_model_parallel_size', 1),
    #         pipeline_model_parallel_size=model_cfg.get('pipeline_model_parallel_size', 1),
    #         virtual_pipeline_model_parallel_size=model_cfg.get('virtual_pipeline_model_parallel_size', None),
    #         pipeline_model_parallel_split_rank=model_cfg.get('pipeline_model_parallel_split_rank', 0),
    #         micro_batch_size=model_cfg.get('micro_batch_size'),
    #         global_batch_size=model_cfg.get('global_batch_size'),
    #         seed=model_cfg.get('seed', 1234),
    #         apex_transformer_log_level=model_cfg.get('apex_transformer_log_level', 30),
    #     )
    #
    #     dtype = None
    #     if trainer_cfg['precision'] == 32:
    #         dtype = torch.float
    #     elif trainer_cfg['precision'] == 16:
    #         dtype = torch.float16
    #     elif trainer_cfg['precision'] == 'bf16':
    #         dtype = torch.bfloat16
    #     else:
    #         raise ValueError(f"precision: {trainer_cfg['precision']} is not supported.")
    #
    #     vit_backbone = VitBackbone(
    #         model_cfg,
    #         init_method=None,
    #         scaled_init_method=None,
    #         pre_process=True,
    #         post_process=True,
    #         single_token_output=True
    #     ).cuda()
    #     vit_backbone.eval()
    #
    #     # shape: (B, C, H, W)
    #     tokens = torch.rand((6, 3, 224, 224))
    #
    #     with torch.no_grad():
    #         B, C, H, W = tokens.shape
    #         assert H == W
    #         with torch.autocast('cuda', dtype=dtype):
    #             output_tensor = vit_backbone(
    #                 tokens.cuda(),
    #             )
    #         # output is (B, #classes)
    #         assert output_tensor.shape == torch.Size([B, model_cfg['hidden_size']])
    #         assert output_tensor.dtype == dtype
    #
    # @pytest.mark.unit
    # def test_vit_head(self, model_cfg, trainer_cfg, precision):
    #     dtype = None
    #     if trainer_cfg['precision'] == 32:
    #         dtype = torch.float
    #     elif trainer_cfg['precision'] == 16:
    #         dtype = torch.float16
    #     elif trainer_cfg['precision'] == 'bf16':
    #         dtype = torch.bfloat16
    #     else:
    #         raise ValueError(f"precision: {trainer_cfg['precision']} is not supported.")
    #
    #     vit_head = VitMlpHead(
    #         24, 50,
    #     ).cuda()
    #     vit_head.eval()
    #
    #     hidden = torch.rand((6, 24))
    #
    #     with torch.no_grad():
    #         with torch.autocast('cuda', dtype=dtype):
    #             output_tensor = vit_head(
    #                 hidden.cuda(),
    #             )
    #         # output is (B, #classes)
    #         assert output_tensor.shape == torch.Size([6, 50])
    #         assert output_tensor.dtype == dtype
