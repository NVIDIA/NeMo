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
from nemo.collections.multimodal.models.multimodal_llm.neva.neva_model import MegatronNevaModel
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer

from nemo.collections.multimodal.data.neva.neva_dataset import (
    DataCollatorForSupervisedDataset,
    make_supervised_data_module,
)
from nemo.collections.nlp.parts.nlp_overrides import NLPDDPStrategy

DEVICE_CAPABILITY = None
if torch.cuda.is_available():
    DEVICE_CAPABILITY = torch.cuda.get_device_capability()


@pytest.fixture()
def model_cfg():

    model_cfg_string = """
      precision: 16
      
      # Batch size guideline for different types of dataset
      micro_batch_size: 2 # limited by GPU memory
      global_batch_size: 4 # will use more micro batches to reach global batch size
    
      tensor_model_parallel_size: 1 # intra-layer model parallelism
      pipeline_model_parallel_size: 1 # inter-layer model parallelism
      virtual_pipeline_model_parallel_size: null # interleaved pipeline
    
      restore_from_path: null # used in fine-tuning
    
      # Multimodal configs
      mm_cfg:
        llm:
          from_pretrained: null # path to nemo checkpoint
          freeze: True
          model_type: llama_2 # `nvgpt` or `llama_2` supported
        vision_encoder:
          from_pretrained: "openai/clip-vit-large-patch14" # path or name
          from_hf: True
          patch_dim: 14
          hidden_size: 1024 # could be found from model but tricky in code
          vision_select_layer: -2   # default to the last layer
          class_token_length: 1
          freeze: True
        pretrain_mm_mlp_adapter: null # path to pretrained mm adapter
        use_im_start_end: False
    
      # LLM configs
      # use GPTModel from megatron.core
      mcore_gpt: False
    
      # model architecture
      encoder_seq_length: 4096
      max_position_embeddings: ${.encoder_seq_length}
      position_embedding_type: rope
      num_layers: 2
      hidden_size: 5120
      ffn_hidden_size: 13824 # Transformer FFN hidden size. Usually 4 * hidden_size.
      num_attention_heads: 40
      init_method_std: 0.014 # Standard deviation of the zero mean normal distribution used for weight initialization.')
      use_scaled_init_method: True # use scaled residuals initialization
      hidden_dropout: 0.0 # Dropout probability for hidden state transformer.
      attention_dropout: 0.0 # Dropout probability for attention
      ffn_dropout: 0.0 # Dropout probability in the feed-forward layer.
      kv_channels: null # Projection weights dimension in multi-head attention. Set to hidden_size // num_attention_heads if null
      apply_query_key_layer_scaling: True # scale Q * K^T by 1 / layer-number.
      normalization: rmsnorm # Type of normalization layers
      layernorm_epsilon: 1e-5
      do_layer_norm_weight_decay: False # True means weight decay on all params
      pre_process: True # add embedding
      post_process: True # add pooler
      persist_layer_norm: True # Use of persistent fused layer norm kernel.
      bias: False # Whether to use bias terms in all weight matrices.
      activation: 'fast-swiglu' # Options ['gelu', 'geglu', 'swiglu', 'reglu', 'squared-relu', 'fast-geglu', 'fast-swiglu', 'fast-reglu']
      headscale: False # Whether to learn extra parameters that scale the output of the each self-attention head.
      transformer_block_type: 'pre_ln' # Options ['pre_ln', 'post_ln', 'normformer']
      normalize_attention_scores: True # Whether to scale the output Q * K^T by 1 / sqrt(hidden_size_per_head). This arg is provided as a configuration option mostly for compatibility with models that have been weight-converted from HF. You almost always want to se this to True.
      rotary_percentage: 1.0 # If using position_embedding_type=rope, then the per head dim is multiplied by this.
      attention_type: 'multihead' # Attention type. Options ['multihead']
      share_embeddings_and_output_weights: False # Share embedding and output layer weights.
      overlap_p2p_comm: False # Overlap p2p communication with computes. This argument is valid only when `virtual_pipeline_model_parallel_size` is larger than 1
      batch_p2p_comm: True # Batch consecutive inter-peer send/recv operations. This argument is valid only when `virtual_pipeline_model_parallel_size` is larger than 1
      seq_len_interpolation_factor: null # RoPE Interpolation factor for sequence length. This is used to build long-context models with RoPE ex: https://arxiv.org/abs/2306.15595.
      num_query_groups: null # Number of query groups for group query attention. If None, normal attention is used.
      use_flash_attention: False
    
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
      bias_dropout_add_fusion: False # Use a kernel that fuses the bias addition, dropout and residual connection addition.
    
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
      async_grad_allreduce: False
      grad_allreduce_chunk_size_mb: 125
      grad_div_ar_fusion: True # Fuse grad division into torch.distributed.all_reduce
    
      # miscellaneous
      seed: 1234
      resume_from_checkpoint: null # manually set the checkpoint file to load from
      apex_transformer_log_level: 30 # Python logging level displays logs with severity greater than or equal to this
      gradient_as_bucket_view: True # PyTorch DDP argument. Allocate gradients in a contiguous bucket to save memory (less fragmentation and buffer memory)
    
      tokenizer:
        library: 'sentencepiece'
        type: null
        model: /lustre/fsw/joc/multimodal/datasets/LLaVA-CC3M-Pretrain-595K/tiny_neva/tokenizer_add_special.model
        vocab_file: null
        merge_file: null
        delimiter: null # only used for tabular tokenizer
        sentencepiece_legacy: False # Legacy=True allows you to add special tokens to sentencepiece tokenizers.
        additional_special_tokens: null # ["<extra_id_0>", "<extra_id_1>", "<extra_id_2>", "<extra_id_3>", "<extra_id_4>", "<extra_id_5>"]
    
      data:
        num_workers: 8
        dataloader_type: cyclic
        data_path: /lustre/fsw/joc/multimodal/datasets/LLaVA-CC3M-Pretrain-595K/tiny_neva/dummy.json
        lazy_preprocess: True
        is_multimodal: True
        sep_image_conv_front: False
        image_token_len: 256
        conv_template: ${mm_cfg.llm.model_type} # check `nemo/collections/multimodal/data/neva/conversation.py`
        image_folder: /lustre/fsw/joc/multimodal/datasets/LLaVA-CC3M-Pretrain-595K/tiny_neva/images
        image_aspect_ratio: 'square'
    
      # Nsys profiling options
      nsys_profile:
        enabled: False
        start_step: 10  # Global batch to start profiling
        end_step: 10 # Global batch to end profiling
        ranks: [ 0 ] # Global rank IDs to profile
        gen_shape: False # Generate model and kernel details including input shapes
    
      optim:
        name: fused_adam
        lr: 2e-3
        weight_decay: 0.
        betas:
          - 0.9
          - 0.95
        sched:
          name: CosineAnnealing
          warmup_steps: 140
          constant_steps: 0
          min_lr: 2e-5
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
        name: megatron_neva
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
def neva_trainer_and_model(model_cfg, trainer_cfg, precision, test_data_dir):
    model_cfg['precision'] = precision
    trainer_cfg['precision'] = precision
    model_cfg['tokenizer']['model'] = (
        os.path.join(test_data_dir, "multimodal/tiny-neva/tokenizer_add_special.model"),
    )

    strategy = NLPDDPStrategy()

    trainer = Trainer(strategy=strategy, **trainer_cfg)

    cfg = DictConfig(model_cfg)

    model = MegatronNevaModel(cfg=cfg, trainer=trainer)

    def dummy():
        return

    if model.trainer.strategy.launcher is not None:
        model.trainer.strategy.launcher.launch(dummy, trainer=model.trainer)
    model.trainer.strategy.setup_environment()

    return trainer, model


def build_datasets(cfg, tokenizer, test_data_dir):
    cfg.data.data_path = (os.path.join(test_data_dir, "multimodal/tiny-neva/dummy.json"),)
    cfg.data.image_folder = os.path.join(test_data_dir, "multimodal/tiny-neva/images")
    ds_dict = make_supervised_data_module(tokenizer=tokenizer, model_cfg=cfg,)
    return ds_dict["train_dataset"], ds_dict["eval_dataset"]


@pytest.mark.run_only_on('GPU')
class TestMegatronNevaModel:
    @pytest.mark.unit
    def test_constructor(self, neva_trainer_and_model):
        neva_model = neva_trainer_and_model[1]
        assert isinstance(neva_model, MegatronNevaModel)

        num_weights = neva_model.num_weights
        assert num_weights == 5248000

    @pytest.mark.unit
    def test_build_dataset(self, neva_trainer_and_model, test_data_dir):
        neva_model = neva_trainer_and_model[1]
        neva_model.cfg.data.data_path = (os.path.join(test_data_dir, "multimodal/tiny-neva/dummy.json"),)
        neva_model.cfg.data.image_folder = os.path.join(test_data_dir, "multimodal/tiny-neva/images")
        ds_dict = make_supervised_data_module(tokenizer=neva_model.tokenizer, model_cfg=neva_model.cfg,)
        train_ds, validation_ds = ds_dict["train_dataset"], ds_dict["eval_dataset"]
        assert len(train_ds) == 60
        assert len(validation_ds) == 60
        sample = next(iter(train_ds))
        assert "tokens" in sample
        assert "labels" in sample
        assert "image" in sample

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
    def test_forward(self, neva_trainer_and_model, test_data_dir, precision=None):
        trainer, neva_model = neva_trainer_and_model

        dtype = None
        if neva_model.cfg['precision'] in [32, '32', '32-true']:
            dtype = torch.float
        elif neva_model.cfg['precision'] in [16, '16', '16-mixed']:
            dtype = torch.float16
        elif neva_model.cfg['precision'] in ['bf16', 'bf16-mixed']:
            dtype = torch.bfloat16
        else:
            raise ValueError(f"precision: {neva_model.cfg['precision']} is not supported.")

        neva_model = neva_model.cuda()
        neva_model.eval()
        _, validation_ds = build_datasets(neva_model.cfg, neva_model.tokenizer, test_data_dir)

        collate_func = DataCollatorForSupervisedDataset(neva_model.cfg, neva_model.tokenizer)
        val_loader = torch.utils.data.DataLoader(validation_ds, batch_size=4, collate_fn=collate_func,)
        batch = next(iter(val_loader))

        tokens = batch['tokens'].cuda()
        position_ids = batch['position_ids'].cuda()
        attention_mask = batch['attention_mask'].cuda()
        labels = batch['labels'].cuda()
        media = batch['media'].cuda()
        with torch.no_grad():
            with torch.autocast('cuda', dtype=dtype):
                output_tensor = neva_model(tokens, position_ids, attention_mask, labels, media)
            print(tokens, output_tensor)
            assert output_tensor.shape == tokens.shape
            assert output_tensor.dtype == torch.float32
