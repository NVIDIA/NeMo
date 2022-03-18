from omegaconf import OmegaConf


class TestTrainingT5Config:

    def test_training_t5_config_220m(self):
        conf = OmegaConf.load('conf/training/t5/220m.yaml')
        s = """
        run:
          name: t5_220m
          results_dir: ${base_results_dir}/${.name}
          time_limit: "7-00:00:00"
          dependency: "singleton"
        
        name: megatron_t5
        restore_from_path: null # used when starting from a .nemo file
        
        trainer:
          num_nodes: 4
          gpus: 8
          accelerator: ddp
          precision: bf16
          amp_backend: native
          logger: False # logger provided by exp_manager
          checkpoint_callback: False
          replace_sampler_ddp: False
          max_epochs: null
          max_steps: 1000000 # consumed_samples = global_step * micro_batch_size * data_parallel_size * accumulate_grad_batches
          max_time: "06:23:30:00"
          log_every_n_steps: 100
          val_check_interval: ${multiply:2000, ${.accumulate_grad_batches}}
          limit_val_batches: ${multiply:50, ${.accumulate_grad_batches}}
          limit_test_batches: ${multiply:500, ${.accumulate_grad_batches}}
          accumulate_grad_batches: 1
          gradient_clip_val: 1.0
        
        
        exp_manager:
          explicit_log_dir: ${training.run.results_dir}
          exp_dir: null
          name: megatron_t5
          create_wandb_logger: True
          wandb_logger_kwargs:
            project: nemo_t5
            name: ${training.run.name}
          resume_if_exists: True
          resume_ignore_no_checkpoint: True
          create_checkpoint_callback: True
          checkpoint_callback_params:
            monitor: val_loss
            save_top_k: 10
            mode: min
            always_save_nemo: False # saves nemo file during validation, not implemented for model parallel
            save_nemo_on_train_end: False # not recommended when training large models on clusters with short time limits
            filename: 'megatron_t5--{val_loss:.2f}-{step}-{consumed_samples}'
            model_parallel_size: ${training.model.tensor_model_parallel_size}
          log_step_timing: True
          step_timing_kwargs:
            sync_cuda: True
            buffer_size: ${multiply:100, ${training.trainer.accumulate_grad_batches}}
        
        model:
          # model parallelism
          micro_batch_size: 64
          tensor_model_parallel_size: 1
          pipeline_model_parallel_size: 1 # T5 PP is not supported yet. Use 1 for now.
        
          # model architecture
          make_vocab_size_divisible_by: 128 # Pad the vocab size to be divisible by this value for computation efficiency.
          pre_process: True # add embedding
          post_process: True # add pooler
        
          megatron_amp_O2: False # use AMP with O2 style mixed precision instead of native amp on-the-fly weight autocasting.
        
          seq_length: 512
          max_position_embeddings: ${.seq_length}
          num_layers: 12
          hidden_size: 768
          ffn_hidden_size: 3072  # Transformer FFN hidden size. 4 * hidden_size.
          num_attention_heads: 12
          init_method_std: 0.015  # Standard deviation of the zero mean normal distribution used for weight initialization.')
          hidden_dropout: 0.1  # Dropout probability for hidden state transformer.
          attention_dropout: 0.1 # Dropout probability in the attention layer.
          kv_channels: null  # Projection weights dimension in multi-head attention. Set to hidden_size // num_attention_heads if null
          apply_query_key_layer_scaling: True # scale Q * K^T by 1 / layer-number.
          layernorm_epsilon: 1e-5
          persist_layer_norm: True # Use of persistent fused layer norm kernel.
          gradient_as_bucket_view: True # Allocate gradients in a contiguous bucket to save memory (less fragmentation and buffer memory)
          encoder_arch: 'transformer'
          decoder_arch: 'transformer'
          activation: 'gelu'
        
          tokenizer:
            library: 'megatron'
            type: 'BertWordPieceCase'
            model: null
            vocab_file: ${data_dir}/bpe/vocab.txt
            merge_file: null
            num_sentinel_tokens: 100
        
          # precision
          native_amp_init_scale: 4294967296 # 2 ** 32
          native_amp_growth_interval: 1000
          fp32_residual_connection: False # Move residual connections to fp32
          fp16_lm_cross_entropy: False # Move the cross entropy unreduced loss calculation for lm head to fp16
        
          # miscellaneous
          seed: 1234
          use_cpu_initialization: False # Init weights on the CPU (slow for large models)
          onnx_safe: False # Use work-arounds for known problems with Torch ONNX exporter.
        
          activations_checkpoint_method: null # 'uniform', 'block'
          activations_checkpoint_num_layers: 1
        
          optim:
            name: fused_adam
            lr: 0.0001
            betas:
              - 0.9
              - 0.999
            eps: 1e-8
            weight_decay: 0.01
            sched:
              name: WarmupAnnealing
              min_lr: 0.00001
              last_epoch: -1
              warmup_ratio: 0.01
        
        
          data:
            data_impl: mmap
            splits_string: "999982,9,9"
            seq_length: ${training.model.seq_length}
            seq_length_dec: 128
            skip_warmup: True
            num_workers: 4
            dataloader_type: single # cyclic
            masked_lm_prob: 0.15
            dataset_type: 't5'
            short_seq_prob: 0.0
            max_ngram_size: 10
            mean_ngram_size: null
            geometric_dist: True
            permutation: False
            whole_word_masking: True
            favor_longer_ngrams: False
            data_prefix: # Should be weight path weight path... for a blended dataset
              - .0333
              - ${data_dir}/my-t5_00_text_document
              - .0333
              - ${data_dir}/my-t5_01_text_document
              - .0333
              - ${data_dir}/my-t5_02_text_document
              - .0333
              - ${data_dir}/my-t5_03_text_document
              - .0333
              - ${data_dir}/my-t5_04_text_document
              - .0333
              - ${data_dir}/my-t5_05_text_document
              - .0333
              - ${data_dir}/my-t5_06_text_document
              - .0333
              - ${data_dir}/my-t5_07_text_document
              - .0333
              - ${data_dir}/my-t5_08_text_document
              - .0333
              - ${data_dir}/my-t5_09_text_document
              - .0333
              - ${data_dir}/my-t5_10_text_document
              - .0333
              - ${data_dir}/my-t5_11_text_document
              - .0333
              - ${data_dir}/my-t5_12_text_document
              - .0333
              - ${data_dir}/my-t5_13_text_document
              - .0333
              - ${data_dir}/my-t5_14_text_document
              - .0333
              - ${data_dir}/my-t5_15_text_document
              - .0333
              - ${data_dir}/my-t5_16_text_document
              - .0333
              - ${data_dir}/my-t5_17_text_document
              - .0333
              - ${data_dir}/my-t5_18_text_document
              - .0333
              - ${data_dir}/my-t5_19_text_document
              - .0333
              - ${data_dir}/my-t5_20_text_document
              - .0333
              - ${data_dir}/my-t5_21_text_document
              - .0333
              - ${data_dir}/my-t5_22_text_document
              - .0333
              - ${data_dir}/my-t5_23_text_document
              - .0333
              - ${data_dir}/my-t5_24_text_document
              - .0333
              - ${data_dir}/my-t5_25_text_document
              - .0333
              - ${data_dir}/my-t5_26_text_document
              - .0333
              - ${data_dir}/my-t5_27_text_document
              - .0333
              - ${data_dir}/my-t5_28_text_document
              - .0334
              - ${data_dir}/my-t5_29_text_document
        """
        expected = OmegaConf.create(s)
        assert expected == conf, f"conf/training/t5/220m.yaml must be set to {expected} but it currently is {conf}."

    def test_training_t5_config_3b(self):
        conf = OmegaConf.load('conf/training/t5/3b.yaml')
        s = """
        run:
          name: t5_3b
          results_dir: ${base_results_dir}/${.name}
          time_limit: "15-00:00:00"
          dependency: "singleton"
        
        name: megatron_t5
        restore_from_path: null # used when starting from a .nemo file
        
        trainer:
          num_nodes: 20
          gpus: 8
          accelerator: ddp
          precision: bf16
          amp_backend: native
          logger: False # logger provided by exp_manager
          checkpoint_callback: False
          replace_sampler_ddp: False
          max_epochs: null
          max_steps: 1000000 # consumed_samples = global_step * micro_batch_size * data_parallel_size * accumulate_grad_batches
          max_time: "14:23:30:00"
          log_every_n_steps: 100
          val_check_interval: ${multiply:2000, ${.accumulate_grad_batches}}
          limit_val_batches: ${multiply:50, ${.accumulate_grad_batches}}
          limit_test_batches: ${multiply:500, ${.accumulate_grad_batches}}
          accumulate_grad_batches: 1
          gradient_clip_val: 1.0
        
        
        exp_manager:
          explicit_log_dir: ${training.run.results_dir}
          exp_dir: null
          name: megatron_t5
          create_wandb_logger: True
          wandb_logger_kwargs:
            project: nemo_t5
            name: ${training.run.name}
          resume_if_exists: True
          resume_ignore_no_checkpoint: True
          create_checkpoint_callback: True
          checkpoint_callback_params:
            monitor: val_loss
            save_top_k: 10
            mode: min
            always_save_nemo: False # saves nemo file during validation, not implemented for model parallel
            save_nemo_on_train_end: False # not recommended when training large models on clusters with short time limits
            filename: 'megatron_t5--{val_loss:.2f}-{step}-{consumed_samples}'
            model_parallel_size: ${training.model.tensor_model_parallel_size}
          log_step_timing: True
          step_timing_kwargs:
            sync_cuda: True
            buffer_size: ${multiply:100, ${training.trainer.accumulate_grad_batches}}
        
        model:
          # model parallelism
          micro_batch_size: 27
          tensor_model_parallel_size: 2
          pipeline_model_parallel_size: 1 # T5 PP is not supported yet. Use 1 for now.
        
          # model architecture
          make_vocab_size_divisible_by: 128 # Pad the vocab size to be divisible by this value for computation efficiency.
          pre_process: True # add embedding
          post_process: True # add pooler
        
          megatron_amp_O2: False # use AMP with O2 style mixed precision instead of native amp on-the-fly weight autocasting.
        
          seq_length: 512
          max_position_embeddings: ${.seq_length}
          num_layers: 24
          hidden_size: 1024
          ffn_hidden_size: 16384  # Transformer FFN hidden size. 4 * hidden_size.
          num_attention_heads: 32
          init_method_std: 0.015  # Standard deviation of the zero mean normal distribution used for weight initialization.')
          hidden_dropout: 0.1  # Dropout probability for hidden state transformer.
          attention_dropout: 0.1 # Dropout probability in the attention layer.
          kv_channels: 128  # Projection weights dimension in multi-head attention. Set to hidden_size // num_attention_heads if null
          apply_query_key_layer_scaling: True # scale Q * K^T by 1 / layer-number.
          layernorm_epsilon: 1e-5
          persist_layer_norm: True # Use of persistent fused layer norm kernel.
          gradient_as_bucket_view: True # Allocate gradients in a contiguous bucket to save memory (less fragmentation and buffer memory)
          encoder_arch: 'transformer'
          decoder_arch: 'transformer'
          activation: 'gelu'
        
          tokenizer:
            library: 'megatron'
            type: 'BertWordPieceCase'
            model: null
            vocab_file: ${data_dir}/bpe/vocab.txt
            merge_file: null
            num_sentinel_tokens: 100
        
          # precision
          native_amp_init_scale: 4294967296 # 2 ** 32
          native_amp_growth_interval: 1000
          fp32_residual_connection: False # Move residual connections to fp32
          fp16_lm_cross_entropy: False # Move the cross entropy unreduced loss calculation for lm head to fp16
        
          # miscellaneous
          seed: 1234
          use_cpu_initialization: False # Init weights on the CPU (slow for large models)
          onnx_safe: False # Use work-arounds for known problems with Torch ONNX exporter.
        
          activations_checkpoint_method: null # 'uniform', 'block'
          activations_checkpoint_num_layers: 1
        
          optim:
            name: fused_adam
            lr: 0.0001
            betas:
              - 0.9
              - 0.999
            eps: 1e-8
            weight_decay: 0.01
            sched:
              name: WarmupAnnealing
              min_lr: 0.00001
              last_epoch: -1
              warmup_ratio: 0.01
        
        
          data:
            data_impl: mmap
            splits_string: "999982,9,9"
            seq_length: ${training.model.seq_length}
            seq_length_dec: 128
            skip_warmup: True
            num_workers: 4
            dataloader_type: single # cyclic
            masked_lm_prob: 0.15
            dataset_type: 't5'
            short_seq_prob: 0.0
            max_ngram_size: 10
            mean_ngram_size: null
            geometric_dist: True
            permutation: False
            whole_word_masking: True
            favor_longer_ngrams: False
            data_prefix: # Should be weight path weight path... for a blended dataset
              - .0333
              - ${data_dir}/my-t5_00_text_document
              - .0333
              - ${data_dir}/my-t5_01_text_document
              - .0333
              - ${data_dir}/my-t5_02_text_document
              - .0333
              - ${data_dir}/my-t5_03_text_document
              - .0333
              - ${data_dir}/my-t5_04_text_document
              - .0333
              - ${data_dir}/my-t5_05_text_document
              - .0333
              - ${data_dir}/my-t5_06_text_document
              - .0333
              - ${data_dir}/my-t5_07_text_document
              - .0333
              - ${data_dir}/my-t5_08_text_document
              - .0333
              - ${data_dir}/my-t5_09_text_document
              - .0333
              - ${data_dir}/my-t5_10_text_document
              - .0333
              - ${data_dir}/my-t5_11_text_document
              - .0333
              - ${data_dir}/my-t5_12_text_document
              - .0333
              - ${data_dir}/my-t5_13_text_document
              - .0333
              - ${data_dir}/my-t5_14_text_document
              - .0333
              - ${data_dir}/my-t5_15_text_document
              - .0333
              - ${data_dir}/my-t5_16_text_document
              - .0333
              - ${data_dir}/my-t5_17_text_document
              - .0333
              - ${data_dir}/my-t5_18_text_document
              - .0333
              - ${data_dir}/my-t5_19_text_document
              - .0333
              - ${data_dir}/my-t5_20_text_document
              - .0333
              - ${data_dir}/my-t5_21_text_document
              - .0333
              - ${data_dir}/my-t5_22_text_document
              - .0333
              - ${data_dir}/my-t5_23_text_document
              - .0333
              - ${data_dir}/my-t5_24_text_document
              - .0333
              - ${data_dir}/my-t5_25_text_document
              - .0333
              - ${data_dir}/my-t5_26_text_document
              - .0333
              - ${data_dir}/my-t5_27_text_document
              - .0333
              - ${data_dir}/my-t5_28_text_document
              - .0334
              - ${data_dir}/my-t5_29_text_document
        """
        expected = OmegaConf.create(s)
        assert expected == conf, f"conf/training/t5/3b.yaml must be set to {expected} but it currently is {conf}."
