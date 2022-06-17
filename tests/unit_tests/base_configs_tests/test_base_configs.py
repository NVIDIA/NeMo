from omegaconf import OmegaConf


class TestBaseConfigs:
    def test_gpt3_base_config(self):
        conf = OmegaConf.load("base_configs/gpt3.yaml")
        s = """
        run:
          name: gpt3_126m
          results_dir: ${base_results_dir}/${.name}
          time_limit: "1-12:00:00"
          dependency: "singleton"

        trainer:
          num_nodes: 8
          devices: 8
          accelerator: gpu
          precision: bf16
          amp_backend: native
          logger: False
          enable_checkpointing: False
          replace_sampler_ddp: False
          max_epochs: null
          max_steps: 600000
          max_time: "01:11:30:00"
          log_every_n_steps: 1
          val_check_interval: 2000
          limit_val_batches: 50
          limit_test_batches: 50
          accumulate_grad_batches: 1
          gradient_clip_val: 1.0


        exp_manager:
          explicit_log_dir: ${training.run.results_dir}
          exp_dir: null
          name: megatron_gpt
          create_wandb_logger: False
          wandb_logger_kwargs:
            project: nemo_gpt3
            name: ${training.run.name}
          resume_if_exists: True
          resume_ignore_no_checkpoint: True
          create_checkpoint_callback: True
          checkpoint_callback_params:
            monitor: val_loss
            save_top_k: 10
            mode: min
            always_save_nemo: False
            save_nemo_on_train_end: False
            filename: 'megatron_gpt--{val_loss:.2f}-{step}-{consumed_samples}'
            model_parallel_size: ${multiply:${training.model.tensor_model_parallel_size}, ${training.model.pipeline_model_parallel_size}}
          log_step_timing: True
          step_timing_kwargs:
            sync_cuda: True
            buffer_size: 5

        model:
          # model parallelism
          micro_batch_size: 4
          global_batch_size: 256
          tensor_model_parallel_size: 1
          pipeline_model_parallel_size: 1
          resume_from_checkpoint: null

          # model architecture
          encoder_seq_length: 2048
          max_position_embeddings: 2048
          num_layers: 12
          hidden_size: 768
          ffn_hidden_size: ${multiply:4, ${.hidden_size}}
          num_attention_heads: 12
          init_method_std: 0.023
          hidden_dropout: 0.1
          kv_channels: null
          apply_query_key_layer_scaling: True
          layernorm_epsilon: 1e-5
          make_vocab_size_divisible_by: 128
          pre_process: True
          post_process: True
          persist_layer_norm: True
          gradient_as_bucket_view: True
          activations_checkpoint_method: block
          activations_checkpoint_num_layers: 0

          tokenizer:
            library: 'megatron'
            type: 'GPT2BPETokenizer'
            model: null
            delimiter: null
            vocab_file: ${data_dir}/bpe/vocab.json
            merge_file: ${data_dir}/bpe/merges.txt

          # precision
          native_amp_init_scale: 4294967296
          native_amp_growth_interval: 1000
          hysteresis: 2
          fp32_residual_connection: False
          fp16_lm_cross_entropy: False

          # Megatron O2-style half-precision
          megatron_amp_O2: True

          # miscellaneous
          seed: 1234
          use_cpu_initialization: False
          onnx_safe: False
          apex_transformer_log_level: 30

          optim:
            name: fused_adam
            lr: 6e-4
            weight_decay: 0.1
            betas:
            - 0.9
            - 0.95
            sched:
              name: CosineAnnealing
              warmup_steps: 636
              constant_steps: 100000
              min_lr: 6e-5

          data:
            data_impl: mmap
            splits_string: "99990,8,2"
            seq_length: 2048
            skip_warmup: True
            num_workers: 2
            dataloader_type: single
            reset_position_ids: False
            reset_attention_mask: False
            eod_mask_loss: False
            index_mapping_dir: null
            data_prefix:
              - 1.0
              - ${data_dir}/my-gpt3_00_text_document
        """
        expected = OmegaConf.create(s)
        assert (
            expected == conf
        ), f"base_configs/gpt3.yaml must be set to {expected} but it currently is {conf}."

    def test_t5_base_config(self):
        conf = OmegaConf.load("base_configs/t5.yaml")
        s = """
        run:
          name: t5_220m
          results_dir: ${base_results_dir}/${.name}
          time_limit: "7-00:00:00"
          dependency: "singleton"

        name: megatron_t5
        restore_from_path: null

        trainer:
          num_nodes: 4
          devices: 8
          accelerator: gpu
          precision: bf16
          amp_backend: native
          logger: False
          enable_checkpointing: False
          replace_sampler_ddp: False
          max_epochs: null
          max_steps: 1000000
          max_time: "06:23:30:00"
          log_every_n_steps: 1
          val_check_interval: 2000
          limit_val_batches: 50
          limit_test_batches: 500
          accumulate_grad_batches: 1
          gradient_clip_val: 1.0


        exp_manager:
          explicit_log_dir: ${training.run.results_dir}
          exp_dir: null
          name: megatron_t5
          create_wandb_logger: False
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
            always_save_nemo: False
            save_nemo_on_train_end: False
            filename: 'megatron_t5--{val_loss:.2f}-{step}-{consumed_samples}'
            model_parallel_size: ${multiply:${training.model.tensor_model_parallel_size}, ${training.model.pipeline_model_parallel_size}}
          log_step_timing: True
          step_timing_kwargs:
            sync_cuda: True
            buffer_size: 5

        model:
          # model parallelism
          micro_batch_size: 64
          global_batch_size: 2048
          tensor_model_parallel_size: 1
          pipeline_model_parallel_size: 1
          resume_from_checkpoint: null
          pipeline_model_parallel_split_rank: ${divide_floor:${.pipeline_model_parallel_size}, 2}

          # model architecture
          make_vocab_size_divisible_by: 128
          pre_process: True
          post_process: True

          megatron_amp_O2: True

          seq_length: 512
          max_position_embeddings: ${.seq_length}
          num_layers: 12
          hidden_size: 768
          ffn_hidden_size: 2048 
          num_attention_heads: 12
          init_method_std: 0.015
          hidden_dropout: 0.1
          attention_dropout: 0.1
          kv_channels: null
          apply_query_key_layer_scaling: True
          layernorm_epsilon: 1e-5
          persist_layer_norm: True
          gradient_as_bucket_view: True
          bias_gelu_fusion: False
          masked_softmax_fusion: True
          encoder_arch: 'transformer'
          decoder_arch: 'transformer'
          activation: 'geglu'

          tokenizer:
            library: 'megatron'
            type: 'BertWordPieceCase'
            model: null
            vocab_file: ${data_dir}/vocab.txt
            merge_file: null
            num_sentinel_tokens: 100

          # precision
          native_amp_init_scale: 4294967296
          native_amp_growth_interval: 1000
          fp32_residual_connection: False
          fp16_lm_cross_entropy: False

          # miscellaneous
          seed: 1234
          use_cpu_initialization: False
          onnx_safe: False
          apex_transformer_log_level: 30

          activations_checkpoint_method: block
          activations_checkpoint_num_layers: 0

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
            seq_length: 512
            seq_length_dec: 128
            skip_warmup: True
            num_workers: 4
            dataloader_type: single
            masked_lm_prob: 0.15
            dataset_type: 't5'
            short_seq_prob: 0.0
            max_ngram_size: 10
            mean_ngram_size: null
            geometric_dist: True
            permutation: False
            whole_word_masking: True
            favor_longer_ngrams: False
            index_mapping_dir: null
            data_prefix:
              - 1.0
              - ${data_dir}/my-t5_00_bert_tokenizer_text_document
        """
        expected = OmegaConf.create(s)
        assert (
            expected == conf
        ), f"base_configs/t5.yaml must be set to {expected} but it currently is {conf}."

    def test_mt5_base_config(self):
        conf = OmegaConf.load("base_configs/mt5.yaml")
        s = """
        run:
          name: mt5_390m
          results_dir: ${base_results_dir}/${.name}
          time_limit: "7-00:00:00"
          dependency: "singleton"
          preprocessed_dir: ${data_dir}/mc4/preprocessed
          blending_alpha: 0.7

        name: megatron_mt5
        restore_from_path: null

        trainer:
          num_nodes: 4
          devices: 8
          accelerator: gpu
          precision: bf16
          amp_backend: native
          logger: False
          enable_checkpointing: False
          replace_sampler_ddp: False
          max_epochs: null
          max_steps: 1066667
          max_time: "06:23:30:00"
          log_every_n_steps: 1
          val_check_interval: 2000
          limit_val_batches: 50
          limit_test_batches: 500
          accumulate_grad_batches: 1
          gradient_clip_val: 1.0


        exp_manager:
          explicit_log_dir: ${training.run.results_dir}
          exp_dir: null
          name: megatron_mt5
          create_wandb_logger: False
          wandb_logger_kwargs:
            project: nemo_mt5
            name: ${training.run.name}
          resume_if_exists: True
          resume_ignore_no_checkpoint: True
          create_checkpoint_callback: True
          checkpoint_callback_params:
            monitor: val_loss
            save_top_k: 10
            mode: min
            always_save_nemo: False
            save_nemo_on_train_end: False
            filename: 'megatron_mt5--{val_loss:.2f}-{step}-{consumed_samples}'
            model_parallel_size: ${multiply:${training.model.tensor_model_parallel_size}, ${training.model.pipeline_model_parallel_size}}
          log_step_timing: True
          step_timing_kwargs:
            sync_cuda: True
            buffer_size: 5

        model:
          # model parallelism
          micro_batch_size: 32
          global_batch_size: 2048
          tensor_model_parallel_size: 1
          pipeline_model_parallel_size: 1
          resume_from_checkpoint: null
          pipeline_model_parallel_split_rank: ${divide_floor:${.pipeline_model_parallel_size}, 2}

          # model architecture
          make_vocab_size_divisible_by: 128
          pre_process: True
          post_process: True

          megatron_amp_O2: True

          seq_length: 512
          max_position_embeddings: ${.seq_length}
          num_layers: 12
          hidden_size: 768
          ffn_hidden_size: 2048
          num_attention_heads: 12
          init_method_std: 0.015
          hidden_dropout: 0.1
          attention_dropout: 0.1
          kv_channels: 64
          apply_query_key_layer_scaling: True
          layernorm_epsilon: 1e-5
          persist_layer_norm: True
          gradient_as_bucket_view: True
          bias_gelu_fusion: False
          masked_softmax_fusion: True
          encoder_arch: 'transformer'
          decoder_arch: 'transformer'
          activation: 'geglu'

          tokenizer:
            library: 'sentencepiece'
            type: null
            model: ${data_dir}/mt5_tokenizer.model
            vocab_file: null
            merge_file: null
            num_sentinel_tokens: 100

          # precision
          native_amp_init_scale: 4294967296
          native_amp_growth_interval: 1000
          fp32_residual_connection: False
          fp16_lm_cross_entropy: False

          # miscellaneous
          seed: 1234
          use_cpu_initialization: False
          onnx_safe: False
          apex_transformer_log_level: 30

          activations_checkpoint_method: block
          activations_checkpoint_num_layers: 0

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
            splits_string: "999892,99,9"
            seq_length: 512
            seq_length_dec: 128
            skip_warmup: True
            num_workers: 8
            dataloader_type: single
            masked_lm_prob: 0.15
            dataset_type: 't5'
            short_seq_prob: 0.0
            max_ngram_size: 10
            mean_ngram_size: null
            geometric_dist: True
            permutation: False
            whole_word_masking: False
            favor_longer_ngrams: False
            index_mapping_dir: null
            data_prefix:
              - 1.0
              - ${data_dir}/mc4/preprocessed/fr_000-008_text_document
        """
        expected = OmegaConf.create(s)
        assert (
            expected == conf
        ), f"base_configs/mt5.yaml must be set to {expected} but it currently is {conf}."
