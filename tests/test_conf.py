from omegaconf import OmegaConf


class TestConfig:

    def test_config(self):
        conf = OmegaConf.load('conf/config.yaml')
        s = """
        defaults:
          - _self_
          - cluster: bcm
          - data_preparation: download_pile
          - training: 5b
          - conversion: convert
          - evaluation: evaluate_all
          - override hydra/job_logging: stdout

        hydra:
          run:
            dir: .
          output_subdir: null

        run_data_preparation: True
        run_training: True
        run_conversion: True
        run_evaluation: True

        cluster_type: bcm
        training_config: 5b
        bignlp_path: ???
        data_dir: ${bignlp_path}/data
        base_results_dir: ${bignlp_path}/results
        container_mounts:
          - null
        container: nvcr.io/ea-bignlp/bignlp-training:22.01-py3

        wandb_api_key_file: null
        nccl_topology_xml_file: null
        """
        expected = OmegaConf.create(s)
        assert expected == conf, f"conf/config.yaml must be set to {expected} but it currently is {conf}."


class TestTrainingConfig:

    def test_training_config_126m(self):
        conf = OmegaConf.load('conf/training/126m.yaml')
        s = """
        run:
          name: 126m
          results_dir: ${base_results_dir}/${.name}
          time_limit: "1-12:00:00"
          dependency: "singleton"

        trainer:
          num_nodes: 8
          gpus: 8
          accelerator: ddp
          precision: 16
          amp_backend: native
          logger: False
          checkpoint_callback: False
          replace_sampler_ddp: False
          max_epochs: null
          max_steps: 600000
          max_time: "01:11:30:00"
          log_every_n_steps: 100
          val_check_interval: ${multiply:2000, ${.accumulate_grad_batches}}
          limit_val_batches: ${multiply:50, ${.accumulate_grad_batches}}
          limit_test_batches: ${multiply:50, ${.accumulate_grad_batches}}
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
            buffer_size: ${multiply:100, ${training.trainer.accumulate_grad_batches}}

        model:
          # model parallelism 
          micro_batch_size: 4
          global_batch_size: 256
          tensor_model_parallel_size: 1
          pipeline_model_parallel_size: 1

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
            data_prefix:
              - .0333
              - ${data_dir}/my-gpt3_00_text_document
              - .0333
              - ${data_dir}/my-gpt3_01_text_document
              - .0333
              - ${data_dir}/my-gpt3_02_text_document
              - .0333
              - ${data_dir}/my-gpt3_03_text_document
              - .0333
              - ${data_dir}/my-gpt3_04_text_document
              - .0333
              - ${data_dir}/my-gpt3_05_text_document
              - .0333
              - ${data_dir}/my-gpt3_06_text_document
              - .0333
              - ${data_dir}/my-gpt3_07_text_document
              - .0333
              - ${data_dir}/my-gpt3_08_text_document
              - .0333
              - ${data_dir}/my-gpt3_09_text_document
              - .0333
              - ${data_dir}/my-gpt3_10_text_document
              - .0333
              - ${data_dir}/my-gpt3_11_text_document
              - .0333
              - ${data_dir}/my-gpt3_12_text_document
              - .0333
              - ${data_dir}/my-gpt3_13_text_document
              - .0333
              - ${data_dir}/my-gpt3_14_text_document
              - .0333
              - ${data_dir}/my-gpt3_15_text_document
              - .0333
              - ${data_dir}/my-gpt3_16_text_document
              - .0333
              - ${data_dir}/my-gpt3_17_text_document
              - .0333
              - ${data_dir}/my-gpt3_18_text_document
              - .0333
              - ${data_dir}/my-gpt3_19_text_document
              - .0333
              - ${data_dir}/my-gpt3_20_text_document
              - .0333
              - ${data_dir}/my-gpt3_21_text_document
              - .0333
              - ${data_dir}/my-gpt3_22_text_document
              - .0333
              - ${data_dir}/my-gpt3_23_text_document
              - .0333
              - ${data_dir}/my-gpt3_24_text_document
              - .0333
              - ${data_dir}/my-gpt3_25_text_document
              - .0333
              - ${data_dir}/my-gpt3_26_text_document
              - .0333
              - ${data_dir}/my-gpt3_27_text_document
              - .0333
              - ${data_dir}/my-gpt3_28_text_document
              - .0334
              - ${data_dir}/my-gpt3_29_text_document
        """
        expected = OmegaConf.create(s)
        assert expected == conf, f"conf/training/126m.yaml must be set to {expected} but it currently is {conf}."

    def test_training_config_5b(self):
        conf = OmegaConf.load('conf/training/5b.yaml')
        s = """
        run:
          name: 5b
          results_dir: ${base_results_dir}/${.name}
          time_limit: "7-00:00:00"
          dependency: "singleton"

        trainer:
          num_nodes: 20
          gpus: 8
          accelerator: ddp
          precision: 16
          amp_backend: native
          logger: False
          checkpoint_callback: False
          replace_sampler_ddp: False
          max_epochs: null
          max_steps: 105000
          max_time: "06:23:30:00"
          log_every_n_steps: 100
          val_check_interval: ${multiply:2000, ${.accumulate_grad_batches}}
          limit_val_batches: ${multiply:50, ${.accumulate_grad_batches}}
          limit_test_batches: ${multiply:50, ${.accumulate_grad_batches}}
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
            buffer_size: ${multiply:100, ${training.trainer.accumulate_grad_batches}}

        model:
          # model parallelism: MBS=2, TPS=2, AGB=9 for 80GB nodes.
          micro_batch_size: 2
          global_batch_size: 1440
          tensor_model_parallel_size: 2
          pipeline_model_parallel_size: 1

          # model architecture
          encoder_seq_length: 2048
          max_position_embeddings: 2048
          num_layers: 24
          hidden_size: 4096
          ffn_hidden_size: ${multiply:4, ${.hidden_size}}
          num_attention_heads: 32
          init_method_std: 0.01
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

          optim:
            name: fused_adam
            lr: 1.2e-4
            weight_decay: 0.1
            betas:
            - 0.9
            - 0.95
            sched:
              name: CosineAnnealing
              warmup_steps: 190
              constant_steps: 20000
              min_lr: 1.2e-5

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
            data_prefix:
              - .0333
              - ${data_dir}/my-gpt3_00_text_document
              - .0333
              - ${data_dir}/my-gpt3_01_text_document
              - .0333
              - ${data_dir}/my-gpt3_02_text_document
              - .0333
              - ${data_dir}/my-gpt3_03_text_document
              - .0333
              - ${data_dir}/my-gpt3_04_text_document
              - .0333
              - ${data_dir}/my-gpt3_05_text_document
              - .0333
              - ${data_dir}/my-gpt3_06_text_document
              - .0333
              - ${data_dir}/my-gpt3_07_text_document
              - .0333
              - ${data_dir}/my-gpt3_08_text_document
              - .0333
              - ${data_dir}/my-gpt3_09_text_document
              - .0333
              - ${data_dir}/my-gpt3_10_text_document
              - .0333
              - ${data_dir}/my-gpt3_11_text_document
              - .0333
              - ${data_dir}/my-gpt3_12_text_document
              - .0333
              - ${data_dir}/my-gpt3_13_text_document
              - .0333
              - ${data_dir}/my-gpt3_14_text_document
              - .0333
              - ${data_dir}/my-gpt3_15_text_document
              - .0333
              - ${data_dir}/my-gpt3_16_text_document
              - .0333
              - ${data_dir}/my-gpt3_17_text_document
              - .0333
              - ${data_dir}/my-gpt3_18_text_document
              - .0333
              - ${data_dir}/my-gpt3_19_text_document
              - .0333
              - ${data_dir}/my-gpt3_20_text_document
              - .0333
              - ${data_dir}/my-gpt3_21_text_document
              - .0333
              - ${data_dir}/my-gpt3_22_text_document
              - .0333
              - ${data_dir}/my-gpt3_23_text_document
              - .0333
              - ${data_dir}/my-gpt3_24_text_document
              - .0333
              - ${data_dir}/my-gpt3_25_text_document
              - .0333
              - ${data_dir}/my-gpt3_26_text_document
              - .0333
              - ${data_dir}/my-gpt3_27_text_document
              - .0333
              - ${data_dir}/my-gpt3_28_text_document
              - .0334
              - ${data_dir}/my-gpt3_29_text_document
        """
        expected = OmegaConf.create(s)
        assert expected == conf, f"conf/training/5b.yaml must be set to {expected} but it currently is {conf}."

    def test_training_config_20b(self):
        conf = OmegaConf.load('conf/training/20b.yaml')
        s = """
        run:
          name: 20b
          results_dir: ${base_results_dir}/${.name}
          time_limit: "8-00:00:00"
          dependency: "singleton"

        trainer:
          gpus: 8
          num_nodes: 80
          accelerator: ddp
          precision: bf16
          amp_backend: native
          logger: False
          checkpoint_callback: False
          replace_sampler_ddp: False
          max_epochs: null
          max_steps: 105000
          max_time: "07:23:30:00"
          log_every_n_steps: 100
          val_check_interval: ${multiply:2000, ${.accumulate_grad_batches}}
          limit_val_batches: ${multiply:50, ${.accumulate_grad_batches}}
          limit_test_batches: ${multiply:50, ${.accumulate_grad_batches}}
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
            buffer_size: ${multiply:100, ${training.trainer.accumulate_grad_batches}}


        model:
          # model parallelism
          micro_batch_size: 2
          global_batch_size: 1440
          tensor_model_parallel_size: 8
          pipeline_model_parallel_size: 1

          # model architecture
          encoder_seq_length: 2048
          max_position_embeddings: 2048
          num_layers: 44
          hidden_size: 6144
          ffn_hidden_size: ${multiply:4, ${.hidden_size}}
          num_attention_heads: 48
          init_method_std: 0.008165
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
          activations_checkpoint_num_layers: 2

          tokenizer:
            library: 'megatron'
            type: 'GPT2BPETokenizer'
            model: null
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

          optim:
            name: fused_adam
            lr: 1e-4
            weight_decay: 0.1
            betas:
            - 0.9
            - 0.95
            sched:
              name: CosineAnnealing
              warmup_steps: 190
              constant_steps: 20000
              min_lr: 1e-5

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
            data_prefix:
              - .0333
              - ${data_dir}/my-gpt3_00_text_document
              - .0333
              - ${data_dir}/my-gpt3_01_text_document
              - .0333
              - ${data_dir}/my-gpt3_02_text_document
              - .0333
              - ${data_dir}/my-gpt3_03_text_document
              - .0333
              - ${data_dir}/my-gpt3_04_text_document
              - .0333
              - ${data_dir}/my-gpt3_05_text_document
              - .0333
              - ${data_dir}/my-gpt3_06_text_document
              - .0333
              - ${data_dir}/my-gpt3_07_text_document
              - .0333
              - ${data_dir}/my-gpt3_08_text_document
              - .0333
              - ${data_dir}/my-gpt3_09_text_document
              - .0333
              - ${data_dir}/my-gpt3_10_text_document
              - .0333
              - ${data_dir}/my-gpt3_11_text_document
              - .0333
              - ${data_dir}/my-gpt3_12_text_document
              - .0333
              - ${data_dir}/my-gpt3_13_text_document
              - .0333
              - ${data_dir}/my-gpt3_14_text_document
              - .0333
              - ${data_dir}/my-gpt3_15_text_document
              - .0333
              - ${data_dir}/my-gpt3_16_text_document
              - .0333
              - ${data_dir}/my-gpt3_17_text_document
              - .0333
              - ${data_dir}/my-gpt3_18_text_document
              - .0333
              - ${data_dir}/my-gpt3_19_text_document
              - .0333
              - ${data_dir}/my-gpt3_20_text_document
              - .0333
              - ${data_dir}/my-gpt3_21_text_document
              - .0333
              - ${data_dir}/my-gpt3_22_text_document
              - .0333
              - ${data_dir}/my-gpt3_23_text_document
              - .0333
              - ${data_dir}/my-gpt3_24_text_document
              - .0333
              - ${data_dir}/my-gpt3_25_text_document
              - .0333
              - ${data_dir}/my-gpt3_26_text_document
              - .0333
              - ${data_dir}/my-gpt3_27_text_document
              - .0333
              - ${data_dir}/my-gpt3_28_text_document
              - .0334
              - ${data_dir}/my-gpt3_29_text_document
        """
        expected = OmegaConf.create(s)
        assert expected == conf, f"conf/training/20b.yaml must be set to {expected} but it currently is {conf}."


class TestDataPrepConfig:

    def test_download_pile_config(self):
        conf = OmegaConf.load('conf/data_preparation/download_pile.yaml')
        s = """
        download_the_pile: True
        the_pile_url: "https://mystic.the-eye.eu/public/AI/pile/train/"
        file_numbers: "0-29" 
        preprocess_data: True
        download_vocab_url: "https://huggingface.co/gpt2/resolve/main/vocab.json"
        download_merges_url: "https://huggingface.co/gpt2/resolve/main/merges.txt"
        vocab_save_dir: ${data_dir}/bpe
        merges_save_dir: ${data_dir}/bpe
        log_dir: ${base_results_dir}/data_preparation/logs
        rm_downloaded: True
        rm_extracted: True
        nodes: 30
        time_limit: "4:00:00"
        bcp_preproc_npernode: 2
        """
        expected = OmegaConf.create(s)
        assert expected == conf, f"conf/data_preparation/download_pile.yaml must be set to {expected} but it currently is {conf}."

