from omegaconf import OmegaConf


class TestFinetuningmT5Config:
    def test_finetuning_mt5_config(self):
        conf = OmegaConf.load("conf/fine_tuning/mt5/xquad.yaml")
        s = """
        run:
          name: ${.task_name}_${.model_train_name}
          time_limit: "04:00:00"
          dependency: "singleton"
          convert_name: convert_nemo
          model_train_name: mt5_390m
          convert_dir: ${base_results_dir}/${fine_tuning.run.model_train_name}/${fine_tuning.run.convert_name}
          task_name: "xquad"
          results_dir: ${base_results_dir}/${.model_train_name}/${.task_name}
        
        trainer:
          devices: 8
          num_nodes: 1
          accelerator: gpu
          precision: bf16
          logger: False # logger provided by exp_manager
          enable_checkpointing: False
          replace_sampler_ddp: False
          max_epochs: 5
          max_steps: null # consumed_samples = global_step * micro_batch_size * data_parallel_size * accumulate_grad_batches
          log_every_n_steps: 10
          val_check_interval: 300
          accumulate_grad_batches: 1
          gradient_clip_val: 1.0
        
        exp_manager:
          explicit_log_dir: ${fine_tuning.run.results_dir}/results
          exp_dir: null
          name: megatron_mt5_${fine_tuning.run.task_name}
          create_wandb_logger: False
          wandb_logger_kwargs:
            project: nemo_mt5_${fine_tuning.run.task_name}
            name: ${fine_tuning.run.name}
          resume_if_exists: True
          resume_ignore_no_checkpoint: True
          create_checkpoint_callback: True
          checkpoint_callback_params:
            monitor: validation_${fine_tuning.model.data.validation_ds.metric.name}
            save_top_k: 5
            mode: max
            always_save_nemo: False
            save_nemo_on_train_end: True # Set to true for subsequent validation runs.
            filename: 'megatron_mt5--{${.monitor}:.3f}-{step}'
            model_parallel_size: ${fine_tuning.model.model_parallel_size}
            save_best_model: True
        
        model: # For different fine_tuning tasks, tuning the hyper parameters accordingly
          restore_from_path: ${fine_tuning.run.convert_dir}/results/megatron_mt5.nemo # Path to a trained mt5 .nemo file
          tensor_model_parallel_size: 1
          pipeline_model_parallel_size: 1
          pipeline_model_parallel_split_rank: ${divide_floor:${.pipeline_model_parallel_size}, 2}
          model_parallel_size: ${multiply:${.tensor_model_parallel_size}, ${.pipeline_model_parallel_size}}
          gradient_as_bucket_view: True # Allocate gradients in a contiguous bucket to save memory (less fragmentation and buffer memory)
          megatron_amp_O2: False # Enable O2 optimization for megatron amp
          resume_from_checkpoint: null
          hidden_dropout: 0.1 # Override dropout prob from pretraining
          attention_dropout: 0.1 # Override attention dropout prob from pretraining
        
          data:
            train_ds:
              src_file_name: ${data_dir}/squad_data/v1.1/train-v1.1_src.txt
              tgt_file_name: ${data_dir}/squad_data/v1.1/train-v1.1_tgt.txt
              global_batch_size: 128
              micro_batch_size: 16
              shuffle: True
              num_workers: 0  # Known issue: > 0 may not work
              pin_memory: True
              max_src_seq_length: 512
              max_tgt_seq_length: 128
              drop_last: True
              concat_sampling_technique: temperature # When providing a list of datasets, this arg defines the sampling strategy. Options: ['temperature', 'random']
              concat_sampling_temperature: 5 # When providing a list of datasets, this arg defines the sampling temperature when strategy='temperature'
              concat_sampling_probabilities: null # When providing a list of datasets, this arg defines the sampling probabilities from each dataset when strategy='random'
        
            validation_ds:
              src_file_name:
                - ${data_dir}/squad_data/xquad/xquad.en_src.txt
                - ${data_dir}/squad_data/xquad/xquad.es_src.txt
                - ${data_dir}/squad_data/xquad/xquad.de_src.txt
                - ${data_dir}/squad_data/xquad/xquad.hi_src.txt
                - ${data_dir}/squad_data/xquad/xquad.zh_src.txt
              tgt_file_name:
                - ${data_dir}/squad_data/xquad/xquad.en_tgt.txt
                - ${data_dir}/squad_data/xquad/xquad.es_tgt.txt
                - ${data_dir}/squad_data/xquad/xquad.de_tgt.txt
                - ${data_dir}/squad_data/xquad/xquad.hi_tgt.txt
                - ${data_dir}/squad_data/xquad/xquad.zh_tgt.txt
              names:
                - xquad_en
                - xquad_es
                - xquad_de
                - xquad_hi
                - xquad_zh
              global_batch_size: 128
              micro_batch_size: 16
              shuffle: False
              num_workers: 4
              pin_memory: True
              max_src_seq_length: 512
              max_tgt_seq_length: 128
              drop_last: False
              write_predictions_to_file: False
              output_file_path_prefix: null # Prefix of the file to write predictions to.
              metric:
                name: "exact_string_match" # Name of the evaluation metric to use.
                average: null # Average the metric over the dataset. Options: ['macro', 'micro']. Works only for 'F1', 'accuracy' etc. Refer to torchmetrics for metrics where this is supported.
                num_classes: null
                class_labels: null
                labels_are_strings: False
        
          optim:
            name: fused_adam
            lr: 2.0e-5
            weight_decay: 0.1
        """
        expected = OmegaConf.create(s)
        assert (
                expected == conf
        ), f"conf/fine_tuning/mt5/xnli.yaml must be set to {expected} but it currently is {conf}."


class TestFinetuningT5Config:
    def test_fine_tuning_t5_config(self):
        conf = OmegaConf.load("conf/fine_tuning/t5/squad.yaml")
        s = """
        run:
          name: ${.task_name}_${.model_train_name}
          time_limit: "04:00:00"
          dependency: "singleton"
          convert_name: convert_nemo
          model_train_name: t5_220m
          convert_dir: ${base_results_dir}/${fine_tuning.run.model_train_name}/${fine_tuning.run.convert_name}
          task_name: "squad"  # SQuAD v1.1
          results_dir: ${base_results_dir}/${.model_train_name}/${.task_name}
        
        trainer:
          devices: 8
          num_nodes: 1
          accelerator: gpu
          precision: bf16
          logger: False # logger provided by exp_manager
          enable_checkpointing: False
          replace_sampler_ddp: False
          max_epochs: 5
          max_steps: null # consumed_samples = global_step * micro_batch_size * data_parallel_size * accumulate_grad_batches
          log_every_n_steps: 10
          val_check_interval: 300
          accumulate_grad_batches: 1
          gradient_clip_val: 1.0
        
        exp_manager:
          explicit_log_dir: ${fine_tuning.run.results_dir}/results
          exp_dir: null
          name: megatron_t5_${fine_tuning.run.task_name}
          create_wandb_logger: False
          wandb_logger_kwargs:
            project: nemo_t5_${fine_tuning.run.task_name}
            name: ${fine_tuning.run.name}
          resume_if_exists: True
          resume_ignore_no_checkpoint: True
          create_checkpoint_callback: True
          checkpoint_callback_params:
            monitor: validation_${fine_tuning.model.data.validation_ds.metric.name}
            save_top_k: 5
            mode: max
            always_save_nemo: False
            save_nemo_on_train_end: True # Set to true for subsequent validation runs.
            filename: 'megatron_t5--{${.monitor}:.3f}-{step}'
            model_parallel_size: ${fine_tuning.model.model_parallel_size}
            save_best_model: True
        
        model: # For different fine_tuning tasks, tuning the hyper parameters accordingly; below is only for MNLI
          restore_from_path: ${fine_tuning.run.convert_dir}/results/megatron_t5.nemo # Path to a trained T5 .nemo file
          tensor_model_parallel_size: 1
          pipeline_model_parallel_size: 1
          pipeline_model_parallel_split_rank: ${divide_floor:${.pipeline_model_parallel_size}, 2}
          model_parallel_size: ${multiply:${.tensor_model_parallel_size}, ${.pipeline_model_parallel_size}}
          gradient_as_bucket_view: True # Allocate gradients in a contiguous bucket to save memory (less fragmentation and buffer memory)
          megatron_amp_O2: False # Enable O2 optimization for megatron amp
          resume_from_checkpoint: null
          hidden_dropout: 0.1 # Override dropout prob from pretraining
          attention_dropout: 0.1 # Override attention dropout prob from pretraining
        
          data:
            train_ds:
              src_file_name: ${data_dir}/squad_data/v1.1/train-v1.1_src.txt
              tgt_file_name: ${data_dir}/squad_data/v1.1/train-v1.1_tgt.txt
              global_batch_size: 128
              micro_batch_size: 16
              shuffle: True
              num_workers: 0  # Known issue: > 0 may not work
              pin_memory: True
              max_src_seq_length: 512
              max_tgt_seq_length: 128
              drop_last: True
              concat_sampling_technique: temperature # When providing a list of datasets, this arg defines the sampling strategy. Options: ['temperature', 'random']
              concat_sampling_temperature: 5 # When providing a list of datasets, this arg defines the sampling temperature when strategy='temperature'
              concat_sampling_probabilities: null # When providing a list of datasets, this arg defines the sampling probabilities from each dataset when strategy='random'
        
            validation_ds:
              src_file_name: ${data_dir}/squad_data/v1.1/dev-v1.1_src.txt
              tgt_file_name: ${data_dir}/squad_data/v1.1/dev-v1.1_tgt.txt
              names: null # If src/tgt file names are ListConfigs, the corresponding label is used to log metrics.
              global_batch_size: 128
              micro_batch_size: 16
              shuffle: False
              num_workers: 4
              pin_memory: True
              max_src_seq_length: 512
              max_tgt_seq_length: 128
              drop_last: False
              write_predictions_to_file: False
              output_file_path_prefix: null # Prefix of the file to write predictions to.
              metric:
                name: "exact_string_match" # Name of the evaluation metric to use.
                average: null # Average the metric over the dataset. Options: ['macro', 'micro']. Works only for 'F1', 'accuracy' etc. Refer to torchmetrics for metrics where this is supported.
                num_classes: null
                class_labels: null
                labels_are_strings: False
        
          optim:
            name: fused_adam
            lr: 2.0e-5
            weight_decay: 0.1
        """
        expected = OmegaConf.create(s)
        assert (
            expected == conf
        ), f"conf/fine_tuning/t5/mnli.yaml must be set to {expected} but it currently is {conf}."
