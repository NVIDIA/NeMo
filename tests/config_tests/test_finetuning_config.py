from omegaconf import OmegaConf


class TestFinetuningT5Config:
    
    def test_finetuning_t5_config(self):
        conf = OmegaConf.load('conf/finetuning/t5/mnli.yaml')
        s = """
        run:
          name: ${.task_name}_${.model_train_name}
          time_limit: "04:00:00"
          dependency: "singleton"
          convert_name: convert_nemo
          model_train_name: t5_220m
          task_name: "mnli" # Supported task names: "cola", "sst-2", "mrpc", "sts-b", "qqp", "mnli", "qnli", "rte"
          results_dir: ${base_results_dir}/${.model_train_name}/${.task_name}
        
        trainer:
          gpus: 8
          num_nodes: 1
          accelerator: ddp
          precision: bf16
          logger: False # logger provided by exp_manager
          checkpoint_callback: False
          replace_sampler_ddp: False
          max_epochs: 4
          max_steps: null # consumed_samples = global_step * micro_batch_size * data_parallel_size * accumulate_grad_batches
          log_every_n_steps: 10
          val_check_interval: 500
          accumulate_grad_batches: 1
          gradient_clip_val: 1.0
        
        exp_manager:
          explicit_log_dir: ${finetuning.run.results_dir}
          exp_dir: null
          name: megatron_t5_glue
          create_wandb_logger: False
          wandb_logger_kwargs:
            project: nemo_t5_glue
            name: ${finetuning.run.name}
          resume_if_exists: True
          resume_ignore_no_checkpoint: True
          create_checkpoint_callback: True
          checkpoint_callback_params:
            monitor: validation_acc
            save_top_k: 5
            mode: max
            always_save_nemo: False
            save_nemo_on_train_end: True # Set to true for subsequent validation runs.
            filename: 'megatron_t5--{validation_acc:.3f}-{step}'
            model_parallel_size: ${finetuning.model.tensor_model_parallel_size}
            save_best_model: True
        
        model: # For different finetuning tasks, tuning the hyper parameters accordingly; below is only for MNLI
          restore_from_path: ${base_results_dir}/${finetuning.run.model_train_name}/${finetuning.run.convert_name}/megatron_t5.nemo # Path to a trained T5 .nemo file
          tensor_model_parallel_size: 1
          gradient_as_bucket_view: True # Allocate gradients in a contiguous bucket to save memory (less fragmentation and buffer memory)
          megatron_amp_O2: False # Enable O2 optimization for megatron amp
        
          data:
            train_ds:
              task_name: ${finetuning.run.task_name}
              file_path: ${data_dir}/glue_data/${finetuning.run.task_name}/train.tsv # Path to the TSV file for MNLI train
              batch_size: 16
              shuffle: True
              num_workers: 4
              pin_memory: True
              max_seq_length: 512
        
            validation_ds:
              task_name: ${finetuning.run.task_name}
              file_path: ${data_dir}/glue_data/${finetuning.run.task_name}/dev_matched.tsv # Path to the TSV file for MNLI dev. Replace `dev_matched.tsv` with `dev.tsv` if not finetuning MNLI
              batch_size: 16
              shuffle: False
              num_workers: 4
              pin_memory: True
              max_seq_length: 512
        
          optim:
            name: fused_adam
            lr: 2.0e-5
            weight_decay: 0.1
            sched:
              name: WarmupAnnealing
              min_lr: 0.0
              last_epoch: -1
              warmup_ratio: 0.0
        """
        expected = OmegaConf.create(s)
        assert expected == conf, f"conf/finetuning/t5/mnli.yaml must be set to {expected} but it currently is {conf}."
