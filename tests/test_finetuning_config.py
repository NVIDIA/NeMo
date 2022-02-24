from omegaconf import OmegaConf


class TestFinetuningT5Config:
    
    def test_finetuning_t5_config(self):
        conf = OmegaConf.load('conf/finetuning/t5/mnli.yaml')
        s = """
        run:
          name: ${.task_name}_${.model_train_name}
          time_limit: "1-12:00:00"
          dependency: "singleton"
          convert_name: convert_nemo
          model_train_name: t5_220m
          task_name: "mnli"
          results_dir: ${base_results_dir}/${.model_train_name}/${.task_name}

        trainer:
          gpus: 8
          num_nodes: 1
          accelerator: ddp
          precision: 16
          logger: False
          checkpoint_callback: False
          replace_sampler_ddp: False
          max_epochs: 4
          max_steps: null
          log_every_n_steps: 10
          val_check_interval: 500
          accumulate_grad_batches: 1
          gradient_clip_val: 1.0

        exp_manager:
          explicit_log_dir: ${finetuning.run.results_dir}
          exp_dir: null
          name: megatron_t5_glue
          create_wandb_logger: True
          wandb_logger_kwargs:
            project: nemo_t5_glue
            name: ${finetuning.run.name}
          resume_if_exists: True
          resume_ignore_no_checkpoint: True
          create_checkpoint_callback: True
          checkpoint_callback_params:
            monitor: val_acc
            save_top_k: 5
            mode: max
            always_save_nemo: False
            save_nemo_on_train_end: True
            filename: 'megatron_t5--{val_acc:.3f}-{step}'
            model_parallel_size: ${finetuning.model.tensor_model_parallel_size}
            save_best_model: True

        model:
          restore_from_path: ???.nemo
          tensor_model_parallel_size: 1

          data:
            train_ds:
              task_name: ${finetuning.run.task_name}
              file_path: ${data_dir}/MNLI/train.tsv
              batch_size: 16
              shuffle: True
              num_workers: 4
              pin_memory: True
              max_seq_length: 512

            validation_ds:
              task_name: ${finetuning.run.task_name}
              file_path: ${data_dir}/MNLI/dev_matched.tsv
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
