from omegaconf import OmegaConf


class TestEvaluationT5Config:
    
    def test_evaluation_t5_mnli_matched_config(self):
        conf = OmegaConf.load('conf/evaluation/t5/mnli_matched.yaml')
        s = """
        run:
          name: ${.task_name}_${.model_train_name}
          time_limit: "1-12:00:00"
          dependency: "singleton"
          convert_name: convert_nemo
          model_train_name: t5_220m
          eval_name: "eval_mnli"
          results_dir: ${base_results_dir}/${.model_train_name}/${.eval_name}

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
          explicit_log_dir: ${evaluation.run.results_dir}
          exp_dir: null
          name: megatron_t5_glue
          create_checkpoint_callback: False

        model:
          restore_from_path: ???.nemo
          tensor_model_parallel_size: 1

          data:
            validation_ds:
              task_name: ${evaluation.run.task_name}
              file_path: ${data_dir}/MNLI/dev_matched.tsv
              batch_size: 16
              shuffle: False
              num_workers: 4
              pin_memory: True
              max_seq_length: 512
        """
        expected = OmegaConf.create(s)
        assert expected == conf, f"conf/evaluation/t5/mnli_matched.yaml must be set to {expected} but it currently is {conf}."


class TestEvaluationGPT3Config:
    
    def test_evaluation_gpt3_evaluate_all_config(self):
        conf = OmegaConf.load('conf/evaluation/gpt3/evaluate_all.yaml')
        s = """
        run:
          name: ${.eval_name}_${.model_train_name}
          time_limit: "4:00:00"
          nodes: 1
          ntasks_per_node: ${evaluation.model.tensor_model_parallel_size}
          gpus_per_task: 1
          eval_name: eval_all
          convert_name: convert_nemo
          model_train_name: 5b
          tasks: all_tasks
          results_dir: ${base_results_dir}/${.model_train_name}/${.eval_name}

        model:
          type: nemo-gpt3
          checkpoint_path: ${base_results_dir}/${evaluation.run.model_train_name}/${evaluation.run.convert_name}/megatron_gpt.nemo 
          tensor_model_parallel_size: 2
          pipeline_model_parallel_size: 1
          precision: 16
          eval_batch_size: 16
        """
        expected = OmegaConf.create(s)
        assert expected == conf, f"conf/evaluation/gpt3/evaluate_all.yaml must be set to {expected} but it currently is {conf}."
    
    def test_evaluation_gpt3_evaluate_lambada_config(self):
        conf = OmegaConf.load('conf/evaluation/gpt3/evaluate_lambada.yaml')
        s = """
        run:
          name: ${.eval_name}_${.model_train_name}
          time_limit: "4:00:00"
          nodes: 1
          ntasks_per_node: ${evaluation.model.tensor_model_parallel_size}
          gpus_per_task: 1
          eval_name: eval_lambada
          convert_name: convert_nemo
          model_train_name: 5b
          tasks: lambada
          results_dir: ${base_results_dir}/${.model_train_name}/${.eval_name}

        model:
          type: nemo-gpt3
          checkpoint_path: ${base_results_dir}/${evaluation.run.model_train_name}/${evaluation.run.convert_name}/megatron_gpt.nemo 
          tensor_model_parallel_size: 2
          pipeline_model_parallel_size: 1
          precision: 16
          eval_batch_size: 16
"""
        expected = OmegaConf.create(s)
        assert expected == conf, f"conf/evaluation/gpt3/evaluate_lambada.yaml must be set to {expected} but it currently is {conf}."
