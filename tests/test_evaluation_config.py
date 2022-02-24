from omegaconf import OmegaConf


class TestEvaluationT5Config:
    
    def test_evaluation_t5_mnli_matched_config(self):
        conf = OmegaConf.load('conf/evaluation/t5/mnli.yaml')
        s = """
        run:
          name: eval_${.task_name}_${.model_train_name}
          time_limit: "0-04:00:00"
          dependency: "singleton"
          convert_name: convert_nemo
          model_train_name: t5_220m
          task_name: "mnli"  # supported: ???
          results_dir: ${base_results_dir}/${.model_train_name}/${.task_name}_eval
        
        trainer:
          gpus: 1
          num_nodes: 1
          accelerator: ddp
          precision: 16
          logger: False # logger provided by exp_manager
          checkpoint_callback: False
          replace_sampler_ddp: False
          log_every_n_steps: 10
        
        
        exp_manager:
          explicit_log_dir: ${evaluation.run.results_dir}
          exp_dir: null
          name: megatron_t5_glue_eval
          create_checkpoint_callback: False
        
        model:
          restore_from_finetuned_path: ${finetuning.run.results_dir}/checkpoints/megatron_t5_mnli.nemo # Path to a finetuned T5 .nemo file
          tensor_model_parallel_size: 1
          gradient_as_bucket_view: True # Allocate gradients in a contiguous bucket to save memory (less fragmentation and buffer memory)
          megatron_amp_O2: False # Enable O2 optimization for megatron amp
        
          data:
            validation_ds:
              task_name: ${evaluation.run.task_name}
              file_path: ${data_dir}/MNLI/dev_matched.tsv # Path to the TSV file for MNLI dev ex: '/raid/Data/GLUE/MNLI/dev_matched.tsv'
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
          name: eval_${.task_name}_${.model_train_name}
          time_limit: "0-04:00:00"
          dependency: "singleton"
          convert_name: convert_nemo
          model_train_name: t5_220m
          task_name: "mnli"
          results_dir: ${base_results_dir}/${.model_train_name}/${.task_name}_eval
        
        trainer:
          gpus: 1
          num_nodes: 1
          accelerator: ddp
          precision: 16
          logger: False # logger provided by exp_manager
          checkpoint_callback: False
          replace_sampler_ddp: False
          log_every_n_steps: 10
        
        
        exp_manager:
          explicit_log_dir: ${evaluation.run.results_dir}
          exp_dir: null
          name: megatron_t5_glue_eval
          create_checkpoint_callback: False
        
        model:
          restore_from_finetuned_path: ${finetuning.run.results_dir}/checkpoints/megatron_t5_mnli.nemo # Path to a finetuned T5 .nemo file
          tensor_model_parallel_size: 1
          gradient_as_bucket_view: True # Allocate gradients in a contiguous bucket to save memory (less fragmentation and buffer memory)
          megatron_amp_O2: False # Enable O2 optimization for megatron amp
        
          data:
            validation_ds:
              task_name: ${evaluation.run.task_name}
              file_path: ${data_dir}/glue_data/MNLI/dev_matched.tsv # Path to the TSV file for MNLI dev ex: '/raid/Data/GLUE/MNLI/dev_matched.tsv'
              batch_size: 32
              shuffle: False
              num_workers: 4
              pin_memory: True
              max_seq_length: 512
        """
        expected = OmegaConf.create(s)
        assert expected == conf, f"conf/evaluation/gpt3/evaluate_lambada.yaml must be set to {expected} but it currently is {conf}."
