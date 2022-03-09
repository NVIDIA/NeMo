from omegaconf import OmegaConf


class TestEvaluationT5Config:
    
    def test_evaluation_t5_mnli_matched_config(self):
        conf = OmegaConf.load('conf/evaluation/t5/mnli_matched.yaml')
        s = """
        run:
          name: eval_${.task_name}_${.model_train_name}
          time_limit: "0-04:00:00"
          dependency: "singleton"
          model_train_name: t5_220m
          task_name: "mnli" # Supported task names: "cola", "sst-2", "mrpc", "sts-b", "qqp", "mnli", "qnli", "rte"
          finetuning_results_dir: ${base_results_dir}/${.model_train_name}/${.task_name}
          results_dir: ${base_results_dir}/${.model_train_name}/${.task_name}_eval
        
        trainer:
          gpus: ${divide_ceil:${evaluation.model.model_parallel_size}, ${.nodes}}
          num_nodes: ${divide_ceil:${evaluation.model.model_parallel_size}, 8}
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
          restore_from_finetuned_path: ${evaluation.run.finetuning_results_dir}/checkpoints/megatron_t5_glue.nemo # Path to a finetuned T5 .nemo file
          tensor_model_parallel_size: 1 # 1 for 220m, 2 for 3b
          pipeline_model_parallel_size: 1
          model_parallel_size: ${multiply:${.tensor_model_parallel_size}, ${.pipeline_model_parallel_size}}
          gradient_as_bucket_view: True # Allocate gradients in a contiguous bucket to save memory (less fragmentation and buffer memory)
          megatron_amp_O2: False # Enable O2 optimization for megatron amp
        
          data:
            validation_ds:
              task_name: ${evaluation.run.task_name}
              file_path: ${data_dir}/glue_data/${evaluation.run.task_name}/dev_matched.tsv # Path to the TSV file for MNLI dev. Replace `dev_matched.tsv` with `dev.tsv` if not evaluating MNLI
              batch_size: 32
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
          nodes: ${divide_ceil:${evaluation.model.model_parallel_size}, 8} # 8 gpus per node
          ntasks_per_node: ${divide_ceil:${evaluation.model.model_parallel_size}, ${.nodes}}
          gpus_per_task: 1
          eval_name: eval_all
          convert_name: convert_nemo
          model_train_name: 5b
          tasks: all_tasks  # supported: lambada, boolq, race, piqa, hellaswag, winogrande, wikitext2, wikitext103 OR all_tasks
          results_dir: ${base_results_dir}/${.model_train_name}/${.eval_name}
        
        model:
          type: nemo-gpt3
          # path of checkpoint; must be .nemo file
          checkpoint_path: ${base_results_dir}/${evaluation.run.model_train_name}/${evaluation.run.convert_name}/megatron_gpt.nemo 
          tensor_model_parallel_size: 2 #1 for 126m, 2 for 5b, 8 for 20b
          pipeline_model_parallel_size: 1
          model_parallel_size: ${multiply:${.tensor_model_parallel_size}, ${.pipeline_model_parallel_size}}
          precision: 16 # must match training precision - 32, 16 or bf16
          eval_batch_size: 16
          vocab_file: ${data_dir}/bpe/vocab.json
          merge_file: ${data_dir}/bpe/merges.txt
        """
        expected = OmegaConf.create(s)
        assert expected == conf, f"conf/evaluation/gpt3/evaluate_all.yaml must be set to {expected} but it currently is {conf}."

    def test_evaluation_gpt3_evaluate_lambada_config(self):
        conf = OmegaConf.load('conf/evaluation/gpt3/evaluate_lambada.yaml')
        s = """
        run:
          name: ${.eval_name}_${.model_train_name}
          time_limit: "4:00:00"
          nodes: ${divide_ceil:${evaluation.model.model_parallel_size}, 8} # 8 gpus per node
          ntasks_per_node: ${divide_ceil:${evaluation.model.model_parallel_size}, ${.nodes}}
          gpus_per_task: 1
          eval_name: eval_lambada
          convert_name: convert_nemo
          model_train_name: 5b
          tasks: lambada  # supported: lambada, boolq, race, piqa, hellaswag, winogrande, wikitext2, wikitext103 OR all_tasks
          results_dir: ${base_results_dir}/${.model_train_name}/${.eval_name}
        
        model:
          type: nemo-gpt3
          # path of checkpoint; must be .nemo file
          checkpoint_path: ${base_results_dir}/${evaluation.run.model_train_name}/${evaluation.run.convert_name}/megatron_gpt.nemo 
          tensor_model_parallel_size: 2 #1 for 126m, 2 for 5b, 8 for 20b
          pipeline_model_parallel_size: 1
          model_parallel_size: ${multiply:${.tensor_model_parallel_size}, ${.pipeline_model_parallel_size}}
          precision: 16 # must match training precision - 32, 16 or bf16
          eval_batch_size: 16
          vocab_file: ${data_dir}/bpe/vocab.json
          merge_file: ${data_dir}/bpe/merges.txt
"""
        expected = OmegaConf.create(s)
        assert expected == conf, f"conf/evaluation/gpt3/evaluate_lambada.yaml must be set to {expected} but it currently is {conf}."
