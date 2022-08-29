from omegaconf import OmegaConf


class TestEvaluationmT5Config:
    def test_evaluation_mt5_config(self):
        conf = OmegaConf.load("conf/evaluation/mt5/xquad.yaml")
        s = """
        run:
          name: eval_${.task_name}_${.model_train_name}
          time_limit: "04:00:00"
          dependency: "singleton"
          model_train_name: mt5_390m
          task_name: "xquad"
          fine_tuning_dir: ${base_results_dir}/${.model_train_name}/${.task_name}
          results_dir: ${base_results_dir}/${.model_train_name}/${.task_name}_eval
        
        trainer:
          devices: 8
          num_nodes: 1
          accelerator: gpu
          precision: bf16
          logger: False # logger provided by exp_manager
          enable_checkpointing: False
          replace_sampler_ddp: False
          log_every_n_steps: 10
        
        exp_manager:
          explicit_log_dir: ${evaluation.run.results_dir}/results
          exp_dir: null
          name: megatron_mt5_${evaluation.run.task_name}_eval
          create_checkpoint_callback: False
        
        model:
          restore_from_path: ${evaluation.run.fine_tuning_dir}/results/checkpoints/megatron_mt5_xquad.nemo # Path to a finetuned mT5 .nemo file
          gradient_as_bucket_view: True # Allocate gradients in a contiguous bucket to save memory (less fragmentation and buffer memory)
          megatron_amp_O2: False # Enable O2 optimization for megatron amp
        
          data:
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
        """
        expected = OmegaConf.create(s)
        assert (
                expected == conf
        ), f"conf/evaluation/mt5/xnli.yaml must be set to {expected} but it currently is {conf}."


class TestEvaluationT5Config:
    def test_evaluation_t5_config(self):
        conf = OmegaConf.load("conf/evaluation/t5/squad.yaml")
        s = """
        run:
          name: eval_${.task_name}_${.model_train_name}
          time_limit: "04:00:00"
          dependency: "singleton"
          model_train_name: t5_220m
          task_name: "squad"  # SQuAD v1.1
          fine_tuning_dir: ${base_results_dir}/${.model_train_name}/${.task_name}
          results_dir: ${base_results_dir}/${.model_train_name}/${.task_name}_eval
        
        trainer:
          devices: 8
          num_nodes: 1
          accelerator: gpu
          precision: bf16
          logger: False # logger provided by exp_manager
          enable_checkpointing: False
          replace_sampler_ddp: False
          log_every_n_steps: 10
        
        
        exp_manager:
          explicit_log_dir: ${evaluation.run.results_dir}/results
          exp_dir: null
          name: megatron_t5_${fine_tuning.run.task_name}_eval
          create_checkpoint_callback: False
        
        model:
          restore_from_path: ${evaluation.run.fine_tuning_dir}/results/checkpoints/megatron_t5_squad.nemo # Path to a finetuned T5 .nemo file
          gradient_as_bucket_view: True # Allocate gradients in a contiguous bucket to save memory (less fragmentation and buffer memory)
          megatron_amp_O2: False # Enable O2 optimization for megatron amp
        
          data:
            validation_ds:
              src_file_name: ${data_dir}/squad_data/v1.1/train-v1.1_src.txt
              tgt_file_name: ${data_dir}/squad_data/v1.1/train-v1.1_tgt.txt
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
        """
        expected = OmegaConf.create(s)
        assert (
            expected == conf
        ), f"conf/evaluation/t5/mnli_matched.yaml must be set to {expected} but it currently is {conf}."


class TestEvaluationGPT3Config:
    def test_evaluation_gpt3_evaluate_all_config(self):
        conf = OmegaConf.load("conf/evaluation/gpt3/evaluate_all.yaml")
        s = """
        run:
          name: ${.eval_name}_${.model_train_name}
          time_limit: "4:00:00"
          dependency: "singleton"
          nodes: ${divide_ceil:${evaluation.model.model_parallel_size}, 8} # 8 gpus per node
          ntasks_per_node: ${divide_ceil:${evaluation.model.model_parallel_size}, ${.nodes}}
          eval_name: eval_all
          model_train_name: gpt3_5b
          train_dir: ${base_results_dir}/${.model_train_name}
          tasks: all_tasks  # supported: lambada, boolq, race, piqa, hellaswag, winogrande, wikitext2, wikitext103 OR all_tasks
          results_dir: ${base_results_dir}/${.model_train_name}/${.eval_name}
        
        model:
          model_type: nemo-gpt3
          checkpoint_folder: ${evaluation.run.train_dir}/results/checkpoints
          checkpoint_name: latest # latest OR name pattern of a checkpoint (e.g. megatron_gpt-*last.ckpt)
          hparams_file: ${evaluation.run.train_dir}/results/hparams.yaml
          tensor_model_parallel_size: 1
          pipeline_model_parallel_size: 1
          model_parallel_size: ${multiply:${.tensor_model_parallel_size}, ${.pipeline_model_parallel_size}}
          precision: bf16 # must match training precision - 32, 16 or bf16
          eval_batch_size: 4
          vocab_file: ${data_dir}/bpe/vocab.json
          merge_file: ${data_dir}/bpe/merges.txt
        """
        expected = OmegaConf.create(s)
        assert (
            expected == conf
        ), f"conf/evaluation/gpt3/evaluate_all.yaml must be set to {expected} but it currently is {conf}."

    def test_evaluation_gpt3_evaluate_lambada_config(self):
        conf = OmegaConf.load("conf/evaluation/gpt3/evaluate_lambada.yaml")
        s = """
        run:
          name: ${.eval_name}_${.model_train_name}
          time_limit: "4:00:00"
          dependency: "singleton"
          nodes: ${divide_ceil:${evaluation.model.model_parallel_size}, 8} # 8 gpus per node
          ntasks_per_node: ${divide_ceil:${evaluation.model.model_parallel_size}, ${.nodes}}
          eval_name: eval_lambada
          model_train_name: gpt3_5b
          train_dir: ${base_results_dir}/${.model_train_name}
          tasks: lambada  # supported: lambada, boolq, race, piqa, hellaswag, winogrande, wikitext2, wikitext103 OR all_tasks
          results_dir: ${base_results_dir}/${.model_train_name}/${.eval_name}
        
        model:
          model_type: nemo-gpt3
          checkpoint_folder: ${evaluation.run.train_dir}/results/checkpoints
          checkpoint_name: latest # latest OR name pattern of a checkpoint (e.g. megatron_gpt-*last.ckpt)
          hparams_file: ${evaluation.run.train_dir}/results/hparams.yaml
          tensor_model_parallel_size: 2 #1 for 126m, 2 for 5b, 8 for 20b
          pipeline_model_parallel_size: 1
          model_parallel_size: ${multiply:${.tensor_model_parallel_size}, ${.pipeline_model_parallel_size}}
          precision: bf16 # must match training precision - 32, 16 or bf16
          eval_batch_size: 4
          vocab_file: ${data_dir}/bpe/vocab.json
          merge_file: ${data_dir}/bpe/merges.txt
"""
        expected = OmegaConf.create(s)
        assert (
            expected == conf
        ), f"conf/evaluation/gpt3/evaluate_lambada.yaml must be set to {expected} but it currently is {conf}."
