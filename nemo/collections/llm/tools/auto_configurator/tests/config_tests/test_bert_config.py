from omegaconf import OmegaConf


class TestBERTConfig:
    def test_bert_config_0_11b(self):
        conf = OmegaConf.load("conf/search_config/bert/0.11b.yaml")
        s = """
        train_settings:
          model_size_in_b: 0.11 # unit in billion parameters
          num_nodes: 8
          gpus_per_node: 8
          gpu_memory_gb: 80  # Memory per GPU, in GB. Currently 40GB and 80GB A100s supported.
          max_training_days: 2 # unit in days
          limit_search_runs: 100 # Max number of runs to be launched in parallel for grid search.
          output_top_n: 10  # The result will print the top N fastest training configs.
          max_steps_per_run: 50 # Max steps per run for the grid search.
          max_minutes_per_run: 20 # minutes per run for the grid search.
          tflops_per_gpu: 140  # Estimated tflops per GPU.
          num_tokens_in_b: 1800  # Unit in billions, typically 300B for GPT3 models.
          vocab_size: 30522
          seq_length: 512 # available seq_length list for BERT models: [512]
          custom_config: null # path to custom .yaml model config instead of using auto-generated
          logs: ${base_results_dir}/${search_config_value}_${.gpu_memory_gb}gb  # Example base_results_dir/gpt3/126m
          tensor_parallel_sizes: auto  # auto to use our recommendation, or a list, such as [1, 2, 4, 8]
          pipeline_parallel_sizes: auto  # auto to use our recommendation, or a list, such as [1, 2, 4, 8, 10]
          min_model_parallel_size: auto  # auto to use our recommendation, or a value for the minimum desired parallelism
          max_model_parallel_size: auto  # auto to use our recommendation, or a value for the maximum desired parallelism
          micro_batch_sizes: auto  # auto to use our recommendation, or a list, such as [1, 2, 4, 8, 16]
          act_ckpt_layers: auto  # auto to use our recommendation, or a list, such as [0, 1, 2, 3]
        """
        expected = OmegaConf.create(s)
        assert (
            expected == conf
        ), f"conf/search_config/bert/0.11b.yaml must be set to {expected} but it currently is {conf}."

    def test_bert_config_4b(self):
        conf = OmegaConf.load("conf/search_config/bert/4b.yaml")
        s = """
        train_settings:
          model_size_in_b: 4 # unit in billion parameters
          num_nodes: 16
          gpus_per_node: 8
          gpu_memory_gb: 80  # Memory per GPU, in GB. Currently 40GB and 80GB A100s supported.
          max_training_days: 7 # unit in days
          limit_search_runs: 100 # Max number of runs to be launched in parallel for grid search.
          output_top_n: 10  # The result will print the top N fastest training configs.
          max_steps_per_run: 50 # Max steps per run for the grid search.
          max_minutes_per_run: 20 # minutes per run for the grid search.
          tflops_per_gpu: 140  # Estimated tflops per GPU.
          num_tokens_in_b: 1800  # Unit in billions, typically 300B for GPT3 models.
          vocab_size: 30522
          seq_length: 512 # available seq_length list for BERT models: [512]
          custom_config: null # path to custom .yaml model config instead of using auto-generated
          logs: ${base_results_dir}/${search_config_value}_${.gpu_memory_gb}gb  # Example base_results_dir/gpt3/126m
          tensor_parallel_sizes: auto  # auto to use our recommendation, or a list, such as [1, 2, 4, 8]
          pipeline_parallel_sizes: auto  # auto to use our recommendation, or a list, such as [1, 2, 4, 8, 10]
          min_model_parallel_size: auto  # auto to use our recommendation, or a value for the minimum desired parallelism
          max_model_parallel_size: auto  # auto to use our recommendation, or a value for the maximum desired parallelism
          micro_batch_sizes: auto  # auto to use our recommendation, or a list, such as [1, 2, 4, 8, 16]
          act_ckpt_layers: auto  # auto to use our recommendation, or a list, such as [0, 1, 2, 3]
 
        """
        expected = OmegaConf.create(s)
        assert (
            expected == conf
        ), f"conf/search_config/bert/4b.yaml must be set to {expected} but it currently is {conf}."

    def test_bert_config_20b(self):
        conf = OmegaConf.load("conf/search_config/bert/20b.yaml")
        s = """
        train_settings:
          model_size_in_b: 20 # unit in billion parameters
          num_nodes: 64
          gpus_per_node: 8
          gpu_memory_gb: 80  # Memory per GPU, in GB. Currently 40GB and 80GB A100s supported.
          max_training_days: 12 # unit in days
          limit_search_runs: 100 # Max number of runs to be launched in parallel for grid search.
          output_top_n: 10  # The result will print the top N fastest training configs.
          max_steps_per_run: 50 # Max steps per run for the grid search.
          max_minutes_per_run: 30 # minutes per run for the grid search.
          tflops_per_gpu: 140  # Estimated tflops per GPU.
          num_tokens_in_b: 1800  # Unit in billions, typically 300B for GPT3 models.
          vocab_size: 30522
          seq_length: 512 # available seq_length list for BERT models: [512]
          custom_config: null # path to custom .yaml model config instead of using auto-generated
          logs: ${base_results_dir}/${search_config_value}_${.gpu_memory_gb}gb  # Example base_results_dir/gpt3/126m
          tensor_parallel_sizes: auto  # auto to use our recommendation, or a list, such as [1, 2, 4, 8]
          pipeline_parallel_sizes: auto  # auto to use our recommendation, or a list, such as [1, 2, 4, 8, 10]
          min_model_parallel_size: auto  # auto to use our recommendation, or a value for the minimum desired parallelism
          max_model_parallel_size: auto  # auto to use our recommendation, or a value for the maximum desired parallelism
          micro_batch_sizes: auto  # auto to use our recommendation, or a list, such as [1, 2, 4, 8, 16]
          act_ckpt_layers: auto  # auto to use our recommendation, or a list, such as [0, 1, 2, 3]
        """
        expected = OmegaConf.create(s)
        assert (
            expected == conf
        ), f"conf/search_config/bert/20b.yaml must be set to {expected} but it currently is {conf}."

    def test_bert_config_100b(self):
        conf = OmegaConf.load("conf/search_config/bert/100b.yaml")
        s = """
        train_settings:
          model_size_in_b: 100 # unit in billion parameters
          num_nodes: 128
          gpus_per_node: 8
          gpu_memory_gb: 80  # Memory per GPU, in GB. Currently 40GB and 80GB A100s supported.
          max_training_days: 50 # unit in days
          limit_search_runs: 100 # Max number of runs to be launched in parallel for grid search.
          output_top_n: 10  # The result will print the top N fastest training configs.
          max_steps_per_run: 50 # Max steps per run for the grid search.
          max_minutes_per_run: 40 # minutes per run for the grid search.
          tflops_per_gpu: 150  # Estimated tflops per GPU.
          num_tokens_in_b: 1800  # Unit in billions, typically 300B for GPT3 models.
          vocab_size: 30522
          seq_length: 512 # available seq_length list for BERT models: [512]
          custom_config: null # path to custom .yaml model config instead of using auto-generated
          logs: ${base_results_dir}/${search_config_value}_${.gpu_memory_gb}gb  # Example base_results_dir/gpt3/126m
          tensor_parallel_sizes: auto  # auto to use our recommendation, or a list, such as [1, 2, 4, 8]
          pipeline_parallel_sizes: auto  # auto to use our recommendation, or a list, such as [1, 2, 4, 8, 10]
          min_model_parallel_size: auto  # auto to use our recommendation, or a value for the minimum desired parallelism
          max_model_parallel_size: auto  # auto to use our recommendation, or a value for the maximum desired parallelism
          micro_batch_sizes: auto  # auto to use our recommendation, or a list, such as [1, 2, 4, 8, 16]
          act_ckpt_layers: auto  # auto to use our recommendation, or a list, such as [0, 1, 2, 3]
        """
        expected = OmegaConf.create(s)
        assert (
            expected == conf
        ), f"conf/search_config/bert/20b.yaml must be set to {expected} but it currently is {conf}."
