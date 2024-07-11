from omegaconf import OmegaConf


class TestmT5Config:
    def test_mt5_config_0_17b(self):
        conf = OmegaConf.load("conf/search_config/mt5/0.17b.yaml")
        s = """
        train_settings:
          model_size_in_b: 0.17
          num_nodes: 4
          gpus_per_node: 8
          gpu_memory_gb: 80
          max_training_days: 4
          limit_search_runs: 100
          output_top_n: 10
          max_steps_per_run: 50
          max_minutes_per_run: 12
          tflops_per_gpu: 140
          num_tokens_in_b: 1000
          vocab_size: 250000
          seq_length: 512 # available seq_length list for MT5 models: [512]
          custom_config: null # path to custom .yaml model config instead of using auto-generated
          logs: ${base_results_dir}/${search_config_value}_${.gpu_memory_gb}gb
          tensor_parallel_sizes: auto
          pipeline_parallel_sizes: auto
          min_model_parallel_size: auto  # auto to use our recommendation, or a value for the minimum desired parallelism
          max_model_parallel_size: auto  # auto to use our recommendation, or a value for the maximum desired parallelism
          micro_batch_sizes: auto
          act_ckpt_layers: auto
        """
        expected = OmegaConf.create(s)
        assert (
            expected == conf
        ), f"conf/search_config/mt5/0.17b.yaml must be set to {expected} but it currently is {conf}."

    def test_mt5_config_0_39b(self):
        conf = OmegaConf.load("conf/search_config/mt5/0.39b.yaml")
        s = """
        train_settings:
          model_size_in_b: 0.39
          num_nodes: 8
          gpus_per_node: 8
          gpu_memory_gb: 80
          max_training_days: 5
          limit_search_runs: 100
          output_top_n: 10
          max_steps_per_run: 50
          max_minutes_per_run: 15
          tflops_per_gpu: 140
          num_tokens_in_b: 1000
          vocab_size: 250000
          seq_length: 512 # available seq_length list for MT5 models: [512]
          custom_config: null # path to custom .yaml model config instead of using auto-generated
          logs: ${base_results_dir}/${search_config_value}_${.gpu_memory_gb}gb
          tensor_parallel_sizes: auto
          pipeline_parallel_sizes: auto
          min_model_parallel_size: auto  # auto to use our recommendation, or a value for the minimum desired parallelism
          max_model_parallel_size: auto  # auto to use our recommendation, or a value for the maximum desired parallelism
          micro_batch_sizes: auto
          act_ckpt_layers: auto
        """
        expected = OmegaConf.create(s)
        assert (
            expected == conf
        ), f"conf/search_config/mt5/0.39b.yaml must be set to {expected} but it currently is {conf}."

    def test_mt5_config_3_2b(self):
        conf = OmegaConf.load("conf/search_config/mt5/3.2b.yaml")
        s = """
        train_settings:
          model_size_in_b: 3.2
          num_nodes: 20
          gpus_per_node: 8
          gpu_memory_gb: 80
          max_training_days: 14
          limit_search_runs: 100
          output_top_n: 10
          max_steps_per_run: 50
          max_minutes_per_run: 35
          tflops_per_gpu: 140
          num_tokens_in_b: 1000
          vocab_size: 250000
          seq_length: 512 # available seq_length list for MT5 models: [512]
          custom_config: null # path to custom .yaml model config instead of using auto-generated
          logs: ${base_results_dir}/${search_config_value}_${.gpu_memory_gb}gb
          tensor_parallel_sizes: auto
          pipeline_parallel_sizes: auto
          min_model_parallel_size: auto  # auto to use our recommendation, or a value for the minimum desired parallelism
          max_model_parallel_size: auto  # auto to use our recommendation, or a value for the maximum desired parallelism
          micro_batch_sizes: auto
          act_ckpt_layers: auto
        """
        expected = OmegaConf.create(s)
        assert (
            expected == conf
        ), f"conf/search_config/mt5/3.2b.yaml must be set to {expected} but it currently is {conf}."

    def test_mt5_config_11_9b(self):
        conf = OmegaConf.load("conf/search_config/mt5/11.9b.yaml")
        s = """
        train_settings:
          model_size_in_b: 11.9
          num_nodes: 20
          gpus_per_node: 8
          gpu_memory_gb: 80
          max_training_days: 50
          limit_search_runs: 100
          output_top_n: 10
          max_steps_per_run: 50
          max_minutes_per_run: 50
          tflops_per_gpu: 140
          num_tokens_in_b: 1000
          vocab_size: 250000
          seq_length: 512 # available seq_length list for MT5 models: [512]
          custom_config: null # path to custom .yaml model config instead of using auto-generated
          logs: ${base_results_dir}/${search_config_value}_${.gpu_memory_gb}gb
          tensor_parallel_sizes: auto
          pipeline_parallel_sizes: auto
          min_model_parallel_size: auto  # auto to use our recommendation, or a value for the minimum desired parallelism
          max_model_parallel_size: auto  # auto to use our recommendation, or a value for the maximum desired parallelism
          micro_batch_sizes: auto
          act_ckpt_layers: auto
        """
        expected = OmegaConf.create(s)
        assert (
            expected == conf
        ), f"conf/search_config/mt5/11.9b.yaml must be set to {expected} but it currently is {conf}."

    def test_mt5_config_24_65b(self):
        conf = OmegaConf.load("conf/search_config/mt5/24.65b.yaml")
        s = """
        train_settings:
          model_size_in_b: 24.65
          num_nodes: 40
          gpus_per_node: 8
          gpu_memory_gb: 80
          max_training_days: 55
          limit_search_runs: 100
          output_top_n: 10
          max_steps_per_run: 50
          max_minutes_per_run: 60
          tflops_per_gpu: 140
          num_tokens_in_b: 1000
          vocab_size: 250000
          seq_length: 512 # available seq_length list for MT5 models: [512]
          custom_config: null # path to custom .yaml model config instead of using auto-generated
          logs: ${base_results_dir}/${search_config_value}_${.gpu_memory_gb}gb
          tensor_parallel_sizes: auto
          pipeline_parallel_sizes: auto
          min_model_parallel_size: auto  # auto to use our recommendation, or a value for the minimum desired parallelism
          max_model_parallel_size: auto  # auto to use our recommendation, or a value for the maximum desired parallelism
          micro_batch_sizes: auto
          act_ckpt_layers: auto
        """
        expected = OmegaConf.create(s)
        assert (
            expected == conf
        ), f"conf/search_config/mt5/24.65b.yaml must be set to {expected} but it currently is {conf}."

    def test_mt5_config_42_54b(self):
        conf = OmegaConf.load("conf/search_config/mt5/42.54b.yaml")
        s = """
        train_settings:
          model_size_in_b: 42.54
          num_nodes: 40
          gpus_per_node: 8
          gpu_memory_gb: 80
          max_training_days: 90
          limit_search_runs: 100
          output_top_n: 10
          max_steps_per_run: 50
          max_minutes_per_run: 80
          tflops_per_gpu: 140
          num_tokens_in_b: 1000
          vocab_size: 250000
          seq_length: 512 # available seq_length list for MT5 models: [512]
          custom_config: null # path to custom .yaml model config instead of using auto-generated
          logs: ${base_results_dir}/${search_config_value}_${.gpu_memory_gb}gb
          tensor_parallel_sizes: auto
          pipeline_parallel_sizes: auto
          min_model_parallel_size: auto  # auto to use our recommendation, or a value for the minimum desired parallelism
          max_model_parallel_size: auto  # auto to use our recommendation, or a value for the maximum desired parallelism
          micro_batch_sizes: auto
          act_ckpt_layers: auto
        """
        expected = OmegaConf.create(s)
        assert (
            expected == conf
        ), f"conf/search_config/mt5/42.54b.yaml must be set to {expected} but it currently is {conf}."
