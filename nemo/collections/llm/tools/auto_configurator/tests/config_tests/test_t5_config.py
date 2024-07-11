from omegaconf import OmegaConf


class TestT5Config:
    def test_t5_config_0_22b(self):
        conf = OmegaConf.load("conf/search_config/t5/0.22b.yaml")
        s = """
        train_settings:
          model_size_in_b: 0.22
          num_nodes: 4
          gpus_per_node: 8
          gpu_memory_gb: 80
          max_training_days: 4
          limit_search_runs: 100
          output_top_n: 10
          max_steps_per_run: 50
          max_minutes_per_run: 10
          tflops_per_gpu: 140
          num_tokens_in_b: 1000
          vocab_size: 29000
          seq_length: 512 # available seq_length list for T5 models: [512]
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
        ), f"conf/search_config/t5/0.22b.yaml must be set to {expected} but it currently is {conf}."

    def test_t5_config_2_8b(self):
        conf = OmegaConf.load("conf/search_config/t5/2.8b.yaml")
        s = """
        train_settings:
          model_size_in_b: 2.8
          num_nodes: 20
          gpus_per_node: 8
          gpu_memory_gb: 80
          max_training_days: 16
          limit_search_runs: 100
          output_top_n: 10
          max_steps_per_run: 50
          max_minutes_per_run: 35
          tflops_per_gpu: 140
          num_tokens_in_b: 1000
          vocab_size: 29000
          seq_length: 512 # available seq_length list for T5 models: [512]
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
        ), f"conf/search_config/t5/2.8b.yaml must be set to {expected} but it currently is {conf}."

    def test_t5_config_11b(self):
        conf = OmegaConf.load("conf/search_config/t5/11b.yaml")
        s = """
        train_settings:
          model_size_in_b: 11
          num_nodes: 20
          gpus_per_node: 8
          gpu_memory_gb: 80
          max_training_days: 45
          limit_search_runs: 100
          output_top_n: 10
          max_steps_per_run: 50
          max_minutes_per_run: 50
          tflops_per_gpu: 140
          num_tokens_in_b: 1000
          vocab_size: 29000
          seq_length: 512 # available seq_length list for T5 models: [512]
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
        ), f"conf/search_config/t5/11b.yaml must be set to {expected} but it currently is {conf}."

    def test_t5_config_23_5b(self):
        conf = OmegaConf.load("conf/search_config/t5/23.5b.yaml")
        s = """
        train_settings:
          model_size_in_b: 23.5
          num_nodes: 40
          gpus_per_node: 8
          gpu_memory_gb: 80
          max_training_days: 48
          limit_search_runs: 100
          output_top_n: 10
          max_steps_per_run: 50
          max_minutes_per_run: 80
          tflops_per_gpu: 140
          num_tokens_in_b: 1000
          vocab_size: 29000
          seq_length: 512 # available seq_length list for T5 models: [512]
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
        ), f"conf/search_config/t5/23.5b.yaml must be set to {expected} but it currently is {conf}."

    def test_t5_config_41_2b(self):
        conf = OmegaConf.load("conf/search_config/t5/41.2b.yaml")
        s = """
        train_settings:
          model_size_in_b: 41.2
          num_nodes: 40
          gpus_per_node: 8
          gpu_memory_gb: 80
          max_training_days: 85
          limit_search_runs: 100
          output_top_n: 10
          max_steps_per_run: 50
          max_minutes_per_run: 90
          tflops_per_gpu: 140
          num_tokens_in_b: 1000
          vocab_size: 29000
          seq_length: 512 # available seq_length list for T5 models: [512]
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
        ), f"conf/search_config/t5/41.2b.yaml must be set to {expected} but it currently is {conf}."
