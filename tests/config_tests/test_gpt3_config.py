from omegaconf import OmegaConf


class TestGPT3Config:
    def test_gpt3_config_0_126b(self):
        conf = OmegaConf.load("conf/search_config/gpt3/0.126b.yaml")
        s = """
        train_settings:
          model_size_in_b: 0.126
          num_nodes: 8
          gpus_per_node: 8
          max_training_days: 2
          limit_search_runs: 100
          output_top_n: 10
          max_steps_per_run: 50
          max_minutes_per_run: 20
          tflops_per_gpu: 140
          num_tokens_in_b: 300
          vocab_size: 51200
          logs: ${base_results_dir}/${search_config_value}
          override_search_num_nodes: null
          tensor_parallel_sizes: null
          pipeline_parallel_sizes: null
          micro_batch_sizes: null
          act_ckpt_layers: null
         
        inference_settings:
          vocab_size: 51200
          start_id: 50256
          end_id: 50256
          max_latency_ms: 500
          input_seq_len: 60
          output_seq_len: 20
          top_n: 10
          logs: ${base_results_dir}/${search_config_value}
          tensor_parallel_sizes: [1, 2]
          pipeline_parallel_sizes: [1]
          max_batch_sizes: [8, 16, 32, 64, 128, 256]
        """
        expected = OmegaConf.create(s)
        assert (
            expected == conf
        ), f"conf/search_config/gpt3/0.126b.yaml must be set to {expected} but it currently is {conf}."

    def test_gpt3_config_5b(self):
        conf = OmegaConf.load("conf/search_config/gpt3/5b.yaml")
        s = """
        train_settings:
          model_size_in_b: 5
          num_nodes: 20
          gpus_per_node: 8
          max_training_days: 5
          limit_search_runs: 100
          output_top_n: 10
          max_steps_per_run: 50
          max_minutes_per_run: 40
          tflops_per_gpu: 140
          num_tokens_in_b: 300
          vocab_size: 51200
          logs: ${base_results_dir}/${search_config_value}
          override_search_num_nodes: null
          tensor_parallel_sizes: null
          pipeline_parallel_sizes: null
          micro_batch_sizes: null
          act_ckpt_layers: null
         
        inference_settings:
          vocab_size: 51200
          start_id: 50256
          end_id: 50256
          input_seq_len: 60
          output_seq_len: 20
          max_latency_ms: 1200
          top_n: 10
          logs: ${base_results_dir}/${search_config_value}
          tensor_parallel_sizes: [1, 2, 4, 8]
          pipeline_parallel_sizes: [1]
          max_batch_sizes: [8, 16, 32, 64, 128, 256]
        """
        expected = OmegaConf.create(s)
        assert (
            expected == conf
        ), f"conf/search_config/gpt3/5b.yaml must be set to {expected} but it currently is {conf}."

    def test_gpt3_config_20b(self):
        conf = OmegaConf.load("conf/search_config/gpt3/20b.yaml")
        s = """
        train_settings:
          model_size_in_b: 20.0
          num_nodes: 80
          gpus_per_node: 8
          max_training_days: 7
          limit_search_runs: 100
          output_top_n: 10
          max_steps_per_run: 50
          max_minutes_per_run: 50
          tflops_per_gpu: 140
          num_tokens_in_b: 300
          vocab_size: 51200
          logs: ${base_results_dir}/${search_config_value}
          override_search_num_nodes: null
          tensor_parallel_sizes: null
          pipeline_parallel_sizes: null
          micro_batch_sizes: null
          act_ckpt_layers: null
         
        inference_settings:
          vocab_size: 51200
          start_id: 50256
          end_id: 50256
          input_seq_len: 60
          output_seq_len: 20
          max_latency_ms: 5000
          top_n: 10
          logs: ${base_results_dir}/${search_config_value}
          tensor_parallel_sizes: [4, 8]
          pipeline_parallel_sizes: [1]
          max_batch_sizes: [4, 8, 16, 32, 64, 128, 256]
        """
        expected = OmegaConf.create(s)
        assert (
            expected == conf
        ), f"conf/search_config/gpt3/20b.yaml must be set to {expected} but it currently is {conf}."

    def test_gpt3_config_40b(self):
        conf = OmegaConf.load("conf/search_config/gpt3/40b.yaml")
        s = """
        train_settings:
          model_size_in_b: 40
          num_nodes: 80
          gpus_per_node: 8
          max_training_days: 13
          limit_search_runs: 100
          output_top_n: 10
          max_steps_per_run: 50
          max_minutes_per_run: 50
          tflops_per_gpu: 140
          num_tokens_in_b: 300
          vocab_size: 51200
          logs: ${base_results_dir}/${search_config_value}
          override_search_num_nodes: null
          tensor_parallel_sizes: null
          pipeline_parallel_sizes: null
          micro_batch_sizes: null
          act_ckpt_layers: null
         
        inference_settings:
          vocab_size: 51200
          start_id: 50256
          end_id: 50256
          input_seq_len: 60
          output_seq_len: 20
          max_latency_ms: 5000
          top_n: 10
          logs: ${base_results_dir}/${search_config_value}
          tensor_parallel_sizes: [8]
          pipeline_parallel_sizes: [1, 2, 4]
          max_batch_sizes: [8, 16, 32, 64, 128, 256]
        """
        expected = OmegaConf.create(s)
        assert (
            expected == conf
        ), f"conf/search_config/gpt3/40b.yaml must be set to {expected} but it currently is {conf}."

    def test_gpt3_config_175b(self):
        conf = OmegaConf.load("conf/search_config/gpt3/175b.yaml")
        s = """
        train_settings:
          model_size_in_b: 175
          num_nodes: 128
          gpus_per_node: 8
          max_training_days: 10
          limit_search_runs: 50
          output_top_n: 10
          max_steps_per_run: 50
          max_minutes_per_run: 60
          tflops_per_gpu: 140
          num_tokens_in_b: 300
          vocab_size: 51200
          logs: ${base_results_dir}/${search_config_value}
          override_search_num_nodes: null
          tensor_parallel_sizes: null
          pipeline_parallel_sizes: null
          micro_batch_sizes: null
          act_ckpt_layers: null
         
        inference_settings:
          vocab_size: 51200
          start_id: 50256
          end_id: 50256
          input_seq_len: 60
          output_seq_len: 20
          max_latency_ms: 10000
          top_n: 10
          logs: ${base_results_dir}/${search_config_value}
          tensor_parallel_sizes: [8]
          pipeline_parallel_sizes: [1, 2, 4, 8]
          max_batch_sizes: [8, 16, 32, 64, 128]
        """
        expected = OmegaConf.create(s)
        assert (
            expected == conf
        ), f"conf/search_config/gpt3/175b.yaml must be set to {expected} but it currently is {conf}."
