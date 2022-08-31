from omegaconf import OmegaConf


class TestGPT3Config:
    def test_gpt3_config_0_126b(self):
        conf = OmegaConf.load("conf/search_config/gpt3/0.126b.yaml")
        s = """
        train_settings:
          model_size_in_b: 0.126
          num_nodes: 8
          gpus_per_node: 8
          gpu_memory_gb: 80
          max_training_days: 2
          limit_search_runs: 100
          output_top_n: 10
          max_steps_per_run: 50
          max_minutes_per_run: 20
          tflops_per_gpu: 140
          num_tokens_in_b: 300
          vocab_size: 51200
          logs: ${base_results_dir}/${search_config_value}_${.gpu_memory_gb}gb
          tensor_parallel_sizes: auto
          pipeline_parallel_sizes: auto
          micro_batch_sizes: auto
          act_ckpt_layers: auto

        inference_settings:
          run:
            model_type: gpt3
            model_train_name: gpt3_0.126b
            data_type: fp16
            timelimit: 0:30:00
            results_dir: ${base_results_dir}/${search_config_value}_${...train_settings.gpu_memory_gb}gb
            top_n: 10
            max_latency_ms: 500
            tensor_parallel_sizes: [1, 2]
            pipeline_parallel_sizes: [1]
          benchmark:
            input_len: 60
            output_len: 20
            batch_sizes: [8, 16, 32, 64, 128, 256]
            triton_wait_time_s: 300
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
          num_nodes: 16
          gpus_per_node: 8
          gpu_memory_gb: 80
          max_training_days: 5
          limit_search_runs: 100
          output_top_n: 10
          max_steps_per_run: 50
          max_minutes_per_run: 10
          tflops_per_gpu: 140
          num_tokens_in_b: 300
          vocab_size: 51200
          logs: ${base_results_dir}/${search_config_value}_${.gpu_memory_gb}gb
          tensor_parallel_sizes: auto
          pipeline_parallel_sizes: auto
          micro_batch_sizes: auto
          act_ckpt_layers: auto

        inference_settings:
          run:
            model_type: gpt3
            model_train_name: gpt3_5b
            data_type: fp16
            time_limit: 0:30:00
            results_dir: ${base_results_dir}/${search_config_value}_${...train_settings.gpu_memory_gb}gb
            top_n: 10
            max_latency_ms: 500
            tensor_parallel_sizes: [1,2,4]
            pipeline_parallel_sizes: [1]
          benchmark:
            input_len: 60
            output_len: 20
            batch_sizes: [8, 16, 32, 64, 128, 256]
            triton_wait_time_s: 300
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
          num_nodes: 64
          gpus_per_node: 8
          gpu_memory_gb: 80
          max_training_days: 7
          limit_search_runs: 100
          output_top_n: 10
          max_steps_per_run: 50
          max_minutes_per_run: 15
          tflops_per_gpu: 140
          num_tokens_in_b: 300
          vocab_size: 51200
          logs: ${base_results_dir}/${search_config_value}_${.gpu_memory_gb}gb
          tensor_parallel_sizes: auto
          pipeline_parallel_sizes: auto
          micro_batch_sizes: auto
         
        inference_settings:
          run:
            model_type: gpt3
            model_train_name: gpt3_20b
            data_type: fp16
            time_limit: 0:30:00
            results_dir: ${base_results_dir}/${search_config_value}_${...train_settings.gpu_memory_gb}gb
            top_n: 10
            max_latency_ms: 500
            tensor_parallel_sizes: [2,4,8]
            pipeline_parallel_sizes: [1]
          benchmark:
            input_len: 60
            output_len: 20
            batch_sizes: [8, 16, 32, 64, 128, 256]
            triton_wait_time_s: 300
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
          num_nodes: 128
          gpus_per_node: 8
          gpu_memory_gb: 80
          max_training_days: 13
          limit_search_runs: 100
          output_top_n: 10
          max_steps_per_run: 50
          max_minutes_per_run: 15
          tflops_per_gpu: 140
          num_tokens_in_b: 300
          vocab_size: 51200
          logs: ${base_results_dir}/${search_config_value}_${.gpu_memory_gb}gb
          tensor_parallel_sizes: auto
          pipeline_parallel_sizes: auto
          micro_batch_sizes: auto
         
        inference_settings:
          run:
            model_type: gpt3
            model_train_name: gpt3_40b
            data_type: fp16
            time_limit: 0:30:00
            results_dir: ${base_results_dir}/${search_config_value}_${...train_settings.gpu_memory_gb}gb
            top_n: 10
            max_latency_ms: 500
            tensor_parallel_sizes: [2, 4, 8]
            pipeline_parallel_sizes: [1]
          benchmark:
            input_len: 60
            output_len: 20
            batch_sizes: [8, 16, 32, 64, 128, 256]
            triton_wait_time_s: 300
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
          gpu_memory_gb: 80
          max_training_days: 10
          limit_search_runs: 50
          output_top_n: 10
          max_steps_per_run: 50
          max_minutes_per_run: 30
          tflops_per_gpu: 140
          num_tokens_in_b: 300
          vocab_size: 51200
          logs: ${base_results_dir}/${search_config_value}_${.gpu_memory_gb}gb
          tensor_parallel_sizes: auto
          pipeline_parallel_sizes: auto
          micro_batch_sizes: auto
         
        inference_settings:
          run:
            model_type: gpt3
            model_train_name: gpt3_175b
            data_type: fp16
            time_limit: 0:30:00
            results_dir: ${base_results_dir}/${search_config_value}_${...train_settings.gpu_memory_gb}gb
            top_n: 10
            max_latency_ms: 500
            tensor_parallel_sizes: [8]
            pipeline_parallel_sizes: [1,2,4,8]
          benchmark:
            input_len: 60
            output_len: 20
            batch_sizes: [8, 16, 32, 64, 128, 256]
            triton_wait_time_s: 300
        """
        expected = OmegaConf.create(s)
        assert (
            expected == conf
        ), f"conf/search_config/gpt3/175b.yaml must be set to {expected} but it currently is {conf}."
