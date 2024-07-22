from nemo.collections.llm.tools.auto_configurator.autoconfig.search_config import search_configs


def get_auto_config(configs):
    auto_configs = []
    for config in configs.values():
        auto_conf_values = config['auto_config'].values()
        auto_configs.append(list(auto_conf_values))

    global_batch_size = config['model'].global_batch_size
    seq_length = config['model'].encoder_seq_length

    return auto_configs, global_batch_size, seq_length


class TestGenerateConfgis:
    def test_gpt_model(self):
        # GPT3 126M
        cfg = {
            "model_type": "gpt3",
            "model_size": 126,
            "model_measure": "M",
            "num_nodes": 8,
            "num_tokens_in_b": 300,
            "tflops_per_gpu": 140,
            "global_batch_size": 256,
            "seq_length": 2048,
            "tensor_parallel_sizes": [4],
            "pipeline_parallel_sizes": [2],
            "micro_batch_sizes": [1, 2],
            "context_parallel_sizes": [1],
            "expert_parallel_sizes": [1],
            "min_model_parallel_size": 8,
            "max_model_parallel_size": 8,
            "max_minutes_per_run": 20,
            "vocab_size": 51200,
            "max_training_days": 2,
            "gpus_per_node": 8,
            "gpu_memory_gb": 80,
        }

        configs = search_configs(cfg)
        auto_configs, global_batch_size, seq_length = get_auto_config(configs)

        assert len(auto_configs) == 2, f"{len(auto_configs)} configurations were generated, but 2 were expected."

        assert auto_configs[0] == [
            4,
            2,
            1,
            1,
            1,
        ], f"[4, 2, 1, 1, 1] is expected configuration output, but got {auto_configs[0]}."

        assert auto_configs[1] == [
            4,
            2,
            2,
            1,
            1,
        ], f"[4, 2, 2, 1, 1] is expected configuration output, but got {auto_configs[1]}."

        assert global_batch_size == 256, f"expected global_batch_size is 256, but got {global_batch_size}."

        assert seq_length == 2048, f"expected seq_length is 2048, but got {seq_length}."

        # GPT3 20B
        cfg = {
            "model_type": "gpt3",
            "model_size": 20,
            "model_measure": "B",
            "num_nodes": 64,
            "num_tokens_in_b": 300,
            "tflops_per_gpu": 140,
            "global_batch_size": 2048,
            "seq_length": 2048,
            "tensor_parallel_sizes": None,
            "pipeline_parallel_sizes": None,
            "micro_batch_sizes": [1],
            "context_parallel_sizes": [1],
            "expert_parallel_sizes": [1],
            "min_model_parallel_size": 16,
            "max_model_parallel_size": 32,
            "max_minutes_per_run": 30,
            "vocab_size": 51200,
            "max_training_days": 8,
            "gpus_per_node": 8,
            "gpu_memory_gb": 80,
        }

        configs = search_configs(cfg)
        auto_configs, _, _ = get_auto_config(configs)

        assert len(auto_configs) == 1, f"{len(auto_configs)} configurations were generated, but 1 were expected."

        assert auto_configs[0] == [
            11,
            4,
            4,
            1,
            1,
            1,
        ], f"[11, 4, 4, 1, 1, 1] is expected configuration output, but got {auto_configs[0]}."

        # GPT3 175B
        cfg = {
            "model_type": "gpt3",
            "model_size": 175,
            "model_measure": "B",
            "num_nodes": 128,
            "num_tokens_in_b": 300,
            "tflops_per_gpu": 140,
            "global_batch_size": 2048,
            "seq_length": 2048,
            "tensor_parallel_sizes": None,
            "pipeline_parallel_sizes": None,
            "micro_batch_sizes": None,
            "context_parallel_sizes": [1],
            "expert_parallel_sizes": [1],
            "min_model_parallel_size": 64,
            "max_model_parallel_size": 64,
            "max_minutes_per_run": 30,
            "vocab_size": 51200,
            "max_training_days": 16,
            "gpus_per_node": 8,
            "gpu_memory_gb": 80,
        }

        configs = search_configs(cfg)
        auto_configs, _, _ = get_auto_config(configs)

        assert len(auto_configs) == 3, f"{len(auto_configs)} configurations were generated, but 3 were expected."

        assert auto_configs[0] == [
            12,
            8,
            8,
            1,
            1,
            1,
        ], f"[12, 8, 8, 1, 1, 1] is expected configuration output, but got {auto_configs[0]}."

        assert auto_configs[1] == [
            12,
            8,
            8,
            2,
            1,
            1,
        ], f"[12, 8, 8, 2, 1, 1] is expected configuration output, but got {auto_configs[1]}."

        assert auto_configs[2] == [
            12,
            8,
            8,
            4,
            1,
            1,
        ], f"[12, 8, 8, 4, 1, 1] is expected configuration output, but got {auto_configs[2]}."

    def test_llama_model(self):
        # Llama2 7B
        cfg = {
            "model_type": "llama",
            "model_size": 7,
            "model_version": 2,
            "model_measure": "B",
            "num_nodes": 16,
            "num_tokens_in_b": 300,
            "tflops_per_gpu": 140,
            "global_batch_size": 2048,
            "seq_length": 4096,
            "tensor_parallel_sizes": [1],
            "pipeline_parallel_sizes": [1],
            "micro_batch_sizes": [1],
            "context_parallel_sizes": [1, 2],
            "expert_parallel_sizes": [1],
            "min_model_parallel_size": 1,
            "max_model_parallel_size": 16,
            "max_minutes_per_run": 20,
            "vocab_size": 32000,
            "max_training_days": 8,
            "gpus_per_node": 8,
            "gpu_memory_gb": 80,
        }

        configs = search_configs(cfg)
        auto_configs, _, _ = get_auto_config(configs)

        assert len(auto_configs) == 2, f"{len(auto_configs)} configurations were generated, but 2 were expected."

        assert auto_configs[0] == [
            1,
            1,
            1,
            1,
            1,
        ], f"[1, 1, 1, 1, 1] is expected configuration output, but got {auto_configs[0]}."

        assert auto_configs[1] == [
            1,
            1,
            1,
            2,
            1,
        ], f"[1, 1, 1, 2, 1] is expected configuration output, but got {auto_configs[1]}."

        # Llama3 8B
        cfg = {
            "model_type": "llama",
            "model_size": 8,
            "model_version": 3,
            "model_measure": "B",
            "num_nodes": 16,
            "num_tokens_in_b": 300,
            "tflops_per_gpu": 140,
            "global_batch_size": 2048,
            "seq_length": 8192,
            "tensor_parallel_sizes": [2],
            "pipeline_parallel_sizes": [2],
            "micro_batch_sizes": [2],
            "context_parallel_sizes": [2],
            "expert_parallel_sizes": [1, 2, 4],
            "min_model_parallel_size": 1,
            "max_model_parallel_size": 16,
            "max_minutes_per_run": 20,
            "vocab_size": 32000,
            "max_training_days": 8,
            "gpus_per_node": 8,
            "gpu_memory_gb": 80,
        }

        configs = search_configs(cfg)
        auto_configs, _, _ = get_auto_config(configs)

        assert len(auto_configs) == 1, f"{len(auto_configs)} configurations were generated, but 1 were expected."

        assert auto_configs[0] == [
            2,
            2,
            2,
            2,
            1,
        ], f"[2, 2, 2, 2, 1] is expected configuration output, but got {auto_configs[0]}."

        # Llama3 70B
        cfg = {
            "model_type": "llama",
            "model_size": 70,
            "model_version": 3,
            "model_measure": "B",
            "num_nodes": 64,
            "num_tokens_in_b": 300,
            "tflops_per_gpu": 140,
            "global_batch_size": 2048,
            "seq_length": 8192,
            "tensor_parallel_sizes": [1, 2],
            "pipeline_parallel_sizes": [1, 2],
            "micro_batch_sizes": [1],
            "context_parallel_sizes": [2],
            "expert_parallel_sizes": [1, 2, 4],
            "min_model_parallel_size": 1,
            "max_model_parallel_size": 4,
            "max_minutes_per_run": 30,
            "vocab_size": 32000,
            "max_training_days": 16,
            "gpus_per_node": 8,
            "gpu_memory_gb": 80,
        }

        configs = search_configs(cfg)
        auto_configs, global_batch_size, seq_length = get_auto_config(configs)

        assert len(auto_configs) == 3, f"{len(auto_configs)} configurations were generated, but 3 were expected."

        assert auto_configs[0] == [
            1,
            1,
            1,
            2,
            1,
        ], f"[1, 1, 1, 2, 1] is expected configuration output, but got {auto_configs[0]}."

        assert auto_configs[1] == [
            1,
            2,
            1,
            2,
            1,
        ], f"[1, 2, 1, 2, 1] is expected configuration output, but got {auto_configs[1]}."

        assert auto_configs[2] == [
            2,
            1,
            1,
            2,
            1,
        ], f"[2, 1, 1, 2, 1] is expected configuration output, but got {auto_configs[2]}."

        assert global_batch_size == 2048, f"expected global_batch_size is 2048, but got {global_batch_size}."

        assert seq_length == 8192, f"expected seq_length is 8192, but got {seq_length}."

    def test_mixtral_model(self):
        # Mixtral 8x7B
        cfg = {
            "model_type": "mixtral",
            "model_size": 7,
            "model_version": 8,
            "model_measure": "B",
            "num_nodes": 16,
            "num_tokens_in_b": 300,
            "tflops_per_gpu": 140,
            "global_batch_size": 2048,
            "seq_length": 4096,
            "tensor_parallel_sizes": [2, 3, 4],
            "pipeline_parallel_sizes": None,
            "micro_batch_sizes": [2],
            "context_parallel_sizes": [1],
            "expert_parallel_sizes": [2, 4],
            "min_model_parallel_size": None,
            "max_model_parallel_size": None,
            "max_minutes_per_run": 20,
            "vocab_size": 32000,
            "max_training_days": 8,
            "gpus_per_node": 8,
            "gpu_memory_gb": 80,
        }

        configs = search_configs(cfg)
        auto_configs, global_batch_size, seq_length = get_auto_config(configs)

        assert len(auto_configs) == 4, f"{len(auto_configs)} configurations were generated, but 4 were expected."

        assert auto_configs[0] == [
            2,
            1,
            2,
            1,
            2,
        ], f"[2, 1, 2, 1, 2] is expected configuration output, but got {auto_configs[0]}."

        assert auto_configs[1] == [
            2,
            1,
            2,
            1,
            4,
        ], f"[2, 1, 2, 1, 4] is expected configuration output, but got {auto_configs[1]}."

        assert auto_configs[2] == [
            2,
            2,
            2,
            1,
            2,
        ], f"[2, 2, 2, 1, 2] is expected configuration output, but got {auto_configs[2]}."

        assert auto_configs[3] == [
            4,
            1,
            2,
            1,
            2,
        ], f"[4, 1, 2, 1, 2] is expected configuration output, but got {auto_configs[3]}."

        assert global_batch_size == 2048, f"expected global_batch_size is 2048, but got {global_batch_size}."

        assert seq_length == 4096, f"expected seq_length is 4096, but got {seq_length}."

    def test_mistral_model(self):
        # Mistral 7B
        cfg = {
            "model_type": "mistral",
            "model_size": 7,
            "model_version": None,
            "model_measure": "B",
            "num_nodes": 16,
            "num_tokens_in_b": 300,
            "tflops_per_gpu": 140,
            "global_batch_size": 2048,
            "seq_length": 16384,
            "tensor_parallel_sizes": [1, 2, 3],
            "pipeline_parallel_sizes": [2, 11, 17],
            "micro_batch_sizes": [1, 256],
            "context_parallel_sizes": [1],
            "expert_parallel_sizes": [2, 13],
            "min_model_parallel_size": None,
            "max_model_parallel_size": None,
            "max_minutes_per_run": 20,
            "vocab_size": 32000,
            "max_training_days": 8,
            "gpus_per_node": 8,
            "gpu_memory_gb": 80,
        }

        configs = search_configs(cfg)
        auto_configs, global_batch_size, seq_length = get_auto_config(configs)

        assert len(auto_configs) == 2, f"{len(auto_configs)} configurations were generated, but 2 were expected."

        assert auto_configs[0] == [
            1,
            2,
            1,
            1,
            2,
        ], f"[1, 2, 1, 1, 2] is expected configuration output, but got {auto_configs[0]}."

        assert auto_configs[1] == [
            2,
            2,
            1,
            1,
            2,
        ], f"[2, 2, 1, 1, 2] is expected configuration output, but got {auto_configs[1]}."

        assert global_batch_size == 2048, f"expected global_batch_size is 2048, but got {global_batch_size}."

        assert seq_length == 16384, f"expected seq_length is 16384, but got {seq_length}."
