from nemo.collections.llm.tools.auto_configurator import AutoConfigurator


def get_auto_config(configs):
    auto_configs = []
    for config in configs.values():
        auto_conf_values = config['auto_config'].values()
        auto_configs.append(list(auto_conf_values))

    global_batch_size = config['model'].global_batch_size
    seq_length = config['model'].seq_length

    return auto_configs, global_batch_size, seq_length


class TestGenerateConfgis:
    def test_gpt_model(self):
        # GPT3 126M
        runner = AutoConfigurator(
            model_type="gpt3",
            model_size=126,
            model_measure="M",
            num_nodes=8,
            seq_length=2048,
            global_batch_size=256,
            tensor_parallel_sizes=[4],
            pipeline_parallel_sizes=[2],
            micro_batch_sizes=[1,2],
            context_parallel_sizes=[1],
            expert_parallel_sizes=[1],
            min_model_parallel_size=8,
            max_model_parallel_size=8,
            data_paths=[""],
        )

        configs = runner.generate_configs()
        auto_configs, global_batch_size, seq_length = get_auto_config(configs)

        assert len(auto_configs) == 2, f"{len(auto_configs)} configurations were generated but 2 were expected."

        assert auto_configs[0] == [
            4,
            2,
            1,
            1,
            1,
        ], f"[4, 2, 1, 1, 1] is expected configuration output but got {auto_configs[0]}."

        assert auto_configs[1] == [
            4,
            2,
            2,
            1,
            1,
        ], f"[4, 2, 2, 1, 1] is expected configuration output but got {auto_configs[1]}."

        assert global_batch_size == 256, f"expected global_batch_size is 256 but got {global_batch_size}."

        assert seq_length == 2048, f"expected seq_length is 2048 but got {seq_length}."

        # GPT3 20B
        runner = AutoConfigurator(
            model_type="gpt3",
            model_size=20,
            num_nodes=64,
            seq_length=2048,
            global_batch_size=2048,
            micro_batch_sizes=[1],
            context_parallel_sizes=[1],
            expert_parallel_sizes=[1],
            min_model_parallel_size=16,
            max_model_parallel_size=32,
            max_training_days=8,
            data_paths=[""],
        )

        configs = runner.generate_configs()
        auto_configs, _, _ = get_auto_config(configs)

        assert len(auto_configs) == 1, f"{len(auto_configs)} configurations were generated but 1 were expected."

        assert auto_configs[0] == [
            11,
            4,
            4,
            1,
            1,
            1,
        ], f"[11, 4, 4, 1, 1, 1] is expected configuration output but got {auto_configs[0]}."

        # GPT3 175B
        runner = AutoConfigurator(
            model_type="gpt3",
            model_size=175,
            num_nodes=128,
            seq_length=2048,
            global_batch_size=2048,
            context_parallel_sizes=[1],
            expert_parallel_sizes=[1],
            min_model_parallel_size=64,
            max_model_parallel_size=64,
            max_training_days=16,
            data_paths=[""],
        )

        configs = runner.generate_configs()
        auto_configs, _, _ = get_auto_config(configs)

        assert len(auto_configs) == 3, f"{len(auto_configs)} configurations were generated but 3 were expected."

        assert auto_configs[0] == [
            12,
            8,
            8,
            1,
            1,
            1,
        ], f"[12, 8, 8, 1, 1, 1] is expected configuration output but got {auto_configs[0]}."

        assert auto_configs[1] == [
            12,
            8,
            8,
            2,
            1,
            1,
        ], f"[12, 8, 8, 2, 1, 1] is expected configuration output but got {auto_configs[1]}."

        assert auto_configs[2] == [
            12,
            8,
            8,
            4,
            1,
            1,
        ], f"[12, 8, 8, 4, 1, 1] is expected configuration output but got {auto_configs[2]}."

    def test_llama_model(self):
        # Llama2 7B
        runner = AutoConfigurator(
            model_type="llama",
            model_size=7,
            model_version=2,
            num_nodes=16,
            seq_length=4096,
            global_batch_size=2048,
            tensor_parallel_sizes=[1],
            pipeline_parallel_sizes=[1],
            micro_batch_sizes=[1],
            context_parallel_sizes=[1, 2],
            expert_parallel_sizes=[1],
            min_model_parallel_size=1,
            max_model_parallel_size=16,
            max_training_days=8,
            data_paths=[""],
        )

        configs = runner.generate_configs()
        auto_configs, _, _ = get_auto_config(configs)

        assert len(auto_configs) == 2, f"{len(auto_configs)} configurations were generated but 2 were expected."

        assert auto_configs[0] == [
            1,
            1,
            1,
            1,
            1,
        ], f"[1, 1, 1, 1, 1] is expected configuration output but got {auto_configs[0]}."

        assert auto_configs[1] == [
            1,
            1,
            1,
            2,
            1,
        ], f"[1, 1, 1, 2, 1] is expected configuration output but got {auto_configs[1]}."

        # Llama3 8B
        runner = AutoConfigurator(
            model_type="llama",
            model_size=8,
            model_version=3,
            num_nodes=16,
            seq_length=8192,
            global_batch_size=2048,
            tensor_parallel_sizes=[2],
            pipeline_parallel_sizes=[2],
            micro_batch_sizes=[2],
            context_parallel_sizes=[2],
            expert_parallel_sizes=[1, 2, 4],
            min_model_parallel_size=1,
            max_model_parallel_size=16,
            max_training_days=8,
            data_paths=[""],
        )

        configs = runner.generate_configs()
        auto_configs, _, _ = get_auto_config(configs)

        assert len(auto_configs) == 1, f"{len(auto_configs)} configurations were generated but 1 were expected."

        assert auto_configs[0] == [
            2,
            2,
            2,
            2,
            1,
        ], f"[2, 2, 2, 2, 1] is expected configuration output but got {auto_configs[0]}."

        # Llama3 70B
        runner = AutoConfigurator(
            model_type="llama",
            model_size=70,
            model_version=3,
            num_nodes=64,
            seq_length=8192,
            global_batch_size=2048,
            tensor_parallel_sizes=[1, 2],
            pipeline_parallel_sizes=[1, 2],
            micro_batch_sizes=[1],
            context_parallel_sizes=[2],
            expert_parallel_sizes=[1, 2, 4],
            min_model_parallel_size=1,
            max_model_parallel_size=4,
            max_training_days=30,
            data_paths=[""],
        )

        configs = runner.generate_configs()
        auto_configs, global_batch_size, seq_length = get_auto_config(configs)

        assert len(auto_configs) == 3, f"{len(auto_configs)} configurations were generated but 3 were expected."

        assert auto_configs[0] == [
            1,
            1,
            1,
            2,
            1,
        ], f"[1, 1, 1, 2, 1] is expected configuration output but got {auto_configs[0]}."

        assert auto_configs[1] == [
            1,
            2,
            1,
            2,
            1,
        ], f"[1, 2, 1, 2, 1] is expected configuration output but got {auto_configs[1]}."

        assert auto_configs[2] == [
            2,
            1,
            1,
            2,
            1,
        ], f"[2, 1, 1, 2, 1] is expected configuration output but got {auto_configs[2]}."

        assert global_batch_size == 2048, f"expected global_batch_size is 2048 but got {global_batch_size}."

        assert seq_length == 8192, f"expected seq_length is 8192 but got {seq_length}."

    def test_mixtral_model(self):
        # Mixtral 8x7B
        runner = AutoConfigurator(
            model_type="mixtral",
            model_size=7,
            model_version=8,
            num_nodes=16,
            seq_length=4096,
            global_batch_size=2048,
            tensor_parallel_sizes=[2, 3, 4],
            micro_batch_sizes=[2],
            expert_parallel_sizes=[2, 4],
            data_paths=[""],
        )

        configs = runner.generate_configs()
        auto_configs, global_batch_size, seq_length = get_auto_config(configs)

        assert len(auto_configs) == 4, f"{len(auto_configs)} configurations were generated but 4 were expected."

        assert auto_configs[0] == [
            2,
            1,
            2,
            1,
            2,
        ], f"[2, 1, 2, 1, 2] is expected configuration output but got {auto_configs[0]}."

        assert auto_configs[1] == [
            2,
            1,
            2,
            1,
            4,
        ], f"[2, 1, 2, 1, 4] is expected configuration output but got {auto_configs[1]}."

        assert auto_configs[2] == [
            2,
            2,
            2,
            1,
            2,
        ], f"[2, 2, 2, 1, 2] is expected configuration output but got {auto_configs[2]}."

        assert auto_configs[3] == [
            4,
            1,
            2,
            1,
            2,
        ], f"[4, 1, 2, 1, 2] is expected configuration output but got {auto_configs[3]}."

        assert global_batch_size == 2048, f"expected global_batch_size is 2048 but got {global_batch_size}."

        assert seq_length == 4096, f"expected seq_length is 4096 but got {seq_length}."

    def test_mistral_model(self):
        # Mistral 7B
        runner = AutoConfigurator(
            model_type="mistral",
            model_size=7,
            num_nodes=16,
            seq_length=16384,
            global_batch_size=2048,
            tensor_parallel_sizes=[1, 2, 3],
            pipeline_parallel_sizes=[2, 11, 17],
            micro_batch_sizes=[1, 256],
            expert_parallel_sizes=[2, 13],
            data_paths=[""],
        )

        configs = runner.generate_configs()
        auto_configs, global_batch_size, seq_length = get_auto_config(configs)

        assert len(auto_configs) == 2, f"{len(auto_configs)} configurations were generated but 2 were expected."

        assert auto_configs[0] == [
            1,
            2,
            1,
            1,
            2,
        ], f"[1, 2, 1, 1, 2] is expected configuration output but got {auto_configs[0]}."

        assert auto_configs[1] == [
            2,
            2,
            1,
            1,
            2,
        ], f"[2, 2, 1, 1, 2] is expected configuration output but got {auto_configs[1]}."

        assert global_batch_size == 2048, f"expected global_batch_size is 2048 but got {global_batch_size}."

        assert seq_length == 16384, f"expected seq_length is 16384 but got {seq_length}."

    def test_custom_model(self):
        # Custom 1B
        runner = AutoConfigurator(
            model_type="llama",
            num_nodes=4,
            seq_length=512,
            global_batch_size=1024,
            tensor_parallel_sizes=[1, 2],
            pipeline_parallel_sizes=[2, 4],
            micro_batch_sizes=[1, 256],
            context_parallel_sizes=[2, 22],
            expert_parallel_sizes=[1, 13],
            min_model_parallel_size=2,
            max_model_parallel_size=8,
            vocab_size=32000,
            max_training_days=7,
            custom_model=True,
            data_paths=[""],
        )

        configs = runner.generate_configs()
        auto_configs, global_batch_size, seq_length = get_auto_config(configs)

        assert len(auto_configs) == 2, f"{len(auto_configs)} configurations were generated but 2 were expected."
        print(auto_configs)
        assert auto_configs[0] == [
            1,
            2,
            1,
            2,
            1,
        ], f"[1, 2, 1, 2, 1] is expected configuration output but got {auto_configs[0]}."

        assert auto_configs[1] == [
            2,
            2,
            1,
            2,
            1,
        ], f"[2, 2, 1, 2, 1] is expected configuration output but got {auto_configs[1]}."

        assert global_batch_size == 1024, f"expected global_batch_size is 1024 but got {global_batch_size}."

        assert seq_length == 512, f"expected seq_length is 512 but got {seq_length}."
