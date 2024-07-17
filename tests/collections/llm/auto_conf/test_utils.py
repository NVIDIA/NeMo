from nemo.collections.llm.tools.auto_configurator.autoconfig.base_config import (
    _estimate_training_time,
    calculate_model_size,
)
from nemo.collections.llm.tools.auto_configurator.autoconfig.utils import calculate_model_size_params


class TestUtils:
    def test_calculate_model_size(self):
        # GPT
        model_size = calculate_model_size(
            8,
            7,
            None,
            140,
            300,
            "gpt3",
        )
        assert model_size == 0.28

        # Llama
        model_size = calculate_model_size(
            128,
            30,
            None,
            100,
            3000,
            "llama",
        )
        assert model_size == 1.38

        # Mixtral
        model_size = calculate_model_size(
            256,
            20,
            None,
            140,
            600,
            "mixtral",
        )
        assert model_size == 12.9

        # Mistral
        model_size = calculate_model_size(
            1028,
            30,
            None,
            240,
            100,
            "mistral",
        )
        assert model_size == 799.37

    def test_calculate_train_time(self):
        # GPT
        train_time = _estimate_training_time(
            175,
            1024,
            140,
            300,
            "gpt3",
        )
        assert train_time == 33.91

        # Llama
        train_time = _estimate_training_time(
            35,
            512,
            60,
            3000,
            "llama",
        )
        assert train_time == 316.48

        # Mixtral
        train_time = _estimate_training_time(
            0.8,
            128,
            140,
            1000,
            "mixtral",
        )
        assert train_time == 4.13

        # Mistral
        train_time = _estimate_training_time(
            11,
            24,
            60,
            250,
            "mistral",
        )
        assert train_time == 176.83

    def test_calculate_model_params(self):
        # GPT
        params = calculate_model_size_params(
            40,
            51200,
            2048,
            "gpt3",
        )
        assert params == (48, 8192, 64, None, None, 8e-05)

        # Llama
        params = calculate_model_size_params(
            70,
            32000,
            8192,
            "llama",
        )
        assert params == (56, 10240, 80, None, None, 7e-05)

        # Mixtral
        params = calculate_model_size_params(
            30,
            32000,
            4096,
            "mixtral",
        )
        assert params == (36, 8192, 64, None, None, 8e-05)

        # Mistral
        params = calculate_model_size_params(
            0.5,
            32000,
            4096,
            "mistral",
        )
        assert params == (16, 1536, 16, None, None, 0.00025)
