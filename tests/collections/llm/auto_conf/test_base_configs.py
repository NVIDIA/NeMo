import re

from megatron.core.optimizer import OptimizerConfig

from nemo.collections.common.tokenizers import AutoTokenizer, SentencePieceTokenizer
from nemo.collections.llm.tools.auto_configurator import base_configs
from nemo.collections.llm.utils import Config


def get_class_name(config_cls):
    match = re.search(r'<Config\[(\w+)\(', repr(config_cls))
    config_cls_name = None
    if match:
        config_cls_name = match.group(1)

    return config_cls_name


class TestBaseConfigs:
    def test_gpt3_base_config(self):
        model_cls = getattr(base_configs, "GPT")

        # GPT3 126M
        model_126m = model_cls(size=126, measure="M", cfg={"nemo_sdk": True})
        config_cls = model_126m.get_model_config()
        config_cls_name = get_class_name(config_cls)
        assert (
            config_cls_name == "GPTConfig126M"
        ), "the name of the config class for the GPT3 126M model should be 'GPTConfig126M'."

        # GPT3 5B
        model_5b = model_cls(size=5)
        config_cls = model_5b.get_model_config()
        assert (
            config_cls.__class__.__name__ == "GPTConfig5B"
        ), "the name of the config class for the GPT3 5B model should be 'GPTConfig5B'."

        # GPT3 7B
        model_7b = model_cls(size=7, cfg={"nemo_sdk": True})
        config_cls = model_7b.get_model_config()
        config_cls_name = get_class_name(config_cls)
        assert (
            config_cls_name == "GPTConfig7B"
        ), "the name of the config class for the GPT3 7B model should be 'GPTConfig7B'."

        # GPT3 20B
        model_20b = model_cls(size=20)
        config_cls = model_20b.get_model_config()
        assert (
            config_cls.__class__.__name__ == "GPTConfig20B"
        ), "the name of the config class for the GPT3 20B model should be 'GPTConfig20B'."

        # GPT3 40B
        model_40b = model_cls(size=40)
        config_cls = model_40b.get_model_config()
        assert (
            config_cls.__class__.__name__ == "GPTConfig40B"
        ), "the name of the config class for the GPT3 40B model should be 'GPTConfig40B'."

        # GPT3 175B
        model_175b = model_cls(size=175, cfg={"nemo_sdk": True})
        config_cls = model_175b.get_model_config()
        config_cls_name = get_class_name(config_cls)
        assert (
            config_cls_name == "GPTConfig175B"
        ), "the name of the config class for the GPT3 175B model should be 'GPTConfig175B'."

        try:
            model_111b = model_cls(size=111)
            config_cls = model_111b.get_model_config()
            config_cls_name = get_class_name(config_cls)
            assert (
                config_cls_name == "GPTConfig111B"
            ), "the name of the config class for the GPT3 111B model should be 'GPTConfig111B'."
        except AttributeError:
            None

    def test_llama_base_config(self):
        model_cls = getattr(base_configs, "Llama")

        # Llama2_7B
        model_7b = model_cls(size=7, cfg={"nemo_sdk": True})
        config_cls = model_7b.get_model_config()
        config_cls_name = get_class_name(config_cls)
        assert (
            config_cls_name == "Llama2Config7B"
        ), "the name of the config class for the Llama2 7B model should be 'Llama2Config7B'."

        # Llama2_13B
        model_13b = model_cls(size=13)
        config_cls = model_13b.get_model_config()
        assert (
            config_cls.__class__.__name__ == "Llama2Config13B"
        ), "the name of the config class for the Llama2 13B model should be 'Llama2Config13B'."

        # Llama2_70B
        model_70b = model_cls(size=70)
        config_cls = model_70b.get_model_config()
        assert (
            config_cls.__class__.__name__ == "Llama2Config70B"
        ), "the name of the config class for the Llama2 70B model should be 'Llama2Config70B'."

        # Llama3_70B
        model_70b = model_cls(size=70, version=3)
        config_cls = model_70b.get_model_config()
        assert (
            config_cls.__class__.__name__ == "Llama3Config70B"
        ), "the name of the config class for the Llama3 70B model should be 'Llama3Config70B'."

        # Llama3_8B
        model_8b = model_cls(size=8, version=3, cfg={"nemo_sdk": True})
        config_cls = model_8b.get_model_config()
        config_cls_name = get_class_name(config_cls)
        assert (
            config_cls_name == "Llama3Config8B"
        ), "the name of the config class for the Llama3 8B model should be 'Llama3Config8B'."

    def test_mixtral_base_config(self):
        model_cls = getattr(base_configs, "Mixtral")

        # Mixtral 8x7B
        model_7b = model_cls(size=7)
        config_cls = model_7b.get_model_config()
        assert (
            config_cls.__class__.__name__ == "MixtralConfig8x7B"
        ), "the name of the config class for the Mixtral 8x7B model should be 'MixtralConfig8x7B'."

    def test_mistral_base_config(self):
        model_cls = getattr(base_configs, "Mistral")

        # Mistral 7B
        model_7b = model_cls(size=7, cfg={"nemo_sdk": True})
        config_cls = model_7b.get_model_config()
        config_cls_name = get_class_name(config_cls)
        assert (
            config_cls_name == "MistralConfig7B"
        ), "the name of the config class for the Mistral 7B model should be 'MistralConfig7B'."

    def test_basic_base_config(self):
        model_cls = getattr(base_configs.basic, "Basic")

        # Basic model class
        model = model_cls(measure="M")

        assert model.name == None
        assert model.version == None
        assert model.size == None
        assert model.measure == "M"
        assert model.cfg == {}

    def test_custom_base_config(self):
        model = base_configs.custom(name="Llama", cfg={})

        assert model.name == "Llama"
        assert model.version == 2
        assert model.size == 7
        assert model.measure == "B"
        assert model.cfg == {}

    def test_trainer_config(self):
        model_cls = getattr(base_configs, "GPT")

        model_126m = model_cls(size=126, measure="M")
        trainer_config_source = model_126m.get_trainer_config()

        trainer_config_target = {
            "accelerator": "gpu",
            "logger": False,
            "enable_checkpointing": False,
            "use_distributed_sampler": False,
            "max_epochs": None,
            "log_every_n_steps": 1,
            "limit_val_batches": 1,
            "limit_test_batches": 1,
            "accumulate_grad_batches": 1,
            "gradient_clip_val": 1.0,
            "num_nodes": None,
            "devices": None,
            "max_steps": None,
            "val_check_interval": None,
        }

        assert (
            trainer_config_target == trainer_config_source
        ), f"{trainer_config_target} is expected trainer config but got {trainer_config_source}"

    def test_data_config(self):
        model_cls = getattr(base_configs, "Llama")

        model_70b = model_cls(size=70)
        data_config_source = model_70b.get_data_config()

        data_config_target = {
            "paths": None,
            "seq_length": None,
            "global_batch_size": None,
            "num_workers": 2,
            "split": "99990,8,2",
            "index_mapping_dir": None,
        }

        assert (
            data_config_target == data_config_source
        ), f"{data_config_target} is expected data config but got {data_config_source}"

    def test_optim_config(self):
        model_cls = getattr(base_configs, "Mixtral")

        model_7b = model_cls(size=7)
        optim_config_source = model_7b.get_optim_config()

        optim_config_target = OptimizerConfig(
            optimizer='adam',
            lr=1e-4,
            min_lr=1e-5,
            use_distributed_optimizer=True,
            bf16=True,
            adam_beta1=0.9,
            adam_beta2=0.95,
            overlap_grad_reduce=False,
            overlap_param_gather=True,
        )

        assert (
            optim_config_target == optim_config_source
        ), f"{optim_config_target} is expected optim config but got {optim_config_source}"

    def test_optim_config_nemo_sdk(self):
        model_cls = getattr(base_configs, "Mixtral")

        model_7b = model_cls(size=7, cfg={"nemo_sdk": True})
        optim_config_source = model_7b.get_optim_config()

        optim_config_target = Config(
            OptimizerConfig,
            optimizer='adam',
            lr=1e-4,
            min_lr=1e-5,
            use_distributed_optimizer=True,
            bf16=True,
            adam_beta1=0.9,
            adam_beta2=0.95,
            overlap_grad_reduce=False,
            overlap_param_gather=True,
        )

        assert (
            optim_config_target == optim_config_source
        ), f"{optim_config_target} is expected optim config but got {optim_config_source}"

    def test_run_config(self):
        model_cls = getattr(base_configs, "Mistral")

        model_7b = model_cls(size=7)
        run_config_source = model_7b.get_run_config()

        run_config_target = {
            "name": f"Mistral_7B",
            "results_dir": None,
            "time_limit": "0-00:30:00",
        }

        assert (
            run_config_target == run_config_source
        ), f"{run_config_target} is expected run config but got {run_config_source}"

    def test_tokenizer_config(self):
        # Mistral
        model_cls = getattr(base_configs, "Mistral")

        model_7b = model_cls(size=7)
        tokenizer_config_source = model_7b.get_tokenizer_config()

        tokenizer_config_target = {
            "class": AutoTokenizer,
            "name": "mistralai/Mistral-7B-v0.1",
        }

        assert (
            tokenizer_config_target == tokenizer_config_source
        ), f"{tokenizer_config_target} is expected tokenizer config but got {tokenizer_config_source}"

        # Mixtral
        model_cls = getattr(base_configs, "Mixtral")

        model_7b = model_cls(size=7)
        tokenizer_config_source = model_7b.get_tokenizer_config()

        tokenizer_config_target = {
            "class": AutoTokenizer,
            "name": "mistralai/Mixtral-8x7B-v0.1",
        }

        assert (
            tokenizer_config_target == tokenizer_config_source
        ), f"{tokenizer_config_target} is expected tokenizer config but got {tokenizer_config_source}"

        # Llama
        model_cls = getattr(base_configs, "Llama")

        model_8b = model_cls(size=8, version=3)
        tokenizer_config_source = model_8b.get_tokenizer_config()

        tokenizer_config_target = {
            "class": SentencePieceTokenizer,
            "path": None,
        }

        assert (
            tokenizer_config_target == tokenizer_config_source
        ), f"{tokenizer_config_target} is expected tokenizer config but got {tokenizer_config_source}"

        # GPT
        model_cls = getattr(base_configs, "GPT")

        model_5b = model_cls(size=5, version=3)
        tokenizer_config_source = model_5b.get_tokenizer_config()

        tokenizer_config_target = {
            "class": AutoTokenizer,
            "name": "GPT2BPETokenizer",
        }

        assert (
            tokenizer_config_target == tokenizer_config_source
        ), f"{tokenizer_config_target} is expected tokenizer config but got {tokenizer_config_source}"
