from nemo.collections.llm.tools.auto_configurator import base_configs

class TestBaseConfigs:
    def test_gpt3_base_config(self):
        model_cls = getattr(base_configs, "GPT")
        
        #GPT3 126M
        model_126m = model_cls(size=126, measure="M")
        assert (model_126m.get_model_config().__name__ == "GPTConfig126M"), \
        "the name of the config class for the GPT3 126M model should be 'GPTConfig126M'"
    
        #GPT3 5B
        model_5b = model_cls(size=5)
        assert (model_5b.get_model_config().__name__ == "GPTConfig5B"), \
        "the name of the config class for the GPT3 5B model should be 'GPTConfig5B'"

        #GPT3 7B
        model_7b = model_cls(size=7)
        assert (model_7b.get_model_config().__name__ == "GPTConfig7B"), \
        "the name of the config class for the GPT3 7B model should be 'GPTConfig7B'"

        #GPT3 20B
        model_20b = model_cls(size=20)
        assert (model_20b.get_model_config().__name__ == "GPTConfig20B"), \
        "the name of the config class for the GPT3 20B model should be 'GPTConfig20B'"

        #GPT3 40B
        model_40b = model_cls(size=40)
        assert (model_40b.get_model_config().__name__ == "GPTConfig40B"), \
        "the name of the config class for the GPT3 40B model should be 'GPTConfig40B'"

        #GPT3 175B
        model_175b = model_cls(size=175)
        assert (model_175b.get_model_config().__name__ == "GPTConfig175B"), \
        "the name of the config class for the GPT3 175B model should be 'GPTConfig175B'"

        try:
            model_111b = model_cls(size=111)
            assert (model_111b.get_model_config().__name__ == "GPTConfig111B"), \
            "the name of the config class for the GPT3 111B model should be 'GPTConfig111B'"
        except AttributeError:
            None

    def test_llama_base_config(self):
        model_cls = getattr(base_configs, "Llama")

        #Llama2_7B
        model_7b = model_cls(size=7)
        assert (model_7b.get_model_config().__name__ == "Llama2Config7B"), \
        "the name of the config class for the Llama2 7B model should be 'Llama2Config7B'"

        #Llama2_13B
        model_13b = model_cls(size=13)
        assert (model_13b.get_model_config().__name__ == "Llama2Config13B"), \
        "the name of the config class for the Llama2 13B model should be 'Llama2Config13B'"

        #Llama2_70B
        model_70b = model_cls(size=70)
        assert (model_70b.get_model_config().__name__ == "Llama2Config70B"), \
        "the name of the config class for the Llama2 70B model should be 'Llama2Config70B'"

        #Llama3_70B
        model_70b = model_cls(size=70, version=3)
        assert (model_70b.get_model_config().__name__ == "Llama3Config70B"), \
        "the name of the config class for the Llama3 70B model should be 'Llama3Config70B'"

        #Llama3_8B
        model_8b = model_cls(size=8, version=3)
        assert (model_8b.get_model_config().__name__ == "Llama3Config8B"), \
        "the name of the config class for the Llama3 8B model should be 'Llama3Config8B'"

    def test_mixtral_base_config(self):
        model_cls = getattr(base_configs, "Mixtral")

        #Mixtral 8x7B
        model_7b = model_cls(size=7)
        assert (model_7b.get_model_config().__name__ == "MixtralConfig8x7B"), \
        "the name of the config class for the Mixtral 8x7B model should be 'MixtralConfig8x7B'"

    def test_mistral_base_config(self):
        model_cls = getattr(base_configs, "Mistral")

        #Mistral 7B
        model_7b = model_cls(size=7)
        assert (model_7b.get_model_config().__name__ == "MistralConfig7B"), \
        "the name of the config class for the Mistral 7B model should be 'MistralConfig7B'"
    
    def test_basic_base_config(self):
        model_cls = getattr(base_configs.basic, "Basic")

        #Basic model class
        model = model_cls(measure="M")

        assert model.name == None
        assert model.version == None
        assert model.size == None
        assert model.measure == "M"
        assert model.cfg == {}

