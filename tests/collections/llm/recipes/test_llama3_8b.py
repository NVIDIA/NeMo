import pytest
from nemo.collections.llm.recipes import llama3_8b
from nemo.collections.llm.gpt.model.llama import LlamaModel
from nemo.lightning import Trainer, AutoResume
from nemo.collections.llm.api import pretrain, finetune
from nemo.collections.llm.gpt.data.mock import MockDataModule
from nemo.collections.llm.gpt.data.squad import SquadDataModule
from nemo.collections.llm.peft.lora import LoRA


class TestLlama3Recipe:
    @pytest.fixture(scope="class")
    def recipe_module(self):
        return llama3_8b

    def test_model(self, recipe_module):
        model_config = recipe_module.model()
        assert model_config is not None
        assert model_config.target == LlamaModel
        assert model_config.config.target.__name__ == "Llama3Config8B"

    def test_trainer(self, recipe_module):
        trainer_config = recipe_module.trainer()
        assert trainer_config is not None
        assert trainer_config.target == Trainer
        assert trainer_config.accelerator == "gpu"
        assert trainer_config.devices == 8
        assert trainer_config.num_nodes == 1
        assert trainer_config.max_steps == 1168251

    def test_hf_resume(self, recipe_module):
        resume_config = recipe_module.hf_resume()
        assert resume_config is not None
        assert resume_config.target == AutoResume
        assert resume_config.restore_config.path == "hf://meta-llama/Meta-Llama-3-8B"

    def test_pretrain_recipe(self, recipe_module):
        recipe = recipe_module.pretrain_recipe()
        assert isinstance(recipe, pretrain.Partial)
        assert recipe.model.target == LlamaModel
        assert recipe.trainer.target == Trainer
        assert recipe.data.target == MockDataModule
        assert recipe.data.seq_length == 8192
        assert recipe.data.global_batch_size == 512

    def test_finetune_recipe(self, recipe_module):
        recipe = recipe_module.finetune_recipe()
        assert isinstance(recipe, finetune.Partial)
        assert recipe.model.target == LlamaModel
        assert recipe.trainer.target == Trainer
        assert recipe.data.target == SquadDataModule
        assert recipe.data.seq_length == 8192
        assert recipe.data.global_batch_size == 512
        assert recipe.peft.target == LoRA

    @pytest.mark.parametrize("num_nodes,num_gpus_per_node", [(1, 8), (2, 4), (4, 2)])
    def test_pretrain_recipe_with_different_configurations(self, recipe_module, num_nodes, num_gpus_per_node):
        recipe = recipe_module.pretrain_recipe(num_nodes=num_nodes, num_gpus_per_node=num_gpus_per_node)
        assert recipe.trainer.num_nodes == num_nodes
        assert recipe.trainer.devices == num_gpus_per_node

    def test_pretrain_recipe_performance(self, recipe_module):
        recipe = recipe_module.pretrain_recipe_performance(
            name="test_perf", ckpt_dir="/tmp", num_nodes=1, num_gpus_per_node=8
        )
        assert any(cb.target.__name__ == "MegatronCommOverlapCallback" for cb in recipe.trainer.callbacks)

    def test_trainer_parallelism_options(self, recipe_module):
        trainer_config = recipe_module.trainer(
            tensor_parallelism=2,
            pipeline_parallelism=2,
            context_parallelism=4,
            sequence_parallelism=True
        )
        assert trainer_config.strategy.tensor_model_parallel_size == 2
        assert trainer_config.strategy.pipeline_model_parallel_size == 2
        assert trainer_config.strategy.context_parallel_size == 4
        assert trainer_config.strategy.sequence_parallel is True

    def test_model_config_parameters(self, recipe_module):
        model_config = recipe_module.model()
        llama_config = model_config.config.target()
        assert llama_config.num_layers == 32
        assert llama_config.hidden_size == 4096
        assert llama_config.num_attention_heads == 32
