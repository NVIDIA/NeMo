import nemo_run as run
import pytest

from nemo.collections.llm.api import pretrain
from nemo.collections.llm.gpt.data.mock import MockDataModule
from nemo.collections.llm.gpt.model.nemotron import Nemotron4Config22B, NemotronModel
from nemo.collections.llm.recipes import nemotron4_22b
from nemo.lightning import Trainer


# TODO(ahmadki): add parallelism tests
class TestNemotron4_15B:
    @pytest.fixture(scope="class")
    def recipe_module(self):
        return nemotron4_22b

    def test_model(self, recipe_module):
        model_config = recipe_module.model()
        assert isinstance(model_config, run.Config)
        assert model_config.__fn_or_cls__ == NemotronModel
        assert isinstance(model_config.config, run.Config)
        assert model_config.config.__fn_or_cls__ == Nemotron4Config22B

    def test_pretrain_recipe(self, recipe_module):
        recipe = recipe_module.pretrain_recipe()
        assert isinstance(recipe, run.Partial)
        assert recipe.__fn_or_cls__ == pretrain
        assert isinstance(recipe.model, run.Config)
        assert recipe.model.__fn_or_cls__ == NemotronModel
        assert isinstance(recipe.trainer, run.Config)
        assert recipe.trainer.__fn_or_cls__ == Trainer
        assert isinstance(recipe.data, run.Config)
        assert recipe.data.__fn_or_cls__ == MockDataModule
        assert recipe.data.seq_length == 4096
        assert recipe.data.global_batch_size == 32
        assert recipe.data.micro_batch_size == 1

    @pytest.mark.parametrize("num_nodes,num_gpus_per_node", [(1, 8), (2, 4), (4, 2)])
    def test_pretrain_recipe_with_different_configurations(self, recipe_module, num_nodes, num_gpus_per_node):
        recipe = recipe_module.pretrain_recipe(num_nodes=num_nodes, num_gpus_per_node=num_gpus_per_node)
        assert recipe.trainer.num_nodes == num_nodes
        assert recipe.trainer.devices == num_gpus_per_node
