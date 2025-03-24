from nemo.collections.diffusion.recipes import flux_12b, flux_535m
from nemo.collections.llm.api import pretrain
import pytest

class TestFluxRecipe:
    @pytest.mark.unit
    def test_flux_12b(self):
        recipe = flux_12b.pretrain_recipe(name="flux_12b_pretrain", num_nodes=1)


        # Check trainer configuration
        assert recipe.trainer.num_nodes == 1
        assert recipe.trainer.devices == 8


        # Check optimizer settings
        assert recipe.optim.config.lr == 1e-4
        assert recipe.optim.config.bf16 is True


    @pytest.mark.unit

    def test_flux_535m(self):
        recipe = flux_535m.unit_test_recipe(name="flux_535m",num_gpus_per_node=1)


        # Check trainer configuration
        assert recipe.trainer.num_nodes == 1
        assert recipe.trainer.devices == 1


        # Check optimizer settings
        assert recipe.optim.config.lr == 1e-4
        assert recipe.optim.config.bf16 is True

