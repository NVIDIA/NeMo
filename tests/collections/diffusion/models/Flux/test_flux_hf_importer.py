from nemo import lightning as nl
from nemo.collections.diffusion.models.flux.model import MegatronFluxModel, FluxModelParams
from nemo.collections import llm

params = FluxModelParams()
model = MegatronFluxModel(flux_params=params)

llm.import_ckpt(model, "hf://black-forest-labs/FLUX.1-dev", "./test_ckpt")