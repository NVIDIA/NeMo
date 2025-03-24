from nemo import lightning as nl
from nemo.collections import llm
from nemo.collections.diffusion.models.flux.model import FluxModelParams, MegatronFluxModel

params = FluxModelParams()
model = MegatronFluxModel(flux_params=params)

llm.import_ckpt(model, "hf://black-forest-labs/FLUX.1-dev", "./test_ckpt")
