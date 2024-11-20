from pytorch_lightning import Trainer
from nemo.collections.nlp.models.language_modeling.megatron_gpt_model import MegatronGPTModel
from nemo.collections.nlp.parts.nlp_overrides import NLPDDPStrategy

input_nemo_file = "/home/models/nemotron5_8b.nemo"
dummy_trainer = Trainer(devices=1, accelerator="cpu", strategy=NLPDDPStrategy())

model_config = MegatronGPTModel.restore_from(input_nemo_file, trainer=dummy_trainer, return_config=True)
model_config.tensor_model_parallel_size = 1
model_config.pipeline_model_parallel_size = 1
model_config.sequence_parallel = False
model_config.transformer_engine = True
model_config.name = "te_gpt"

map_location = None

model = MegatronGPTModel.restore_from(
        input_nemo_file, trainer=dummy_trainer, override_config_path=model_config, map_location=map_location
)

print(model)
print(model.state_dict().keys())
print(dir(model))
