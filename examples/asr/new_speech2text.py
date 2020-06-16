import pytorch_lightning as pl

# Load model definition
from ruamel.yaml import YAML

from nemo.collections.asr.models.asrconvctcmodel2 import ASRConvCTCModel

yaml = YAML(typ="safe")
with open('/home/okuchaiev/repos/NeMo/examples/asr/configs/jasper_an4-2.yaml') as f:
    model_config = yaml.load(f)

asr_model = ASRConvCTCModel(
    preprocessor_params=model_config['AudioToMelSpectrogramPreprocessor'],
    encoder_params=model_config['JasperEncoder'],
    decoder_params=model_config['JasperDecoder'],
)
# asr_model = ASRConvCTCModel.from_cloud(name="QuartzNet15x5-En")

# Setup where your training data is
asr_model.setup_training_data(model_config['AudioToTextDataLayer'])
asr_model.setup_validation_data(model_config['AudioToTextDataLayer_eval'])

trainer = pl.Trainer(val_check_interval=5, amp_level='O1', precision=16, gpus=2, max_epochs=50, distributed_backend='ddp')
trainer.fit(asr_model)

# Export for Jarvis
asr_model.save_to('qn.nemo', optimize_for_deployment=True)
