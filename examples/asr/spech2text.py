# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# TODO: This is WIP and needs a lot of polishing

import pytorch_lightning as pl
from ruamel.yaml import YAML

from nemo.collections.asr.models import EncDecCTCModel

yaml = YAML(typ="safe")
with open('/Users/okuchaiev/repos/NeMo/examples/asr/bad_asr_config.yaml') as f:
    model_config = yaml.load(f)

asr_model = EncDecCTCModel(
    preprocessor_params=model_config['AudioToMelSpectrogramPreprocessor'],
    encoder_params=model_config['JasperEncoder'],
    decoder_params=model_config['JasperDecoder'],
)

# Setup where your training data is
asr_model.setup_training_data(model_config['AudioToTextDataLayer'])
asr_model.setup_validation_data(model_config['AudioToTextDataLayer_eval'])
asr_model.setup_optimization(optim_params={'lr': 0.0003})
# trainer = pl.Trainer(val_check_interval=5, amp_level='O1', precision=16, gpus=2, max_epochs=50, distributed_backend='ddp')
trainer = pl.Trainer(val_check_interval=5)
trainer.fit(asr_model)
