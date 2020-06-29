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
from argparse import ArgumentParser

import pytorch_lightning as pl
from ruamel.yaml import YAML

from nemo.collections.asr.models import EncDecCTCModel

def main(args):
    
    # Instantiate the model which we'll train
    if args.asr_model.endswith('.yaml'):
        print(f"SpeechToText: Will train from scratch using config from {args.asr_model}")
        yaml = YAML(typ="safe")
        with open(args.asr_model) as f:
            model_config = yaml.load(f)
    else:
        raise NotImplemented("from_pretrained not implemented yet")

    asr_model = EncDecCTCModel(
        preprocessor_params=model_config['AudioToMelSpectrogramPreprocessor'],
        encoder_params=model_config['JasperEncoder'],
        decoder_params=model_config['JasperDecoder'],
    )

    # Setup where your training data is
    asr_model.setup_training_data(model_config['AudioToTextDataLayer'])
    asr_model.setup_validation_data(model_config['AudioToTextDataLayer_eval'])
    asr_model.setup_optimization(optim_params={'lr': 0.0003})
    # trainer = pl.Trainer(
    #    val_check_interval=5, amp_level='O1', precision=16, gpus=2, max_epochs=30, distributed_backend='ddp'
    # )
    trainer = pl.Trainer(val_check_interval=5, max_epochs=100, gpus=4, distributed_backend='ddp')
    trainer.fit(asr_model)

if __name__ == '__main__':
    parent_parser = ArgumentParser(add_help=False) 
    parent_parser.add_argument(
        "--asr_model",
        type=str,
        default="QuartzNet15x5-En",
        required=True,
        help="Pass: 'QuartzNet15x5-En', 'QuartzNet15x5-Zh', or 'JasperNet10x5-En' to train from pre-trained models. To train from scratch pass path to modelfile ending with .yaml.",
    )

    args = parent_parser.parse_args()

    main(args)