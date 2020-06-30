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
# python examples/asr/spech2text.py --asr_model=examples/asr/bad_asr_config.yaml --train_data=/Users/okuchaiev/Data/an4_dataset/an4_train.json --eval_dataset=/Users/okuchaiev/Data/an4_dataset/an4_val.json
# python spech2text.py --asr_model=./bad_asr_config.yaml --train_data=./an4/train_manifest.json --eval_dataset=./an4/test_manifest.json

from argparse import ArgumentParser

import pytorch_lightning as pl
from ruamel.yaml import YAML

from nemo.collections.asr.models import EncDecCTCModel


def main():
    parser = ArgumentParser()
    parser.add_argument("--asr_model", type=str, required=True, default="bad_quartznet15x5.yaml", help="")
    parser.add_argument("--train_dataset", type=str, required=True, default=None, help="training dataset path")
    parser.add_argument("--eval_dataset", type=str, required=True, help="evaluation dataset path")
    parser.add_argument("--num_epochs", default=5, type=int, help="number of epochs to train")

    args = parser.parse_args()

    yaml = YAML(typ="safe")
    with open(args.asr_model) as f:
        model_config = yaml.load(f)

    asr_model = EncDecCTCModel(
        preprocessor_params=model_config['preprocessor_params'],
        encoder_params=model_config['encoder_params'],
        decoder_params=model_config['decoder_params'],
        spec_augment_params=model_config.get('spec_augment_params', None),
    )

    # Setup where your training data is
    model_config['AudioToTextDataLayer']['manifest_filepath'] = args.train_dataset
    model_config['AudioToTextDataLayer_eval']['manifest_filepath'] = args.eval_dataset
    asr_model.setup_training_data(model_config['AudioToTextDataLayer'])
    asr_model.setup_validation_data(model_config['AudioToTextDataLayer_eval'])
    asr_model.setup_optimization(optim_params={'lr': 0.0003})
    trainer = pl.Trainer(
        val_check_interval=35, amp_level='O1', precision=16, gpus=4, max_epochs=123, distributed_backend='ddp'
    )
    # trainer = pl.Trainer(val_check_interval=5, max_epochs=args.num_epochs)
    trainer.fit(asr_model)


if __name__ == '__main__':
    main()
