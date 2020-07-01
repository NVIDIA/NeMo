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
# python speech_to_text.py \
#         --asr_model "bad_quartznet15x5.yaml" \
#         --train_dataset "./an4/train_manifest.json" \
#         --eval_dataset "./an4/test_manifest.json" \
#         --gpus 4 \
#         --distributed_backend "ddp" \
#         --max_epochs 1 \
#         --fast_dev_run \
#         --lr 0.001 \

from argparse import ArgumentParser

import pytorch_lightning as pl
from ruamel.yaml import YAML

from nemo.collections.asr.arguments import add_asr_args
from nemo.collections.asr.models import EncDecCTCModel
from nemo.core.optim.optimizers import add_optimizer_args


def main(args):

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
    asr_model.setup_optimization(optim_params={'optimizer': args.optimizer, 'lr': args.lr, 'opt_args': args.opt_args})
    # trainer = pl.Trainer(
    #     val_check_interval=1, amp_level='O1', precision=16, gpus=4, max_epochs=123, distributed_backend='ddp'
    # )
    # trainer = pl.Trainer(val_check_interval=5, max_epochs=args.num_epochs)
    trainer = pl.Trainer.from_argparse_args(args)
    trainer.fit(asr_model)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser = add_asr_args(parser)
    parser = add_optimizer_args(parser)

    args = parser.parse_args()

    main(args)
