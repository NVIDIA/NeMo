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

# TODO: WIP

from argparse import ArgumentParser

import pytorch_lightning as pl

from nemo.collections.nlp.models import PunctuationCapitalizationModel


def main():
    parser = ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, default='', help="Path to data folder")
    parser.add_argument("--punct_num_classes", type=int, default=4, help="Number of classes")
    parser.add_argument("--num_epochs", default=5, type=int, help="Number of epochs to train")

    args = parser.parse_args()

    model = PunctuationCapitalizationModel(punct_num_classes=args.punct_num_classes)
    model.setup_training_data(args.data_dir, train_data_layer_params={'shuffle': True})
    model.setup_validation_data(data_dir=args.data_dir, val_data_layer_params={'shuffle': False})
    model.setup_optimization(optim_params={'lr': 0.0003})

    # multi GPU
    trainer = pl.Trainer(
        val_check_interval=35,
        amp_level='O1',
        precision=16,
        gpus=2,
        max_epochs=123,
        distributed_backend='ddp',
        fast_dev_run=True,
    )
    # single GPU
    # trainer = pl.Trainer(fast_dev_run=True)
    trainer.fit(model)


if __name__ == '__main__':
    main()
