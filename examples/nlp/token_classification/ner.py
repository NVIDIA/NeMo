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

import os
from argparse import ArgumentParser

import pytorch_lightning as pl

import nemo.collections.nlp as nemo_nlp


def main():
    parser = ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, default='', help="Path to data folder")
    parser.add_argument("--num_classes", type=int, default=9, help="Number of classes")
    parser.add_argument("--num_epochs", default=5, type=int, help="Number of epochs to train")
    parser.add_argument(
        "--pretrained_model_name",
        default="bert-base-uncased",
        type=str,
        help="Pretrained language model name",
        choices=nemo_nlp.modules.get_pretrained_lm_models_list(),
    )
    parser.add_argument("--work_dir", default='output', type=str, help="Number of epochs to train")
    args = parser.parse_args()

    if not os.path.exists(args.data_dir):
        raise FileNotFoundError(
            "Dataset not found. For NER, CoNLL-2003 dataset can be obtained at"
            "https://github.com/kyzhouhzau/BERT-NER/tree/master/data."
        )

    ner_model = nemo_nlp.models.NERModel(num_classes=args.num_classes)
    ner_model.setup_training_data(args.data_dir, train_data_layer_config={'shuffle': True})
    ner_model.setup_validation_data(data_dir=args.data_dir, val_data_layer_config={'shuffle': False})
    ner_model.setup_optimization(optim_config={'lr': 0.0003})

    trainer = pl.Trainer(fast_dev_run=True)
    # trainer = pl.Trainer(
    #     val_check_interval=35, amp_level='O1', precision=16, gpus=2, max_epochs=123, distributed_backend='ddp', fast_dev_run=True
    # )
    trainer.fit(ner_model)


if __name__ == '__main__':
    main()
