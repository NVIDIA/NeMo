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

import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf

from nemo.collections.nlp.models import PunctuationCapitalizationModel
from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.exp_manager import exp_manager


"""
To run this script and train the model from scratch, use:
    python punctuation_and_capitalization.py \
    model.dataset.data_dir=PATH_TO_DATA_DIR

To use one of the pretrained versions of the model, run:
    python punctuation_and_capitalization.py \
    pretrained_model=Punctuation_Capitalization_with_BERT

To use one of the pretrained versions of the model and finetune it, run:
    python punctuation_and_capitalization.py \
    pretrained_model=Punctuation_Capitalization_with_BERT \
    model.dataset.data_dir=PATH_TO_DATA_DIR

More details on the task and data format could be found in tutorials/nlp/Punctuation_and_Capitalization.ipynb
"""


@hydra_runner(config_path="conf", config_name="punctuation_capitalization_config")
def main(cfg: DictConfig) -> None:
    trainer = pl.Trainer(**cfg.trainer)
    exp_manager(trainer, cfg.get("exp_manager", None))
    do_training = True
    if not cfg.pretrained_model:
        logging.info(f'Config: {OmegaConf.to_yaml(cfg)}')
        model = PunctuationCapitalizationModel(cfg.model, trainer=trainer)
    else:
        logging.info(f'Loading pretrained model {cfg.pretrained_model}')
        # TODO: Remove strict, when lightning has persistent parameter support for add_state()
        model = PunctuationCapitalizationModel.from_pretrained(cfg.pretrained_model, strict=False)
        data_dir = cfg.model.dataset.get('data_dir', None)
        if data_dir:
            # we can also do finetunining of the pretrained model but it will require
            # setting up train and validation Pytorch DataLoaders
            model.setup_training_data(data_dir=data_dir)
            # evaluation could be done on multiple files, use model.validation_ds.ds_items to specify multiple
            # data directories if needed
            model.setup_validation_data(data_dirs=data_dir)
            logging.info(f'Using config file of the pretrained model')
        else:
            do_training = False
            logging.info(
                f'Data dir should be specified for training/finetuning. '
                f'Using pretrained {cfg.pretrained_model} model weights and skipping finetuning.'
            )

    if do_training:
        trainer.fit(model)
        if cfg.model.nemo_path:
            model.save_to(cfg.model.nemo_path)

    logging.info(
        'During evaluation/testing, it is currently advisable to construct a new Trainer with single GPU '
        'and no DDP to obtain accurate results'
    )
    gpu = 1 if cfg.trainer.gpus != 0 else 0
    trainer = pl.Trainer(gpus=gpu)
    model.set_trainer(trainer)

    # run an inference on a few examples
    queries = [
        'we bought four shirts one pen and a mug from the nvidia gear store in santa clara',
        'what can i do for you today',
        'how are you',
    ]
    inference_results = model.add_punctuation_capitalization(queries)

    for query, result in zip(queries, inference_results):
        logging.info(f'Query : {query}')
        logging.info(f'Result: {result.strip()}\n')


if __name__ == '__main__':
    main()
