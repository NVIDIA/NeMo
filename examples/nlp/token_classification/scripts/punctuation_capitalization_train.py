# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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

import os

import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf

from nemo.collections.nlp.models import PunctuationCapitalizationModel
from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.exp_manager import exp_manager


"""
More details on the task and data format could be found in tutorials/nlp/Punctuation_and_Capitalization.ipynb

To run this script and train the model from scratch, use:
    python punctuation_and_capitalization_train.py \
           model.dataset.data_dir=<PATH_TO_DATA_DIR>

To use one of the pretrained versions of the model and finetune it, run:
    python punctuation_and_capitalization.py \
    pretrained_model=Punctuation_Capitalization_with_BERT_base_uncased \
    model.dataset.data_dir=<PATH_TO_DATA_DIR>
    
    <PATH_TO_DATA_DIR> - a directory that contains test_ds.text_file and test_ds.labels_file (see the config)
    pretrained_model   - pretrained PunctuationCapitalization model from list_available_models() or 
                     path to a .nemo file, for example: Punctuation_Capitalization_with_BERT_base_uncased or model.nemo

"""


@hydra_runner(config_path="../conf", config_name="punctuation_capitalization_config")
def main(cfg: DictConfig) -> None:
    trainer = pl.Trainer(**cfg.trainer)
    exp_manager(trainer, cfg.get("exp_manager", None))

    if not cfg.pretrained_model:
        logging.info(f'Config: {OmegaConf.to_yaml(cfg)}')
        model = PunctuationCapitalizationModel(cfg.model, trainer=trainer)
    else:
        if os.path.exists(cfg.pretrained_model):
            model = PunctuationCapitalizationModel.restore_from(cfg.pretrained_model)
        elif cfg.pretrained_model in PunctuationCapitalizationModel.get_available_model_names():
            model = PunctuationCapitalizationModel.from_pretrained(cfg.pretrained_model)
        else:
            raise ValueError(
                f'Provide path to the pre-trained .nemo file or choose from {PunctuationCapitalizationModel.list_available_models()}'
            )

        data_dir = cfg.model.dataset.get('data_dir', None)
        if data_dir:
            if not os.path.exists(data_dir):
                raise ValueError(f'{data_dir} is not found at')

            # we can also do finetuning of the pretrained model but we would need to update the data dir
            model.update_data_dir(data_dir)
            # setup train and validation Pytorch DataLoaders
            model.setup_training_data()
            model.setup_validation_data()
            logging.info(f'Using config file of the pretrained model')
        else:
            raise ValueError(
                'Specify a valid dataset directory that contains test_ds.text_file and test_ds.labels_file \
                with "model.dataset.data_dir" argument'
            )

    trainer.fit(model)

    if cfg.model.nemo_path:
        model.save_to(cfg.model.nemo_path)
        logging.info(f'The model was saved to {cfg.model.nemo_path}')


if __name__ == '__main__':
    main()
