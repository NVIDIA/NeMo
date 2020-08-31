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
from omegaconf import DictConfig

from nemo.collections.nlp.models import PunctuationCapitalizationModel
from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.exp_manager import exp_manager


"""
To run this script, use:
python punctuation_and_capitalization.py model.dataset.data_dir=PATH_TO_DATA_DIR

More details on the task and data format could be found in tutorials/nlp/Punctuation_and_Capitalization.ipynb
"""


@hydra_runner(config_path="conf", config_name="punctuation_capitalization_config")
def main(cfg: DictConfig) -> None:
    logging.info(f'Config: {cfg.pretty()}')
    trainer = pl.Trainer(**cfg.trainer)
    exp_manager(trainer, cfg.get("exp_manager", None))
    # model = PunctuationCapitalizationModel(cfg.model, trainer=trainer)
    # trainer.fit(model)
    # if cfg.model.nemo_path:
    #     model.save_to(cfg.model.nemo_path)
    #
    # # run an inference on a few examples
    # queries = [
    #     'we bought four shirts and one mug from the nvidia gear store in santa clara',
    #     'what can i do for you today',
    #     'how are you',
    #     'how is the weather in',
    # ]
    # inference_results = model.add_punctuation_capitalization(queries)
    #
    # for query, result in zip(queries, inference_results):
    #     logging.info(f'Query   : {query}')
    #     logging.info(f'Combined: {result.strip()}\n')

    pretrained_model = PunctuationCapitalizationModel.restore_from(
        '/home/ebakhturina/nemo_ckpts/punctuation/ptl/bert_on_tatoeba_v3/1383000/punct_capit_distil.nemo'
    )
    # then we need to setup the data dir to get class weights statistics
    pretrained_model.update_data_dir('/home/ebakhturina/tatoeba/sample')

    # setup train and validation Pytorch DataLoaders
    pretrained_model.setup_training_data()
    pretrained_model.setup_validation_data()

    # and now we can create a PyTorch Lightning trainer and call `fit` again
    # for this tutorial we are setting fast_dev_run to True, and the trainer will run 1 training batch and 1 validation batch
    # for actual model training, disable the flag
    fast_dev_run = True
    trainer = pl.Trainer(gpus=[1], fast_dev_run=fast_dev_run)
    trainer.fit(pretrained_model)


if __name__ == '__main__':
    main()
