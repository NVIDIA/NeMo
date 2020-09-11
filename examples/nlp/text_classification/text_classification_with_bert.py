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
from omegaconf import OmegaConf, DictConfig

from nemo.collections.nlp.models.text_classification import TextClassificationModel
from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.exp_manager import exp_manager


@hydra_runner(config_path="conf", config_name="text_classification_config")
def main(cfg: DictConfig) -> None:
    logging.info(f'Config Params:\n {cfg.pretty()}')
    trainer = pl.Trainer(**cfg.trainer)
    exp_manager(trainer, cfg.get("exp_manager", None))

    text_classification_model = TextClassificationModel(cfg.model, trainer=trainer)
    logging.info('Starting training...')
    trainer.fit(text_classification_model)
    logging.info('Training finished!')

    if cfg.model.nemo_path:
        text_classification_model.save_to(cfg.model.nemo_path)
        logging.info('Model is saved into `.nemo` file: cfg.model.nemo_path')

    """
    After model training is done, you can use the model for inference.
    You can either evaluate data from a text file that follows the training data format,
    or provide a list of queries you want to add entities to
    """

    # create a config object for evaluation
    # we would evaluate on the validation data specified in the config file, you may update the file_path to evaluate on other files
    if cfg.model.validation_ds.file_path:
        eval_config = OmegaConf.create({'file_path': './data/SST-2/dev.tsv', 'batch_size': 64, 'shuffle': False, 'num_samples': -1})
        # setup the validation dataloader again
        text_classification_model.setup_validation_data(eval_config)
        # run evaluation on the test data
        logging.info('Evaluating the model on the validation data...')
        trainer.test(text_classification_model)
    else:
        logging.info("No file_path was set for validation_ds, so final evaluation is skipped!")

    # run inference on a few examples
    queries = [
        'we bought four shirts from the nvidia gear store in santa clara.',
        'Nvidia is a company.',
        'The Adventures of Tom Sawyer by Mark Twain is an 1876 novel about a young boy growing up along the Mississippi River.',
    ]
    results = text_classification_model.infer(queries=queries, batch_size=64)

    logging.info('The prediction results of some sample queries with the trained model:')
    for query, result in zip(queries, results):
        logging.info(f'Query : {query}')
        logging.info(f'Predicted label: {result}')


if __name__ == '__main__':
    main()
