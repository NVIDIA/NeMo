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

import os

import pytorch_lightning as pl
from omegaconf import DictConfig

from nemo.collections.nlp.models import TokenClassificationModel
from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.exp_manager import exp_manager


@hydra_runner(config_path="conf", config_name="token_classification_config")
def main(cfg: DictConfig) -> None:
    logging.info(f'Config: {cfg.pretty()}')
    trainer = pl.Trainer(**cfg.trainer)
    exp_dir = exp_manager(trainer, cfg.get("exp_manager", None))
    model = TokenClassificationModel(cfg.model, trainer=trainer)
    trainer.fit(model)
    if cfg.model.nemo_path:
        model.save_to(cfg.model.nemo_path)

    """
    After model training is done, you can use the model for inference.
    You can either evaluate data from a text_file that follows training data format,
    or provide a list of queries you want to add entities to
    """
    # run evaluation on a dataset from file
    model.evaluate_from_file(
        text_file=os.path.join(cfg.model.dataset.data_dir, cfg.model.validation_ds.text_file),
        labels_file=os.path.join(cfg.model.dataset.data_dir, cfg.model.validation_ds.labels_file),
        output_dir=exp_dir,
        add_confusion_matrix=True,
        normalize_confusion_matrix=True,
    )

    # run an inference on a few examples
    queries = [
        'we bought four shirts from the nvidia gear store in santa clara.',
        'Nvidia is a company.',
        'The Adventures of Tom Sawyer by Mark Twain is an 1876 novel about a young boy growing '
        + 'up along the Mississippi River.',
    ]
    results = model.add_enties(queries)

    for query, result in zip(queries, results):
        logging.info('')
        logging.info(f'Query : {query}')
        logging.info(f'Result: {result.strip()}\n')


if __name__ == '__main__':
    main()
