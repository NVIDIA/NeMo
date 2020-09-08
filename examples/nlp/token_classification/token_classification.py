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

"""
## Tasks
Token Classificatin script supports Named Entity Recognition task and other token level classification tasks,
as long as the data followes the format specified below.

Token Classification Model requires the data to be splitted into 2 files: text.txt and labels.txt.
Each line of the text.txt file contains text sequences, where words are separated with spaces, i.e.:
[WORD] [SPACE] [WORD] [SPACE] [WORD].
The labels.txt file contains corresponding labels for each word in text.txt, the labels are separated with spaces, i.e.:
[LABEL] [SPACE] [LABEL] [SPACE] [LABEL].

Example of a text.txt file:
Jennifer is from New York City .
She likes ...
...

Corresponding labels.txt file:
B-PER O O B-LOC I-LOC I-LOC O
O O ...
...


## Preparing the dataset
To convert an IOB format data to the format required for training, run
examples/nlp/token_classification/data/import_from_iob_format.py on your train and dev files, as follows:

python examples/nlp/token_classification/data/import_from_iob_format.py --data_file PATH_TO_IOB_FORMAT_DATAFILE


## Model Training

To train TokenClassification model with the default config file, run:

    python token_classification.py \
    model.dataset.data_dir=<PATH_TO_DATA_DIR>  \
    trainer.max_epochs=<NUM_EPOCHS> \
    trainer.gpus="[<CHANGE_TO_GPU_YOU_WANT_TO_USE>]

More details on how to use this script could be found in
tutorials/nlp/Token_Classification_Named_Entity_Recognition.ipynb
"""
import os

import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf

from nemo.collections.nlp.models import TokenClassificationModel
from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.exp_manager import exp_manager


@hydra_runner(config_path="conf", config_name="token_classification_config")
def main(cfg: DictConfig) -> None:
    logging.info(f'Config: {OmegaConf.to_yaml(cfg)}')
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
    results = model.add_predictions(queries)

    for query, result in zip(queries, results):
        logging.info('')
        logging.info(f'Query : {query}')
        logging.info(f'Result: {result.strip()}\n')


if __name__ == '__main__':
    main()
