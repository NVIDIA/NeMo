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

from dataclasses import dataclass, field
from os.path import expanduser
from typing import Any, List

import hydra
from omegaconf import MISSING, OmegaConf

from nemo.collections.cv.models import Model
from nemo.collections.vis.datasets import CLEVRConfig
from nemo.collections.vis.transforms import Compose
from nemo.core.config import DataLoaderConfig, hydra_runner
from nemo.utils import logging


@dataclass
class AppConfig:
    """
    This is structured config for this application.
    """

    dataloader: DataLoaderConfig = DataLoaderConfig()
    training_dataset: CLEVRConfig = CLEVRConfig(split="training", stream_images=False)
    validation_dataset: CLEVRConfig = CLEVRConfig(split="validation", stream_images=False)
    question_text_transforms: List[Any] = field(default_factory=list)
    question_word_mappings_filepath: str = MISSING
    answers_word_mappings_filepath: str = MISSING


# Load configuration file from "conf" dir using schema for validation/retrieving the default values.
@hydra_runner(config_path="conf", config_name="clevr_process_dataset", schema=AppConfig)
def main(cfg: AppConfig):
    # Show configuration.
    logging.info("Application settings\n" + OmegaConf.to_yaml(cfg))

    # Instantiate the dataloader/dataset.
    train_dl = Model.instantiate_dataloader(cfg.dataloader, cfg.training_dataset)
    valid_dl = Model.instantiate_dataloader(cfg.dataloader, cfg.validation_dataset)

    # Instantiate text transforms - the same as we will use during training.
    text_transforms = Compose([hydra.utils.instantiate(trans) for trans in cfg.question_text_transforms])

    logging.info("Processing training dataset with {} batches/{} samples".format(len(train_dl), len(train_dl.dataset)))

    q_vocab = set()
    a_vocab = set()

    # 1. Collect input and output vocabularies from training split.
    for batch in iter(train_dl):
        _, _, _, questions, answers, _, _ = batch
        # Process questions.
        proc_questions = text_transforms(questions)
        # Add tokens to vocab.
        for sample in proc_questions:
            for token in sample:
                q_vocab.add(token)

        # User anwsers as they are.
        for answer in answers:
            a_vocab.add(answer)

    logging.info("Question vocabulary collected from the training split ({}):\n{}".format(len(q_vocab), q_vocab))
    logging.info("Answer vocabulary collected from the training split ({}):\n{}".format(len(q_vocab), q_vocab))

    logging.info(
        "Processing validation dataset with {} batches/{} samples".format(len(valid_dl), len(valid_dl.dataset))
    )

    # 2. Collect input and output vocabularies from validation split  - just in case.
    for batch in iter(valid_dl):
        _, _, _, questions, answers, _, _ = batch
        # Process questions.
        proc_questions = text_transforms(questions)
        # Add tokens to vocab.
        for sample in proc_questions:
            for token in sample:
                q_vocab.add(token)

        # User answers as they are.
        for answer in answers:
            a_vocab.add(answer)

    logging.info("Question vocabulary updated using the validation split ({}):\n{}".format(len(q_vocab), q_vocab))
    logging.info("Answer vocabulary updated using the validation split ({}):\n{}".format(len(q_vocab), q_vocab))

    # 3.1 Store question word mappings.
    with open(expanduser(cfg.question_word_mappings_filepath), 'w') as out_file:
        for id, word in enumerate(q_vocab):
            out_file.write("{},{}\n".format(word, id))
    logging.info("Question word mappings stored to `{}`".format(expanduser(cfg.question_word_mappings_filepath)))

    # 3.2 Store answer word mappings.
    with open(expanduser(cfg.answers_word_mappings_filepath), 'w') as out_file:
        for id, word in enumerate(a_vocab):
            out_file.write("{},{}\n".format(word, id))
    logging.info("Answer word mappings stored to `{}`".format(expanduser(cfg.answers_word_mappings_filepath)))


if __name__ == "__main__":
    main()
