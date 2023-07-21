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
This script contains an example on how to train, evaluate and perform inference with the TextClassificationModel.
TextClassificationModel in NeMo supports text classification problems such as sentiment analysis or
domain/intent detection for dialogue systems, as long as the data follows the format specified below.

***Data format***
TextClassificationModel requires the data to be stored in TAB separated files (.tsv) with two columns of sentence and
label. Each line of the data file contains text sequences, where words are separated with spaces and label separated
with [TAB], i.e.:

[WORD][SPACE][WORD][SPACE][WORD][TAB][LABEL]

For example:

hide new secretions from the parental units[TAB]0
that loves its characters and communicates something rather beautiful about human nature[TAB]1
...

If your dataset is stored in another format, you need to convert it to this format to use the TextClassificationModel.


***Setting the configs***
The model and the PT trainer are defined in a config file which declares multiple important sections.
The most important ones are:
    model: All arguments that are related to the Model - language model, tokenizer, head classifier, optimizer,
            schedulers, and datasets/data loaders.
    trainer: Any argument to be passed to PyTorch Lightning including number of epochs, number of GPUs,
            precision level, etc.

This script uses the `/examples/nlp/text_classification/conf/text_classification_config.yaml` default config file
by default. You may update the config file from the file directly or by using the command line arguments.
Other option is to set another config file via command line arguments by `--config-name=CONFIG_FILE_PATH'.

You first need to set the num_classes in the config file which specifies the number of classes in the dataset.
Notice that some config lines, including `model.dataset.classes_num`, have `???` as their value, this means that values
for these fields are required to be specified by the user. We need to specify and set the `model.train_ds.file_name`,
`model.validation_ds.file_name`, and `model.test_ds.file_name` in the config file to the paths of the train, validation,
 and test files if they exist. We may do it by updating the config file or by setting them from the command line.


***How to run the script?***
For example the following would train a model for 50 epochs in 2 GPUs on a classification task with 2 classes:

# python text_classification_with_bert.py
        model.dataset.num_classes=2
        model.train_ds=PATH_TO_TRAIN_FILE
        model.validation_ds=PATH_TO_VAL_FILE
        trainer.max_epochs=50
        trainer.devices=2

This script would also reload the last checkpoint after the training is done and does evaluation on the dev set,
then performs inference on some sample queries.

By default, this script uses examples/nlp/text_classification/conf/text_classifciation_config.py config file, and
you may update all the params in the config file from the command line. You may also use another config file like this:

# python text_classification_with_bert.py --config-name==PATH_TO_CONFIG_FILE
        model.dataset.num_classes=2
        model.train_ds=PATH_TO_TRAIN_FILE
        model.validation_ds=PATH_TO_VAL_FILE
        trainer.max_epochs=50
        trainer.devices=2

***Load a saved model***
This script would save the model after training into '.nemo' checkpoint file specified by nemo_path of the model config.
You may restore the saved model like this:
    model = TextClassificationModel.restore_from(restore_path=NEMO_FILE_PATH)

***Evaluation a saved model on another dataset***
# If you wanted to evaluate the saved model on another dataset, you may restore the model and create a new data loader:
    eval_model = TextClassificationModel.restore_from(restore_path=checkpoint_path)

# Then, you may create a dataloader config for evaluation:
    eval_config = OmegaConf.create(
        {'file_path': cfg.model.test_ds.file_path, 'batch_size': 64, 'shuffle': False, 'num_workers': 3}
    )
    eval_model.setup_test_data(test_data_config=eval_config)

# You need to create a new trainer:
    eval_trainer = pl.Trainer(devices=1)
    eval_model.set_trainer(eval_trainer)
    eval_trainer.test(model=eval_model, verbose=False)
"""
import json
import multiprocessing as mp
import os
import time
from typing import Dict, List, Optional

import joblib
import pytorch_lightning as pl
from googleapiclient import discovery
from googleapiclient.errors import HttpError
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
from torch import softmax
from functools import partial

from nemo.collections.nlp.models.text_classification import TextClassificationModel
from nemo.collections.nlp.parts.nlp_overrides import NLPDDPStrategy
from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.exp_manager import exp_manager


class PerspectiveApiScorer:
    """
    This class provides a method for accessing Perspective API using the Google API Client to obtain attribute scores for generated texts.
    """

    DEFAULT_ATTRIBUTES = ['toxicity', 'severe_toxicity', 'sexually_explicit', 'threat', 'profanity', 'identity_attack']

    API_KEY = os.getenv("GOOGLE_API_KEY")

    def __init__(self):
        """
        :param api_key: the API key to use. For details, see https://support.perspectiveapi.com/s/docs-get-started
        """
        self._service = discovery.build(
            "commentanalyzer",
            "v1alpha1",
            developerKey=PerspectiveApiScorer.API_KEY,
            discoveryServiceUrl="https://commentanalyzer.googleapis.com/$discovery/rest?version=v1alpha1",
            static_discovery=False,
        )

    def get_scores(self, input_text: str, requested_attributes: Optional[List[str]] = None) -> Dict[str, float]:
        """
        Get attribute scores for a given text via Perspective API.
        :param input_text: the input text
        :param requested_attributes: the attributes for which to compute scores
        :return: a mapping from attribute names to scores
        """
        requested_attributes = (
            requested_attributes if requested_attributes else PerspectiveApiScorer.DEFAULT_ATTRIBUTES
        )

        analyze_request = {
            'comment': {'text': input_text},
            'requestedAttributes': {attribute.upper(): {} for attribute in requested_attributes},
            'spanAnnotations': False,
            'languages': ['en'],
            'doNotStore': True,
        }

        response = None
        while not response:
            try:
                response = self._service.comments().analyze(body=analyze_request).execute()
            except HttpError as e:
                print(f'Perspective API threw an error: {e}\n Retrying in 1 second...')
                #print(input_text)
                time.sleep(1)
        return {
            attribute: response['attributeScores'][attribute.upper()]['summaryScore']['value']
            for attribute in requested_attributes
        }


scorer = PerspectiveApiScorer()


def get_perspective_score(text, aspect='toxicity'):
    score = scorer.get_scores(text)#[aspect]
    return score


def save_predictions(
    filename, generated_field, ground_truth_field, inputs,
):
    """
    Save predictions as a jsonl file

    Args:
        Each arg is a list of strings (all args have the same length)
    """
    docs = []
    for i in range(len(inputs)):
        docs.append(
            {"input": inputs[i], "ground_truth": ground_truth_field[i], "generated": generated_field[i],}
        )
    with open(filename, 'w', encoding="UTF-8") as f:
        for item in docs:
            f.write(json.dumps(item) + "\n")


@hydra_runner(config_path="conf", config_name="text_classification_config")
def main(cfg: DictConfig) -> None:
    logging.info(f'\nConfig Params:\n{OmegaConf.to_yaml(cfg)}')
    try:
        strategy = NLPDDPStrategy()
    except (ImportError, ModuleNotFoundError):
        strategy = None

    trainer = pl.Trainer(strategy=strategy, **cfg.trainer)
    exp_manager(trainer, cfg.get("exp_manager", None))

    if cfg.model.nemo_path and os.path.exists(cfg.model.nemo_path):
        model = TextClassificationModel.restore_from(cfg.model.nemo_path, override_config_path=cfg.model)
        model.set_trainer(trainer)
    else:
        pass
        # raise ValueError("No valid .nemo path specified")
        # model = TextClassificationModel(cfg.model, trainer=trainer)

    
    # logging.info("===========================================================================================")
    # logging.info('Starting training...')
    # trainer.fit(model)
    # logging.info('Training finished!')
    # logging.info("===========================================================================================")

    # if cfg.model.nemo_path:
    #     # '.nemo' file contains the last checkpoint and the params to initialize the model
    #     model.save_to(cfg.model.nemo_path)
    #     logging.info(f'Model is saved into `.nemo` file: {cfg.model.nemo_path}')

    # We evaluate the trained model on the test set if test_ds is set in the config file
    # if cfg.model.test_ds.file_path:
    #     logging.info("===========================================================================================")
    #     logging.info("Starting the testing of the trained model on test set...")
    #     trainer.test(model=model, ckpt_path=None, verbose=False)
    #     logging.info("Testing finished!")
    #     logging.info("===========================================================================================")

    if cfg.model.test_ds.file_path:
        with open(cfg.model.test_ds.file_path, "r") as f:
            lines = f.readlines()
        sentences = []
        gt_labels = []
        for line in lines:
            if cfg.model.test_ds.file_path.endswith(".tsv"):
                elements = line.split("\t")
                if len(elements) == 2:
                    sentence, gt_label = elements
                else:
                    sentence = line
                    gt_label = 0
                if len(sentence) < 1:
                    sentence = " "
            elif cfg.model.test_ds.file_path.endswith(".jsonl"):
                sample = json.loads(line)
                sentence = sample['text'][:15000] # cutting to first 15,000 chars because perspective has a 20KB cutoff
                gt_label = 0 # assume there's no label in text
            sentences.append(sentence)
            gt_labels.append(gt_label)
            # if len(sentences) == 100:
            #     break
        # perform inference on a list of queries.
        # if "infer_samples" in cfg.model and cfg.model.infer_samples:
        logging.info("===========================================================================================")
        logging.info("Starting the inference on some sample queries...")

        if 'prediction_filename' in cfg.model.test_ds:

            if 'perspective' in cfg.model.test_ds.prediction_filename:
                pool = mp.Pool(25)
                results = []

                # if 'severe_toxicity' in cfg.model.test_ds.prediction_filename:
                #     aspect = 'severe_toxicity'
                # elif 'sexually_explicit' in cfg.model.test_ds.prediction_filename:
                #     aspect = 'sexually_explicit'
                # elif 'threat' in cfg.model.test_ds.prediction_filename:
                #     aspect = 'threat'
                # elif 'profanity' in cfg.model.test_ds.prediction_filename:
                #     aspect = 'profanity'
                # elif 'identity_attack' in cfg.model.test_ds.prediction_filename:
                #     aspect = 'identity_attack'
                # else:
                #     aspect = 'toxicity'

                #get_perspective_score_aspect = partial(get_perspective_score, aspect=aspect)
                delayed_scores = pool.imap(get_perspective_score, sentences, 25)
                for score in tqdm(delayed_scores, total=len(sentences)):
                    results.append(score)
                #print("Average: ", sum(results)/len(results))
            else:
                #print("reached here")
                # max_seq_length=512 is the maximum length BERT supports.
                # set -1 for no truncation
                results = model.classifytext(queries=sentences, batch_size=16, max_seq_length=-1)

            save_predictions(cfg.model.test_ds.prediction_filename, results, gt_labels, sentences)
        # raise ValueError
        # logging.info('The prediction results of some sample queries with the trained model:')
        # for query, result in zip(sentences[:5], results):
        #     logging.info(f'Query : {query}')
        #     logging.info(f'Predicted label: {result}')


if __name__ == '__main__':
    main()
