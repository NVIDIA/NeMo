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
import time
import numpy as np
import pytorch_lightning as pl

from tqdm import tqdm
from math import ceil
from omegaconf import DictConfig, OmegaConf
from utils import TAGGER_MODEL, DECODER_MODEL, initialize_model_and_trainer

from nemo.utils import logging
from nemo.core.config import hydra_runner
from nemo.collections.nlp.models import NeuralTextNormalizationModel
from nemo.collections.nlp.data.text_normalization import TextNormalizationTestDataset

@hydra_runner(config_path="conf", config_name="text_normalization_config")
def main(cfg: DictConfig) -> None:
    logging.info(f'Config Params: {OmegaConf.to_yaml(cfg)}')
    tagger_trainer, tagger_model = initialize_model_and_trainer(cfg, TAGGER_MODEL, False)
    decoder_trainer, decoder_model = initialize_model_and_trainer(cfg, DECODER_MODEL, False)
    tn_model = NeuralTextNormalizationModel(tagger_model, decoder_model)

    # Setup test_dataset
    test = TextNormalizationTestDataset(cfg.data.test_ds.data_path)

    # Apply the model on the test dataset
    all_inputs, all_preds, all_targets, all_run_times = [], [], [], []
    batch_size = cfg.data.test_ds.batch_size
    nb_iters = int(ceil(len(test) / batch_size))
    for i in tqdm(range(nb_iters)):
        start_idx = i * batch_size
        end_idx = (i+1) * batch_size
        batch_insts = test[start_idx:end_idx]
        batch_inputs, batch_targets = zip(*batch_insts)
        # Inference and Running Time Measurement
        batch_start_time = time.time()
        batch_preds = tn_model._infer(batch_inputs)
        batch_run_time = (time.time() - batch_start_time) * 1000  # milliseconds
        all_run_times.append(batch_run_time)
        # Update all_inputs, all_preds and all_targets
        all_inputs.extend(batch_inputs)
        all_preds.extend(batch_preds)
        all_targets.extend(batch_targets)

    # Metrics
    sent_accuracy = TextNormalizationTestDataset.compute_sent_accuracy(all_preds, all_targets)
    logging.info(f'Sentence Accuracy: {sent_accuracy}')
    logging.info(f'Average running time: {np.average(all_run_times)} ms')

if __name__ == '__main__':
    main()
