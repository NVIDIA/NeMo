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

import json
import os

from omegaconf import DictConfig, OmegaConf

from nemo.collections.nlp.models import ZeroShotIntentModel
from nemo.core.config import hydra_runner
from nemo.utils import logging


@hydra_runner(config_path="conf", config_name="zero_shot_intent_config")
def main(cfg: DictConfig) -> None:
    logging.info(f'Config Params:\n {OmegaConf.to_yaml(cfg)}')

    # initialize the model using the config file
    if cfg.pretrained_model and os.path.exists(cfg.pretrained_model):
        model = ZeroShotIntentModel.restore_from(cfg.pretrained_model, strict=False)
    else:
        raise ValueError('Provide path to the pre-trained .nemo checkpoint')

    # predicting an intent of a query
    queries = [
        "I'd like a veggie burger and fries",
        "Turn off the lights in the living room",
    ]

    candidate_labels = ['Food order', 'Play music', 'Request for directions', 'Change lighting', 'Calendar query']

    predictions = model.predict(queries, candidate_labels, batch_size=4, multi_label=True)

    logging.info('The prediction results of some sample queries with the trained model:')
    for query in predictions:
        logging.info(json.dumps(query, indent=4))
    logging.info("Inference finished!")


if __name__ == '__main__':
    main()
