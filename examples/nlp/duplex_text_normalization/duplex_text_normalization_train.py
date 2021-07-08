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

"""
This script contains an example on how to train a DuplexTextNormalizationModel.
Note that DuplexTextNormalizationModel is essentially a wrapper class around
two other classes:

(1) DuplexTaggerModel is a model for identifying spans in the input that need to
be normalized. Usually, such spans belong to semiotic classes (e.g., DATE, NUMBERS, ...).

(2) DuplexDecoderModel is a model for normalizing the spans identified by the tagger.
For example, in the text normalization (TN) problem, each span will be converted to its
spoken form. In the inverse text normalization (ITN) problem, each span will be converted
to its written form.

Therefore, this script consists of two parts, one is for training the tagger model
and the other is for training the decoder.

This script uses the `/examples/nlp/duplex_text_normalization/conf/duplex_tn_config.yaml`
config file by default. The other option is to set another config file via command
line arguments by `--config-name=CONFIG_FILE_PATH'. Probably it is worth looking
at the example config file to see the list of parameters used for training.

USAGE Example:
1. Obtain a processed dataset (refer to the `text_normalization doc <https://github.com/NVIDIA/NeMo/blob/main/docs/source/nlp/text_normalization.rst>`)
2.
# python duplex_text_normalization_train.py
        data.base_dir=PATH_TO_DATASET_DIR
        mode={tn,itn,joint}

There are 3 different modes. `tn` mode is for training a system for TN only.
`itn` mode is for training a system for ITN. `joint` is for training a system
that can do both TN and ITN at the same time. Note that the above command will
first train a tagger and then train a decoder sequentially.

You can also train only a tagger (without training a decoder) by running the
following command:
# python duplex_text_normalization_train.py
        data.base_dir=PATH_TO_DATASET_DIR
        mode={tn,itn,joint}
        decoder_model.do_training=false

Or you can also train only a decoder (without training a tagger):
# python duplex_text_normalization_train.py
        data.base_dir=PATH_TO_DATASET_DIR
        mode={tn,itn,joint}
        tagger_model.do_training=false

Information on the arguments:

Most arguments in the example config file are quite self-explanatory (e.g.,
`decoder_model.optim.lr` refers to the learning rate for training the decoder).
Some arguments we want to mention are:

+ data.base_dir: The path to the dataset directory. It is expected that the
directory contains three files: train.tsv, dev.tsv, and test.tsv.

+ tagger_model.nemo_path: This is the path where the final trained tagger model
will be saved to.

+ decoder_model.nemo_path: This is the path where the final trained decoder model
will be saved to.
"""


from helpers import DECODER_MODEL, TAGGER_MODEL, instantiate_model_and_trainer
from omegaconf import DictConfig, OmegaConf

from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.exp_manager import exp_manager


@hydra_runner(config_path="conf", config_name="duplex_tn_config")
def main(cfg: DictConfig) -> None:
    logging.info(f'Config Params: {OmegaConf.to_yaml(cfg)}')

    # Train the tagger
    if cfg.tagger_model.do_training:
        logging.info(
            "================================================================================================"
        )
        logging.info('Starting training tagger...')
        tagger_trainer, tagger_model = instantiate_model_and_trainer(cfg, TAGGER_MODEL, True)
        exp_manager(tagger_trainer, cfg.get('tagger_exp_manager', None))
        tagger_trainer.fit(tagger_model)
        if cfg.tagger_model.nemo_path:
            tagger_model.to(tagger_trainer.accelerator.root_device)
            tagger_model.save_to(cfg.tagger_model.nemo_path)
        logging.info('Training finished!')

    # Train the decoder
    if cfg.decoder_model.do_training:
        logging.info(
            "================================================================================================"
        )
        logging.info('Starting training decoder...')
        decoder_trainer, decoder_model = instantiate_model_and_trainer(cfg, DECODER_MODEL, True)
        exp_manager(decoder_trainer, cfg.get('decoder_exp_manager', None))
        decoder_trainer.fit(decoder_model)
        if cfg.decoder_model.nemo_path:
            decoder_model.to(decoder_trainer.accelerator.root_device)
            decoder_model.save_to(cfg.decoder_model.nemo_path)
        logging.info('Training finished!')


if __name__ == '__main__':
    main()
