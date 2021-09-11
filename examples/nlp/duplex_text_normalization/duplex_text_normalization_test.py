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
This script contains an example on how to evaluate a DuplexTextNormalizationModel.
Note that DuplexTextNormalizationModel is essentially a wrapper class around
DuplexTaggerModel and DuplexDecoderModel. Therefore, two trained NeMo models
should be specified before evaluation (one is a trained DuplexTaggerModel
and the other is a trained DuplexDecoderModel).

USAGE Example:
1. Obtain a processed test data file (refer to the `text_normalization doc <https://github.com/NVIDIA/NeMo/blob/main/docs/source/nlp/text_normalization.rst>`)
2.
# python duplex_text_normalization_test.py
        tagger_pretrained_model=PATH_TO_TRAINED_TAGGER
        decoder_pretrained_model=PATH_TO_TRAINED_DECODER
        data.test_ds.data_path=PATH_TO_TEST_FILE
        mode={tn,itn,joint}
        lang={en,ru,de}

The script also supports the `interactive` mode where a user can just make the model
run on any input text:
# python duplex_text_normalization_test.py \
        data.test_ds.data_path=PATH_TO_TEST_FILE \
        tagger_pretrained_model=PATH_TO_TRAINED_TAGGER \
        decoder_pretrained_model=PATH_TO_TRAINED_DECODER \
        mode={tn,itn,joint} \
        lang={en,ru,de} \
        inference.interactive=true

This script uses the `/examples/nlp/duplex_text_normalization/conf/duplex_tn_config.yaml`
config file by default. The other option is to set another config file via command
line arguments by `--config-name=CONFIG_FILE_PATH'.

Note that when evaluating a DuplexTextNormalizationModel on a labeled dataset,
the script will automatically generate a file for logging the errors made
by the model. The location of this file is determined by the argument
`inference.errors_log_fp`.

"""


from helpers import DECODER_MODEL, TAGGER_MODEL, instantiate_model_and_trainer
from omegaconf import DictConfig, OmegaConf

from nemo.collections.nlp.data.text_normalization import TextNormalizationTestDataset, constants
from nemo.collections.nlp.data.text_normalization.utils import basic_tokenize
from nemo.collections.nlp.models import DuplexTextNormalizationModel
from nemo.core.config import hydra_runner
from nemo.utils import logging


@hydra_runner(config_path="conf", config_name="duplex_tn_config")
def main(cfg: DictConfig) -> None:
    logging.info(f'Config Params: {OmegaConf.to_yaml(cfg)}')
    lang = cfg.lang
    tagger_trainer, tagger_model = instantiate_model_and_trainer(cfg, TAGGER_MODEL, False)
    decoder_trainer, decoder_model = instantiate_model_and_trainer(cfg, DECODER_MODEL, False)
    tn_model = DuplexTextNormalizationModel(tagger_model, decoder_model, lang)

    if not cfg.inference.interactive:
        # Setup test_dataset
        test_dataset = TextNormalizationTestDataset(cfg.data.test_ds.data_path, cfg.mode, lang)
        results = tn_model.evaluate(test_dataset, cfg.data.test_ds.batch_size, cfg.inference.errors_log_fp)
        print(f'\nTest results: {results}')
    else:
        while True:
            test_input = input('Input a test input:')
            test_input = ' '.join(basic_tokenize(test_input, lang))
            outputs = tn_model._infer([test_input, test_input], [constants.INST_BACKWARD, constants.INST_FORWARD])[-1]
            print(f'Prediction (ITN): {outputs[0]}')
            print(f'Prediction (TN): {outputs[1]}')

            should_continue = input('\nContinue (y/n): ').strip().lower()
            if should_continue.startswith('n'):
                break


if __name__ == '__main__':
    main()
