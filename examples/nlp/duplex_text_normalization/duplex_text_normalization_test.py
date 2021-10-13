# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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
This script runs evaluation on the test data. For more details on the data format refer to the
`text_normalization doc <https://github.com/NVIDIA/NeMo/blob/main/docs/source/nlp/text_normalization.rst>`

1. To evaluate the tagger model:
    python duplex_text_normalization_test.py \
        tagger_pretrained_model=PATH_TO_TRAINED_TAGGER \
        mode={tn,itn,joint} \
        lang={en,ru,de}

2. To evaluate the decoder model:
    python duplex_text_normalization_test.py \
        decoder_pretrained_model=PATH_TO_TRAINED_DECODER \
        mode={tn,itn,joint} \
        lang={en,ru,de}

3. To jointly evaluate "tagger -> decoder" pipeline the DuplexTextNormalizationModel will be used.
    DuplexTextNormalizationModel is essentially a wrapper class around DuplexTaggerModel and DuplexDecoderModel.
    Therefore, two trained NeMo models should be specified to run the joint evaluation
    (one is a trained DuplexTaggerModel and the other is a trained DuplexDecoderModel).
    Additionally, an error log will be saved in a file specified with data.test_ds.errors_log_fp (this file can be
    later used with analyze_errors.py)

    python duplex_text_normalization_test.py \
        tagger_pretrained_model=PATH_TO_TRAINED_TAGGER \
        decoder_pretrained_model=PATH_TO_TRAINED_DECODER \
        mode={tn,itn,joint} \
        lang={en,ru,de} \
        data.test_ds.errors_log_fp=PATH_TO_FILE_TO_SAVE_ERROR_LOG \
        data.test_ds.use_cache=true \
        data.test_ds.batch_size=256
"""

from helpers import DECODER_MODEL, TAGGER_MODEL, instantiate_model_and_trainer
from omegaconf import DictConfig

from nemo.collections.nlp.data.text_normalization import TextNormalizationTestDataset
from nemo.collections.nlp.models import DuplexTextNormalizationModel
from nemo.core.config import hydra_runner
from nemo.utils import logging


@hydra_runner(config_path="conf", config_name="duplex_tn_config")
def main(cfg: DictConfig) -> None:
    lang = cfg.lang

    if cfg.tagger_pretrained_model:
        tagger_trainer, tagger_model = instantiate_model_and_trainer(cfg, TAGGER_MODEL, False)
        tagger_model.max_sequence_len = 512
        tagger_model.setup_test_data(cfg.data.test_ds)
        logging.info('Evaluating the tagger...')
        tagger_trainer.test(model=tagger_model, verbose=False)
    else:
        logging.info('Tagger checkpoint is not provided, skipping tagger evaluation')

    if cfg.decoder_pretrained_model:
        decoder_trainer, decoder_model = instantiate_model_and_trainer(cfg, DECODER_MODEL, False)
        decoder_model.max_sequence_len = 512
        decoder_model.setup_multiple_test_data(cfg.data.test_ds)
        logging.info('Evaluating the decoder...')
        decoder_trainer.test(decoder_model)
    else:
        logging.info('Decoder checkpoint is not provided, skipping decoder evaluation')

    if cfg.tagger_pretrained_model and cfg.decoder_pretrained_model:
        logging.info('Running evaluation of the duplex model (tagger + decoder) on the test set.')
        tn_model = DuplexTextNormalizationModel(tagger_model, decoder_model, lang)
        test_dataset = TextNormalizationTestDataset(cfg.data.test_ds.data_path, cfg.mode, lang)
        results = tn_model.evaluate(test_dataset, cfg.data.test_ds.batch_size, cfg.data.test_ds.errors_log_fp)
        print(f'\nTest results: {results}')


if __name__ == '__main__':
    main()
