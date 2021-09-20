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
This script contains an example on how to evaluate a DecoderModel.

USAGE Example:
1. Obtain a processed test data file (refer to the
    `text_normalization doc <https://github.com/NVIDIA/NeMo/blob/main/docs/source/nlp/text_normalization.rst>`)
2. python duplex_text_normalization_test.py
        decoder_pretrained_model=PATH_TO_TRAINED_DECODER
        data.test_ds.data_path=PATH_TO_TEST_FILE
        mode={tn,itn,joint}
"""


import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf

from nemo.collections.nlp.models import DuplexDecoderModel
from nemo.core.config import hydra_runner
from nemo.utils import logging


@hydra_runner(config_path="conf", config_name="duplex_tn_config")
def main(cfg: DictConfig) -> None:
    logging.info(f'Config Params: {OmegaConf.to_yaml(cfg)}')
    trainer = pl.Trainer(
        gpus=[0],
        precision=cfg.decoder_trainer.precision,
        amp_level=cfg.decoder_trainer.amp_level,
        logger=False,
        checkpoint_callback=False,
    )
    decoder_model = DuplexDecoderModel.restore_from(cfg.decoder_pretrained_model)
    # Setup test_dataset
    decoder_model.setup_multiple_test_data(cfg.data.test_ds)
    trainer.test(decoder_model)


if __name__ == '__main__':
    main()
