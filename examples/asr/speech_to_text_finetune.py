# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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
This script can used to fine-tune a speech-to-text model of any instance type when users want to 
fine-tune an existing model without changing its core architecture but may change the tokenizer. 
One can mention the pretrained model in two ways: 
1) `init_from_nemo_model` or 
2) `init_from_pretrained_model` in the configuration.

To update the model architecture in conjunction with other modifications, it is advisable to use the primary 'speech_to_text_rnnt/ctc_*.py' script.

Note: To create a single script for all model types, we currently only support two types of 
initializations:
1) `init_from_nemo_model`, and
2) `init_from_pretrained_model`,
but not `init_from_ptl_ckpt`.

To train with prior base model tokenizer keep `model.tokenizer.update_tokenizer` as false else
make it true and provide tokenizer dir along with tokenizer type.

To fine-tune the model, use the following commands:

For initialization from a NEMO model:
```sh
python <NEMO_ROOT>/examples/asr/speech_to_text_finetune.py \
    init_from_nemo_model=<path_to_nemo_model>
```

For initialization from a pretrained model:
```sh
python <NEMO_ROOT>/examples/asr/speech_to_text_finetune.py \
    init_from_pretrained_model=<pretrained_model_name>
```

# Fine-Tune a Model

For documentation on fine-tuning this model, please visit:
https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/asr/configs.html#fine-tuning-configurations
"""

import copy

import pytorch_lightning as pl
from omegaconf import OmegaConf
from pytorch_lightning.utilities import rank_zero_only

from nemo.collections.asr.models import ASRModel
from nemo.core.config import hydra_runner
from nemo.utils import logging, model_utils
from nemo.utils.exp_manager import exp_manager


@hydra_runner(config_path="conf", config_name="speech_to_text_finetune")
def main(cfg):
    logging.info(f'Hydra config: {OmegaConf.to_yaml(cfg)}')

    trainer = pl.Trainer(**cfg.trainer)
    exp_manager(trainer, cfg.get("exp_manager", None))

    if hasattr(cfg, 'init_from_ptl_ckpt') and cfg.init_from_ptl_ckpt is not None:
        raise NotImplementedError(
            "Currently for simplicity of single script for all model types, we only support `init_from_nemo_model` and `init_from_pretrained_model`"
        )

    @rank_zero_only
    def get_base_model(cfg):
        asr_model = None
        nemo_model_path = cfg.get('init_from_nemo_model', None)
        pretrained_name = cfg.get('init_from_pretrained_model', None)
        if nemo_model_path is not None and pretrained_name is not None:
            raise ValueError("Only pass `init_from_nemo_model` or `init_from_pretrained_model` but not both")
        elif nemo_model_path is None and pretrained_name is None:
            raise ValueError(
                "Both `init_from_nemo_model` and `init_from_pretrained_model cannot be None, should pass atleast one of them"
            )
        elif nemo_model_path is not None:
            asr_model = ASRModel.restore_from(restore_path=nemo_model_path)
        elif pretrained_name is not None:
            asr_model = ASRModel.from_pretrained(model_name=pretrained_name)

        return asr_model

    asr_model = get_base_model(cfg)
    vocab_size = asr_model.tokenizer.vocab_size

    # if new tokenizer is provided, use it
    if hasattr(cfg.model.tokenizer, 'update_tokenizer') and cfg.model.tokenizer.update_tokenizer:
        decoder = copy.deepcopy(asr_model.decoder)
        joint_state = copy.deepcopy(asr_model.joint)

        if cfg.model.tokenizer.dir is None:
            raise ValueError("dir must be specified if update_tokenizer is True")
        logging.info("Using the tokenizer provided through config")
        asr_model.change_vocabulary(
            new_tokenizer_dir=cfg.model.tokenizer.dir, new_tokenizer_type=cfg.model.tokenizer.type
        )
        if asr_model.tokenizer.vocab_size != vocab_size:
            logging.warning(
                "The vocabulary size of the new tokenizer differs from that of the loaded model. As a result, finetuning will proceed with the new vocabulary, and the decoder will be reinitialized."
            )
        else:
            asr_model.decoder = decoder
            asr_model.joint = joint_state
    else:
        logging.info("Reusing the tokenizer from the loaded model.")

    # Setup Data
    cfg = model_utils.convert_model_config_to_dict_config(cfg)
    asr_model.setup_training_data(cfg.model.train_ds)
    asr_model.setup_validation_data(cfg.model.validation_ds)
    if hasattr(cfg.model, 'test_ds') and cfg.model.test_ds.manifest_filepath is not None:
        asr_model.setup_test_data(cfg.model.test_ds)

    # Setup Optimizer
    asr_model.setup_optimization(cfg.model.optim)

    # Setup SpecAug
    if hasattr(cfg.model, 'spec_augment') and cfg.model.spec_augment is not None:
        asr_model.spec_augment = ASRModel.from_config_dict(cfg.model.spec_augment)

    trainer.fit(asr_model)


if __name__ == '__main__':
    main()  # noqa pylint: disable=no-value-for-parameter
