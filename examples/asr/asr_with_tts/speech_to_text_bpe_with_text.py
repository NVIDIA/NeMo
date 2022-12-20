# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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
# Training the model from scratch
```sh
python speech_to_text_bpe_with_text.py \
    # (Optional: --config-path=<path to dir of configs> --config-name=<name of config without .yaml>) \
    ++model.tts_model_path={tts_model_path} \
    ++model.asr_model_type=<rnnt_bpe or ctc_bpe> \
    model.train_ds.manifest_filepath=<path to manifest with audio-text pairs or null> \
    ++model.train_ds.text_data.manifest_filepath=<path to manifest with train text> \
    ++model.train_ds.text_data.speakers_filepath=<path to speakers list> \
    ++model.train_ds.text_data.min_words=1 \
    ++model.train_ds.text_data.max_words=45 \
    model.validation_ds.manifest_filepath=<path to val/test manifest> \
    exp_manager.create_wandb_logger=True \
    exp_manager.wandb_logger_kwargs.name="<Name of experiment>" \
    exp_manager.wandb_logger_kwargs.project="<Name of project>"
```

# Fine-tune a model

For documentation on fine-tuning this model, please visit -
ToDo

"""


import pytorch_lightning as pl
from omegaconf import OmegaConf

from nemo.collections.asr.models.hybrid_asr_tts_models import ASRWithTTSModel
from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.exp_manager import exp_manager


@hydra_runner(config_path="examples/asr/asr_tts", config_name="hybrid_asr_tts")
def main(cfg):
    logging.info(f'Hydra config: {OmegaConf.to_yaml(cfg)}')
    OmegaConf.resolve(cfg)

    trainer = pl.Trainer(**cfg.trainer)
    exp_manager(trainer, cfg.get("exp_manager", None))

    asr_model = ASRWithTTSModel.from_asr_config(
        asr_cfg=cfg.model, asr_model_type=cfg.asr_model_type, tts_model_path=cfg.tts_model_path, trainer=trainer
    )

    # Initialize the weights of the model from another model, if provided via config
    asr_model.maybe_init_from_pretrained_checkpoint(cfg)

    trainer.validate(asr_model)
    trainer.fit(asr_model)

    if hasattr(cfg.model, 'test_ds') and cfg.model.test_ds.manifest_filepath is not None:
        if asr_model.prepare_test(trainer):
            trainer.test(asr_model)


if __name__ == '__main__':
    main()  # noqa pylint: disable=no-value-for-parameter
