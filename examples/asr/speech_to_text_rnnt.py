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
# Training the model

Basic run (on CPU for 50 epochs):
    python examples/asr/speech_to_text_rnnt.py \
        # (Optional: --config-path=<path to dir of configs> --config-name=<name of config without .yaml>) \
        model.train_ds.manifest_filepath="<path to manifest file>" \
        model.validation_ds.manifest_filepath="<path to manifest file>" \
        trainer.gpus=0 \
        trainer.max_epochs=50


Add PyTorch Lightning Trainer arguments from CLI:
    python speech_to_text_rnnt.py \
        ... \
        +trainer.fast_dev_run=true

Hydra logs will be found in "$(./outputs/$(date +"%y-%m-%d")/$(date +"%H-%M-%S")/.hydra)"
PTL logs will be found in "$(./outputs/$(date +"%y-%m-%d")/$(date +"%H-%M-%S")/lightning_logs)"

Override some args of optimizer:
    python speech_to_text_rnnt.py \
    --config-path="experimental/contextnet_rnnt" \
    --config-name="config_rnnt" \
    model.train_ds.manifest_filepath="./an4/train_manifest.json" \
    model.validation_ds.manifest_filepath="./an4/test_manifest.json" \
    trainer.gpus=2 \
    trainer.precision=16 \
    trainer.max_epochs=2 \
    model.optim.betas=[0.8,0.5] \
    model.optim.weight_decay=0.0001

Override optimizer entirely
    python speech_to_text_rnnt.py \
    --config-path="experimental/contextnet_rnnt" \
    --config-name="config_rnnt" \
    model.train_ds.manifest_filepath="./an4/train_manifest.json" \
    model.validation_ds.manifest_filepath="./an4/test_manifest.json" \
    trainer.gpus=2 \
    trainer.precision=16 \
    trainer.max_epochs=2 \
    model.optim.name=adamw \
    model.optim.lr=0.001 \
    ~model.optim.args \
    +model.optim.args.betas=[0.8,0.5]\
    +model.optim.args.weight_decay=0.0005

Finetune a model
1) Finetune from a .nemo file

```sh
    python examples/asr/speech_to_text_rnnt.py \
        --config-path=<path to dir of configs> \
        --config-name=<name of config without .yaml>) \
        model.train_ds.manifest_filepath="<path to manifest file>" \
        model.validation_ds.manifest_filepath="<path to manifest file>" \
        trainer.gpus=-1 \
        trainer.max_epochs=50 \
        +init_from_nemo_model="<path to .nemo model file>"
```

2) Finetune from a pretrained model (via NGC)

```sh
    python examples/asr/speech_to_text_rnnt.py \
        --config-path=<path to dir of configs> \
        --config-name=<name of config without .yaml>) \
        model.train_ds.manifest_filepath="<path to manifest file>" \
        model.validation_ds.manifest_filepath="<path to manifest file>" \
        trainer.gpus=-1 \
        trainer.max_epochs=50 \
        +init_from_pretrained_model="<name of pretrained checkpoint>"
```
"""

import pytorch_lightning as pl
from omegaconf import OmegaConf, open_dict

from nemo.collections.asr.models import EncDecRNNTModel
from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.exp_manager import exp_manager


def restore_weights_if_required(model: EncDecRNNTModel, cfg: OmegaConf):
    if 'init_from_nemo_model' not in cfg and 'init_from_pretrained_model' not in cfg:
        # model weights do not need to be restored
        return

    if 'init_from_nemo_model' in cfg and 'init_from_pretrained_model' in cfg:
        raise ValueError("Cannot pass both `init_from_nemo_model` and `init_from_pretrained_model` to config!")

    if 'init_from_nemo_model' in cfg and cfg.init_from_nemo_model is not None:
        with open_dict(cfg):
            # Restore model
            model_path = cfg.pop('init_from_nemo_model')
            restored_model = EncDecRNNTModel.restore_from(model_path, map_location='cpu', strict=True)

            # Restore checkpoint into current model
            model.load_state_dict(restored_model.state_dict(), strict=False)
            logging.info(f'Model checkpoint restored from nemo file with path : `{model_path}`')

            del restored_model

    if 'init_from_pretrained_model' in cfg and cfg.init_from_pretrained_model is not None:
        with open_dict(cfg):
            # Restore model
            model_name = cfg.pop('init_from_pretrained_model')
            restored_model = EncDecRNNTModel.from_pretrained(model_name, map_location='cpu', strict=True)

            # Restore checkpoint into current model
            model.load_state_dict(restored_model.state_dict(), strict=False)
            logging.info(f'Model checkpoint restored from pretrained chackpoint with name : `{model_name}`')

            del restored_model


@hydra_runner(config_path="experimental/contextnet_rnnt", config_name="config_rnnt")
def main(cfg):
    logging.info(f'Hydra config: {OmegaConf.to_yaml(cfg)}')

    trainer = pl.Trainer(**cfg.trainer)
    exp_manager(trainer, cfg.get("exp_manager", None))
    asr_model = EncDecRNNTModel(cfg=cfg.model, trainer=trainer)

    restore_weights_if_required(asr_model, cfg)

    trainer.fit(asr_model)

    if hasattr(cfg.model, 'test_ds') and cfg.model.test_ds.manifest_filepath is not None:
        gpu = 1 if cfg.trainer.gpus != 0 else 0
        test_trainer = pl.Trainer(
            gpus=gpu,
            precision=trainer.precision,
            amp_level=trainer.accelerator_connector.amp_level,
            amp_backend=cfg.trainer.get("amp_backend", "native"),
        )
        if asr_model.prepare_test(test_trainer):
            test_trainer.test(asr_model)


if __name__ == '__main__':
    main()  # noqa pylint: disable=no-value-for-parameter
