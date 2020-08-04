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

import pytorch_lightning as pl

from nemo.collections.asr.models import EncDecClassificationModel
from nemo.core.config import hydra_runner
from nemo.utils.exp_manager import exp_manager


"""
Basic run (on 1 GPU + AMP O1 for 200 epochs):
    python examples/asr/speech_to_label.py \
        model.train_ds.manifest_filepath="<path to train manifest>" \
        model.validation_ds.manifest_filepath="<path to test manifest>" \
        pl.trainer.gpus=1 \
        pl.trainer.max_epochs=200 \
        +pl.trainer.precision=16 \
        +pl.trainer.amp_level=O1
"""


@hydra_runner(config_path="conf", config_name="matchboxnet_3x1x64_v1.yaml")
def main(cfg):
    trainer = pl.Trainer(**cfg.trainer)
    exp_manager(trainer, cfg.get("exp_manager", None))
    asr_model = EncDecClassificationModel(cfg=cfg.model, trainer=trainer)

    trainer.fit(asr_model)

    if hasattr(cfg.model, 'test_ds') and cfg.model.test_ds.manifest_filepath is not None:
        trainer = pl.Trainer()
        if asr_model.prepare_test(trainer):
            trainer.test(asr_model)


if __name__ == '__main__':
    main()  # noqa pylint: disable=no-value-for-parameter
