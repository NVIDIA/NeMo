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
import hydra
import pytorch_lightning as pl

from nemo.collections.asr.models import EncDecCTCModel
from nemo.utils.exp_manager import exp_manager


"""
Basic run (on CPU for 50 epochs):
    python examples/asr/speech_to_text.py \
        model.train_ds.manifest_filepath="/Users/okuchaiev/Data/an4_dataset/an4_train.json" \
        model.validation_ds.manifest_filepath="/Users/okuchaiev/Data/an4_dataset/an4_val.json" \
        hydra.run.dir="." \
        pl.trainer.gpus=0 \
        pl.trainer.max_epochs=50


Add PyTorch Lightning Trainer arguments from CLI:
    python speech_to_text.py \
        ... \
        +pl.trainer.fast_dev_run=true

Hydra logs will be found in "$(./outputs/$(date +"%y-%m-%d")/$(date +"%H-%M-%S")/.hydra)"
PTL logs will be found in "$(./outputs/$(date +"%y-%m-%d")/$(date +"%H-%M-%S")/lightning_logs)"

Override some args of optimizer:
    python speech_to_text.py \
    model.train_ds.manifest_filepath="./an4/train_manifest.json" \
    model.validation_ds.manifest_filepath="./an4/test_manifest.json" \
    hydra.run.dir="." \
    pl.trainer.gpus=2 \
    pl.trainer.max_epochs=2 \
    model.optim.args.params.betas=[0.8,0.5] \
    model.optim.args.params.weight_decay=0.0001

Overide optimizer entirely
    python speech_to_text.py \
    model.train_ds.manifest_filepath="./an4/train_manifest.json" \
    model.validation_ds.manifest_filepath="./an4/test_manifest.json" \
    hydra.run.dir="." \
    pl.trainer.gpus=2 \
    pl.trainer.max_epochs=2 \
    model.optim.name=adamw \
    model.optim.lr=0.001 \
    ~model.optim.args \
    +model.optim.args.betas=[0.8,0.5]\
    +model.optim.args.weight_decay=0.0005

"""


@hydra.main(config_path="conf", config_name="config")
def main(cfg):
    trainer = pl.Trainer(**cfg.pl.trainer)
    # exp_manager(trainer, cfg.get("exp_manager", None))
    asr_model = EncDecCTCModel(cfg=cfg.model, trainer=trainer)
    trainer.fit(asr_model)


if __name__ == '__main__':
    main()  # noqa pylint: disable=no-value-for-parameter
