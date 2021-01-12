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
from omegaconf import DictConfig

from nemo.collections.asr.models.wav2vec.wav2vec_model import Wav2VecEncoderModel
from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.exp_manager import exp_manager


"""
Pre-train a wav2vec 2.0 transformer model on audio. Uses a contrastive loss function to pre-train on unlabelled audio,
using a task similar to masked language modeling in NLP. In wav2vec, we mask portions of the audio 
and the model is trained by minimising the distance of the ground truth for the masked section, 
using the ground truth quantized codebook representation. Distractors are obtained from other time steps.
See :class:`Wav2VecCriterion` for more information.

Reference: https://arxiv.org/abs/2006.11477

    python examples/asr/experimental/wav2vec/wav2vec_pretrain.py \
        model.train_ds.manifest_filepath="./examples/asr/train.tsv" \
        model.validation_ds.manifest_filepath="./examples/asr/valid.tsv" \
        trainer.gpus=1 \
        trainer.max_epochs=100
        
Basic run (on CPU for 50 epochs):
    python examples/asr/experimental/wav2vec/wav2vec_pretrain.py \
        model.train_ds.manifest_filepath="./examples/asr/train.tsv" \
        model.validation_ds.manifest_filepath="./examples/asr/valid.tsv" \
        trainer.gpus=1 \
        trainer.max_epochs=50

Using wav2vec-large with mixed precision:
    python examples/asr/experimental/wav2vec/wav2vec_pretrain.py \
        --config-name=wav2vec_pretrain_large \
        model.train_ds.manifest_filepath="./examples/asr/train.tsv" \
        model.validation_ds.manifest_filepath="./examples/asr/valid.tsv" \
        trainer.gpus=1 \
        trainer.max_epochs=100 \
        trainer.precision=16

Add PyTorch Lightning Trainer arguments from CLI:
    python wav2vec.py \
        ... \
        +trainer.fast_dev_run=true

Hydra logs will be found in "$(./outputs/$(date +"%y-%m-%d")/$(date +"%H-%M-%S")/.hydra)"
PTL logs will be found in "$(./outputs/$(date +"%y-%m-%d")/$(date +"%H-%M-%S")/lightning_logs)"

Override some args of optimizer:
    python examples/asr/experimental/wav2vec/wav2vec_pretrain.py \
        model.train_ds.manifest_filepath="./examples/asr/train.tsv" \
        model.validation_ds.manifest_filepath="./examples/asr/valid.tsv" \
        trainer.gpus=2 \
        trainer.max_epochs=2 \
        model.optim.args.params.betas=[0.8,0.5] \
        model.optim.args.params.weight_decay=0.0001

Override optimizer entirely
    python examples/asr/experimental/wav2vec/wav2vec_pretrain.py \
        model.train_ds.manifest_filepath="./examples/asr/train.tsv" \
        model.validation_ds.manifest_filepath="./examples/asr/valid.tsv" \
        trainer.gpus=2 \
        trainer.max_epochs=2 \
        ~model.optim.args \
        +model.optim.args.betas=[0.8,0.5]\
        +model.optim.args.weight_decay=0.0005

"""


@hydra_runner(config_path="configs", config_name="wav2vec_pretrain")
def main(cfg: DictConfig):
    logging.info("Application config\n" + cfg.pretty())

    trainer = pl.Trainer(**cfg.trainer)
    exp_manager(trainer, cfg.get("exp_manager", None))

    wav2vec_encoder_model = Wav2VecEncoderModel(cfg=cfg.model, trainer=trainer)
    trainer.fit(wav2vec_encoder_model)


if __name__ == '__main__':
    main()  # noqa pylint: disable=no-value-for-parameter
