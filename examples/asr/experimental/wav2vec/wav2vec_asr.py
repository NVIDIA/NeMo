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

from nemo.collections.asr.models.wav2vec.wav2vec_asr_model import Wav2VecASRModel
from nemo.collections.asr.models.wav2vec.wav2vec_model import Wav2VecEncoderModel
from nemo.core.config import hydra_runner
from nemo.utils.exp_manager import exp_manager

"""
Fine-tune a pre-trained Wav2Vec 2.0 model on ASR data. 
Assumes you've pretrained a model on audio using the wav2vec.py script.
We add an additional classification layer to the model, and train the model using the ctc loss function using labeled
ASR datasets.

The code has been ported from fairseq which can be seen here: 
https://github.com/pytorch/fairseq/tree/master/examples/wav2vec

Reference: https://arxiv.org/abs/2006.11477

python examples/asr/wav2vec_asr.py \
        model.encoder_path="./examples/asr/wav2vec_checkpoints/wav2vec_pretrained.ckpt" # Path to pre-trained wav2vec
        model.train_ds.manifest_path="./examples/asr/train.tsv" \
        model.validation_ds.manifest_path="./examples/asr/valid.tsv" \
        model.test_ds.manifest_path="./examples/asr/valid.tsv" \
        hydra.run.dir="." \
        trainer.gpus=1 \
        trainer.max_epochs=100
        
        
Basic run (on CPU for 50 epochs):
    python examples/asr/wav2vec_asr.py \
        model.encoder_path="./examples/asr/wav2vec_checkpoints/wav2vec_pretrained.ckpt" # Path to pre-trained wav2vec
        model.train_ds.manifest_path="./examples/asr/train.tsv" \
        model.validation_ds.manifest_path="./examples/asr/valid.tsv" \
        model.test_ds.manifest_path="./examples/asr/valid.tsv" \
        hydra.run.dir="." \
        trainer.gpus=1 \
        trainer.max_epochs=50


Add PyTorch Lightning Trainer arguments from CLI:
    python wav2vec_asr.py \
        ... \
        +trainer.fast_dev_run=true

Hydra logs will be found in "$(./outputs/$(date +"%y-%m-%d")/$(date +"%H-%M-%S")/.hydra)"
PTL logs will be found in "$(./outputs/$(date +"%y-%m-%d")/$(date +"%H-%M-%S")/lightning_logs)"

Override some args of optimizer:
    python examples/asr/wav2vec_asr.py \
        model.encoder_path="./examples/asr/wav2vec_checkpoints/wav2vec_pretrained.ckpt" # Path to pre-trained wav2vec
        model.train_ds.manifest_path="./examples/asr/train.tsv" \
        model.validation_ds.manifest_path="./examples/asr/valid.tsv" \
        model.test_ds.manifest_path="./examples/asr/valid.tsv" \
        hydra.run.dir="." \
        trainer.gpus=2 \
        trainer.max_epochs=2 \
        model.optim.args.params.betas=[0.8,0.5] \
        model.optim.args.params.weight_decay=0.0001

Override optimizer entirely
    python examples/asr/wav2vec_asr.py \
        model.encoder_path="./examples/asr/wav2vec_checkpoints/wav2vec_pretrained.ckpt" # Path to pre-trained wav2vec
        model.train_ds.manifest_path="./examples/asr/train.tsv" \
        model.validation_ds.manifest_path="./examples/asr/valid.tsv" \
        model.test_ds.manifest_path="./examples/asr/valid.tsv" \
        hydra.run.dir="." \
        trainer.gpus=2 \
        trainer.max_epochs=2 \
        ~model.optim.args \
        +model.optim.args.betas=[0.8,0.5]\
        +model.optim.args.weight_decay=0.0005

"""


@hydra_runner(config_path="configs", config_name="wav2vec_asr")
def main(cfg: DictConfig):
    trainer = pl.Trainer(**cfg.trainer)
    exp_manager(trainer, cfg.get("exp_manager", None))

    encoder_model = Wav2VecEncoderModel.load_from_checkpoint(cfg['model']['encoder_path'])
    wav2vec_model = Wav2VecASRModel(encoder=encoder_model, cfg=cfg.model, trainer=trainer)
    trainer.fit(wav2vec_model)


if __name__ == '__main__':
    main()  # noqa pylint: disable=no-value-for-parameter
