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

# TODO: This is WIP and needs a lot of polishing
# python speech_to_text_bpe.py \
#         --asr_model "./experimental/configs/contextnet_128_v2.yaml" \
#         --train_dataset "./an4/train_manifest.json" \
#         --eval_dataset "./an4/test_manifest.json" \
#         --tokenizer_path "./an4/tokenizer/LibriSpeechTokenizer/librispeech_tokenizer_bpe_v1024/"
#         --gpus 2 \
#         --distributed_backend "ddp" \
#         --max_epochs 100 \
#         --optimizer adamw \
#         --lr 0.1 \
#         --opt_args weight_decay=1e-4 betas=0.9,0.999 \
#         ---warmup_ratio=0.05 --min_lr 1e-6

from argparse import ArgumentParser

import hydra
import pytorch_lightning as pl
from pytorch_lightning.logging import WandbLogger

from nemo.collections.asr.models.ctc_bpe_models import EncDecCTCModelBPE
from nemo.utils import logging


@hydra.main(config_path="conf", config_name="config")
def main(cfg):
    logging.info(f'Hydra config: {cfg.pretty()}')

    asr_model = EncDecCTCModelBPE(
        preprocessor_config=cfg.preprocessor,
        encoder_config=cfg.encoder,
        decoder_config=cfg.decoder,
        tokenizer_path=cfg.tokenizer_path,
        spec_augment_config=cfg.spec_augment,
    )

    asr_model.setup_training_data(cfg.AudioToTextDataLayer)
    asr_model.setup_validation_data(cfg.AudioToTextDataLayer_eval)

    # Setup optimizer and scheduler
    if 'sched' in cfg.optim:
        if cfg.pl.trainer.max_steps is None:
            if cfg.pl.trainer.gpus == 0:
                # training on CPU
                iters_per_batch = cfg.pl.trainer.max_epochs / float(
                    cfg.pl.trainer.num_nodes * cfg.pl.trainer.accumulate_grad_batches
                )
            else:
                iters_per_batch = cfg.pl.trainer.max_epochs / float(
                    cfg.pl.trainer.gpus * cfg.pl.trainer.num_nodes * cfg.pl.trainer.accumulate_grad_batches
                )
            cfg.optim.sched.iters_per_batch = iters_per_batch
        else:
            cfg.optim.sched.max_steps = cfg.pl.trainer.max_steps

    asr_model.setup_optimization(cfg.optim)

    trainer = pl.Trainer(**cfg.pl.trainer)

    if 'logger' in cfg:
        if cfg.logger.experiment_name is not None and cfg.logger.project_name is not None:
            logger = WandbLogger(name=cfg.logger.experiment_name, project=cfg.logger.project_name)
            trainer.configure_logger(logger)

    trainer.fit(asr_model)


if __name__ == '__main__':
    main()
