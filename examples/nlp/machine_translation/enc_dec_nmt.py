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

from dataclasses import asdict, dataclass
from logging import NullHandler
from nemo.core import optim
from nemo.core.config import optimizers
from nemo.collections.nlp.models.enc_dec_nlp_model import (
    OptimConfig,
    SchedConfig,
    TokenClassifierConfig,
    TokenizerConfig,
    TransformerDecoderConfig,
    TransformerEmbeddingConfig,
    TransformerEncoderConfig,
)
import hydra
from hydra.utils import instantiate

from nemo.collections.nlp.models.machine_translation.mt_enc_dec_model import MTEncDecModel, TranslationDataConfig
import pytorch_lightning as pl
from omegaconf import DictConfig

from nemo.collections.common.callbacks import MachineTranslationLogEvalCallback
from nemo.collections.nlp.models.machine_translation import TransformerMTModel
from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.exp_manager import exp_manager

from nemo.collections.nlp.models.machine_translation.mt_enc_dec_model import MTEncDecModelConfig
from nemo.core.config.pytorch_lightning import TrainerConfig


@hydra_runner(config_path="conf", config_name="enc_dec")
def main(cfg: DictConfig) -> None:
    logging.info(f'Config: {cfg.pretty()}')
    trainer_config = TrainerConfig(
        gpus=1,
        num_nodes=1,
        max_epochs=1,
        max_steps=10000,
        precision=16,
        accelerator='ddp',
        checkpoint_callback=False,
        logger=False,
        log_every_n_steps=10,
        val_check_interval=0.1,
    )
    trainer = instantiate(trainer_config)
    # trainer = pl.Trainer(**asdict(trainer_config))
    # exp_manager(trainer, cfg.get("exp_manager", None))
    encoder_tokenizer_config = TokenizerConfig(
        tokenizer_name="yttm", tokenizer_model='/raid/data/68792/tokenizer.BPE.37K.model'
    )
    decoder_tokenizer_config = TokenizerConfig(
        tokenizer_name="yttm", tokenizer_model='/raid/data/68792/tokenizer.BPE.37K.model'
    )
    encoder_embedding_config = TransformerEmbeddingConfig(vocab_size=37000, hidden_size=512, embedding_dropout=0.1)
    # dec embedding happens to be the same in this case (will change for other configs)
    decoder_embedding_config = TransformerEmbeddingConfig(vocab_size=37000, hidden_size=512, embedding_dropout=0.1)

    encoder_config = TransformerEncoderConfig(
        hidden_size=512,
        inner_size=2048,
        num_layers=6,
        num_attention_heads=8,
        ffn_dropout=0.1,
        attn_score_dropout=0.1,
        attn_layer_dropout=0.1,
    )

    decoder_config = TransformerDecoderConfig(
        hidden_size=512,
        inner_size=2048,
        num_layers=6,
        num_attention_heads=8,
        ffn_dropout=0.1,
        attn_score_dropout=0.1,
        attn_layer_dropout=0.1,
    )

    head_config = TokenClassifierConfig(
        hidden_size=decoder_config.hidden_size, num_classes=decoder_embedding_config.vocab_size, log_softmax=True
    )

    sched_config = SchedConfig(name='InverseSquareRootAnnealing')
    sched_config.warmup_steps = None
    sched_config.warmup_ratio = 0.1
    sched_config.last_epoch = -1

    optim_config = OptimConfig(name='adam', lr=1e-3, sched=sched_config)
    optim_config.betas = [0.9, 0.98]
    optim_config.weight_decay = 0.0

    num_samples = -1  # for dev
    train_ds_config = TranslationDataConfig(
        src_file_name='/raid/data/68792/train.clean.en.shuffled',
        tgt_file_name='/raid/data/68792/train.clean.de.shuffled',
        tokens_in_batch=16000,
        clean=True,
        shuffle=True,
        num_samples=num_samples,
        cache_ids=True,
        use_cache=True,
    )

    validation_ds_config = TranslationDataConfig(
        src_file_name='/raid/data/68792/wmt14-en-de.src',
        tgt_file_name='/raid/data/68792/wmt14-en-de.ref',
        tokens_in_batch=512,
        clean=False,
        shuffle=False,
        num_samples=num_samples,
        cache_ids=True,
        use_cache=True,
    )

    test_ds_config = TranslationDataConfig(
        src_file_name='/raid/data/68792/wmt14-en-de.src',
        tgt_file_name='/raid/data/68792/wmt14-en-de.ref',
        tokens_in_batch=512,
        clean=False,
        shuffle=False,
        num_samples=num_samples,
        cache_ids=True,
        use_cache=True,
    )

    mt_config = MTEncDecModelConfig(
        encoder_tokenizer=encoder_tokenizer_config,
        decoder_tokenizer=decoder_tokenizer_config,
        encoder_embedding=encoder_embedding_config,
        decoder_embedding=decoder_embedding_config,
        encoder=encoder_config,
        decoder=decoder_config,
        head=head_config,
        optim=optim_config,
        train_ds=train_ds_config,
        validation_ds=validation_ds_config,
        test_ds=test_ds_config,
        beam_size=4,
        len_pen=0.6,
        max_generation_delta=50,
        label_smoothing=0.1,
    )

    mt_model = MTEncDecModel(mt_config, trainer=trainer)

    trainer.fit(mt_model)
    # transformer_mt.save_to("transformer.nemo")
    # print("Model saved to: transformer.nemo")


if __name__ == '__main__':
    main()
