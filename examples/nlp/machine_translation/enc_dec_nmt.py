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
from typing import Any, Optional, Tuple

import hydra
import pytorch_lightning as pl
from hydra.utils import instantiate
from omegaconf import MISSING, DictConfig

from nemo.collections.common.callbacks import MachineTranslationLogEvalCallback
from nemo.collections.nlp.models.enc_dec_nlp_model import (
    EncDecNLPModelConfig,
    OptimConfig,
    SchedConfig,
    TokenClassifierConfig,
    TokenizerConfig,
    TransformerDecoderConfig,
    TransformerEmbeddingConfig,
    TransformerEncoderConfig,
)
from nemo.collections.nlp.models.machine_translation import TransformerMTModel
from nemo.collections.nlp.models.machine_translation.mt_enc_dec_model import (
    MTEncDecModel,
    MTEncDecModelConfig,
    MTOptimConfig,
    MTSchedConfig,
    TranslationDataConfig,
)
from nemo.core import optim
from nemo.core.config import hydra_runner, optimizers
from nemo.core.config.pytorch_lightning import TrainerConfig
from nemo.utils import logging
from nemo.utils.exp_manager import exp_manager


@dataclass
class DefaultConfig:
    # pytorch lightning trainer configurations
    trainer: TrainerConfig = TrainerConfig(
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

    # dataset configurations
    train_ds: Optional[TranslationDataConfig] = TranslationDataConfig(
        src_file_name=MISSING,
        tgt_file_name=MISSING,
        tokens_in_batch=512,
        clean=True,
        shuffle=True,
        cache_ids=True,
        use_cache=True,
    )
    validation_ds: Optional[TranslationDataConfig] = TranslationDataConfig(
        src_file_name=MISSING,
        tgt_file_name=MISSING,
        tokens_in_batch=512,
        clean=False,
        shuffle=False,
        cache_ids=True,
        use_cache=True,
    )
    test_ds: Optional[TranslationDataConfig] = TranslationDataConfig(
        src_file_name=MISSING,
        tgt_file_name=MISSING,
        tokens_in_batch=512,
        clean=False,
        shuffle=False,
        cache_ids=True,
        use_cache=True,
    )

    # model architecture configurations
    encoder_tokenizer: TokenizerConfig = TokenizerConfig(tokenizer_name='yttm')
    decoder_tokenizer: TokenizerConfig = TokenizerConfig(tokenizer_name='yttm')
    encoder_embedding: TransformerEmbeddingConfig = TransformerEmbeddingConfig(
        vocab_size=37000, hidden_size=512, embedding_dropout=0.1
    )
    encoder: TransformerEncoderConfig = TransformerEncoderConfig(
        hidden_size=512,
        inner_size=2048,
        num_layers=6,
        num_attention_heads=8,
        ffn_dropout=0.1,
        attn_score_dropout=0.1,
        attn_layer_dropout=0.1,
    )
    decoder_embedding: TransformerEmbeddingConfig = TransformerEmbeddingConfig(
        vocab_size=37000, hidden_size=512, embedding_dropout=0.1
    )
    decoder: TransformerDecoderConfig = TransformerDecoderConfig(
        hidden_size=512,
        inner_size=2048,
        num_layers=6,
        num_attention_heads=8,
        ffn_dropout=0.1,
        attn_score_dropout=0.1,
        attn_layer_dropout=0.1,
    )
    head: TokenClassifierConfig = TokenClassifierConfig(
        hidden_size=decoder.hidden_size, num_classes=decoder_embedding.vocab_size, log_softmax=True
    )

    # machine translation configurations
    num_val_examples: int = 3
    num_test_examples: int = 3
    beam_size: int = 1
    len_pen: float = 0.0
    max_generation_delta: int = 50
    label_smoothing: Optional[float] = 0.0

    # optimizer configurations
    optim: MTOptimConfig = MTOptimConfig(sched=MTSchedConfig())


@hydra_runner(config_path="conf", config_name="enc_dec", schema=DefaultConfig)
def main(cfg: DefaultConfig) -> None:
    logging.info(f'Config: {cfg.pretty()}')
    trainer = instantiate(cfg.trainer)

    mt_config = MTEncDecModelConfig(
        encoder_tokenizer=cfg.encoder_tokenizer,
        decoder_tokenizer=cfg.decoder_tokenizer,
        encoder_embedding=cfg.encoder_embedding,
        decoder_embedding=cfg.decoder_embedding,
        encoder=cfg.encoder,
        decoder=cfg.decoder,
        head=cfg.head,
        optim=cfg.optim,
        train_ds=cfg.train_ds,
        validation_ds=cfg.validation_ds,
        test_ds=cfg.test_ds,
        beam_size=cfg.beam_size,
        len_pen=cfg.len_pen,
        max_generation_delta=cfg.max_generation_delta,
        label_smoothing=cfg.label_smoothing,
    )

    mt_model = MTEncDecModel(mt_config, trainer=trainer)

    trainer.fit(mt_model)
    # transformer_mt.save_to("transformer.nemo")
    # print("Model saved to: transformer.nemo")


if __name__ == '__main__':
    main()
