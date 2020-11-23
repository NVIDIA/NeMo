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

from dataclasses import asdict
from nemo.collections.nlp.models.enc_dec_nlp_model import (
    TokenizerConfig,
    TransformerDecoderConfig,
    TransformerEmbeddingConfig,
    TransformerEncoderConfig,
)
import hydra
from hydra.utils import instantiate

from nemo.collections.nlp.models.machine_translation.mt_enc_dec_model import MTEncDecModel
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
    trainer_config = TrainerConfig()
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

    mt_config = MTEncDecModelConfig(
        encoder_tokenizer=encoder_tokenizer_config,
        decoder_tokenizer=decoder_tokenizer_config,
        encoder_embedding=encoder_embedding_config,
        decoder_embedding=decoder_embedding_config,
        encoder=encoder_config,
        decoder=decoder_config,
    )

    mt_model = MTEncDecModel(mt_config, trainer=trainer)

    # trainer.fit(transformer_mt)
    # transformer_mt.save_to("transformer.nemo")
    # print("Model saved to: transformer.nemo")


if __name__ == '__main__':
    main()
