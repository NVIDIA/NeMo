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
from nemo.collections.nlp.models.enc_dec_nlp_model import TokenizerConfig, TransformerEmbeddingConfig
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
    enc_tokenizer_config = TokenizerConfig(
        tokenizer_name="yttm", tokenizer_model='/raid/data/68792/tokenizer.BPE.37K.model'
    )
    dec_tokenizer_config = TokenizerConfig(
        tokenizer_name="yttm", tokenizer_model='/raid/data/68792/tokenizer.BPE.37K.model'
    )
    enc_embedding_config = TransformerEmbeddingConfig(vocab_size=37000, hidden_size=512, embedding_dropout=0.1)
    # dec embedding happens to be the same in this case (will change for other configs)
    dec_embedding_config = TransformerEmbeddingConfig(vocab_size=37000, hidden_size=512, embedding_dropout=0.1)

    mt_config = MTEncDecModelConfig(
        enc_tokenizer=enc_tokenizer_config,
        dec_tokenizer=dec_tokenizer_config,
        enc_embedding=enc_embedding_config,
        dec_embedding=dec_embedding_config,
    )

    mt_model = MTEncDecModel(mt_config, trainer=trainer)

    # trainer.fit(transformer_mt)
    # transformer_mt.save_to("transformer.nemo")
    # print("Model saved to: transformer.nemo")


if __name__ == '__main__':
    main()
