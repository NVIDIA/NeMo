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

from dataclasses import dataclass
from typing import Optional, Tuple

from omegaconf.omegaconf import MISSING

from nemo.collections.nlp.data.machine_translation.machine_translation_dataset import TranslationDataConfig
from nemo.collections.nlp.models.enc_dec_nlp_model import EncDecNLPModelConfig
from nemo.collections.nlp.modules.common.token_classifier import TokenClassifierConfig
from nemo.collections.nlp.modules.common.tokenizer_utils import TokenizerConfig
from nemo.collections.nlp.modules.common.transformer.transformer import TransformerConfig, TransformerEncoderConfig
from nemo.core.config.modelPT import ModelConfig, OptimConfig, SchedConfig


@dataclass
class MTEncDecModelConfig(EncDecNLPModelConfig):
    train_ds: Optional[TranslationDataConfig] = None
    validation_ds: Optional[TranslationDataConfig] = None
    test_ds: Optional[TranslationDataConfig] = None
    beam_size: int = 4
    len_pen: float = 0.0
    max_generation_delta: int = 3
    label_smoothing: Optional[float] = 0.0
    src_language: str = 'en'
    tgt_language: str = 'en'
    find_unused_parameters: Optional[bool] = True
    shared_tokenizer: Optional[bool] = True
    preproc_out_dir: Optional[str] = None
    sentencepiece_model: Optional[str] = None


@dataclass
class AAYNBaseSchedConfig(SchedConfig):
    name: str = 'InverseSquareRootAnnealing'
    warmup_ratio: Optional[float] = None
    last_epoch: int = -1


# TODO: Refactor this dataclass to to support more optimizers (it pins the optimizer to Adam-like optimizers).
@dataclass
class AAYNBaseOptimConfig(OptimConfig):
    name: str = 'adam'
    lr: float = 1e-3
    betas: Tuple[float, float] = (0.9, 0.98)
    weight_decay: float = 0.0
    sched: Optional[AAYNBaseSchedConfig] = AAYNBaseSchedConfig()


@dataclass
class AAYNBaseConfig(MTEncDecModelConfig):
    # machine translation configurations
    num_val_examples: int = 3
    num_test_examples: int = 3
    beam_size: int = 1
    len_pen: float = 0.0
    max_generation_delta: int = 10
    label_smoothing: Optional[float] = 0.0

    # Attention is All You Need Base Configuration
    encoder_tokenizer: TokenizerConfig = TokenizerConfig(tokenizer_name='yttm')
    decoder_tokenizer: TokenizerConfig = TokenizerConfig(tokenizer_name='yttm')

    encoder: TransformerEncoderConfig = TransformerEncoderConfig(
        hidden_size=512,
        inner_size=2048,
        num_layers=6,
        num_attention_heads=8,
        ffn_dropout=0.1,
        attn_score_dropout=0.1,
        attn_layer_dropout=0.1,
    )

    decoder: TransformerConfig = TransformerConfig(
        hidden_size=512,
        inner_size=2048,
        num_layers=6,
        num_attention_heads=8,
        ffn_dropout=0.1,
        attn_score_dropout=0.1,
        attn_layer_dropout=0.1,
    )

    head: TokenClassifierConfig = TokenClassifierConfig(log_softmax=True)

    # dataset configurations
    train_ds: Optional[TranslationDataConfig] = TranslationDataConfig(
        src_file_name=MISSING,
        tgt_file_name=MISSING,
        tokens_in_batch=512,
        clean=True,
        shuffle=True,
        cache_ids=False,
        use_cache=False,
    )
    validation_ds: Optional[TranslationDataConfig] = TranslationDataConfig(
        src_file_name=MISSING,
        tgt_file_name=MISSING,
        tokens_in_batch=512,
        clean=False,
        shuffle=False,
        cache_ids=False,
        use_cache=False,
    )
    test_ds: Optional[TranslationDataConfig] = TranslationDataConfig(
        src_file_name=MISSING,
        tgt_file_name=MISSING,
        tokens_in_batch=512,
        clean=False,
        shuffle=False,
        cache_ids=False,
        use_cache=False,
    )
    optim: Optional[OptimConfig] = AAYNBaseOptimConfig()
