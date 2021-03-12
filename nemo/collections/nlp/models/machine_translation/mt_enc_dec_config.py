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
from typing import Any, Optional, Tuple

from omegaconf.omegaconf import MISSING

from nemo.collections.nlp.data.machine_translation.machine_translation_dataset import TranslationDataConfig
from nemo.collections.nlp.models.enc_dec_nlp_model import EncDecNLPModelConfig
from nemo.collections.nlp.modules.common.token_classifier import TokenClassifierConfig
from nemo.collections.nlp.modules.common.tokenizer_utils import TokenizerConfig
from nemo.collections.nlp.modules.common.transformer.transformer import (
    NeMoTransformerConfig,
    NeMoTransformerEncoderConfig,
)
from nemo.core.config.modelPT import ModelConfig, OptimConfig, SchedConfig


@dataclass
class MTSchedConfig(SchedConfig):
    name: str = 'InverseSquareRootAnnealing'
    warmup_ratio: Optional[float] = None
    last_epoch: int = -1


# TODO: Refactor this dataclass to to support more optimizers (it pins the optimizer to Adam-like optimizers).
@dataclass
class MTOptimConfig(OptimConfig):
    name: str = 'adam'
    lr: float = 1e-3
    betas: Tuple[float, float] = (0.9, 0.98)
    weight_decay: float = 0.0
    sched: Optional[MTSchedConfig] = MTSchedConfig()


@dataclass
class MTEncDecModelConfig(EncDecNLPModelConfig):
    # machine translation configurations
    num_val_examples: int = 3
    num_test_examples: int = 3
    max_generation_delta: int = 10
    label_smoothing: Optional[float] = 0.0
    beam_size: int = 4
    len_pen: float = 0.0
    src_language: str = 'en'
    tgt_language: str = 'en'
    find_unused_parameters: Optional[bool] = True
    shared_tokenizer: Optional[bool] = True
    preproc_out_dir: Optional[str] = None

    # network architecture configuration
    encoder_tokenizer: Any = MISSING
    encoder: Any = MISSING

    decoder_tokenizer: Any = MISSING
    decoder: Any = MISSING

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
    optim: Optional[OptimConfig] = MTOptimConfig()


@dataclass
class AAYNBaseConfig(MTEncDecModelConfig):

    # Attention is All You Need Base Configuration
    encoder_tokenizer: TokenizerConfig = TokenizerConfig(library='yttm')
    decoder_tokenizer: TokenizerConfig = TokenizerConfig(library='yttm')

    encoder: NeMoTransformerEncoderConfig = NeMoTransformerEncoderConfig(
        library='nemo',
        model_name=None,
        pretrained=False,
        hidden_size=512,
        inner_size=2048,
        num_layers=6,
        num_attention_heads=8,
        ffn_dropout=0.1,
        attn_score_dropout=0.1,
        attn_layer_dropout=0.1,
    )

    decoder: NeMoTransformerConfig = NeMoTransformerConfig(
        library='nemo',
        model_name=None,
        pretrained=False,
        inner_size=2048,
        num_layers=6,
        num_attention_heads=8,
        ffn_dropout=0.1,
        attn_score_dropout=0.1,
        attn_layer_dropout=0.1,
    )
