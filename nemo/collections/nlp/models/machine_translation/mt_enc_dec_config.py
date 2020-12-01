from dataclasses import dataclass
from nemo.core.config.modelPT import ModelConfig
from typing import Optional

from hydra.utils import instantiate
from omegaconf import MISSING

from nemo.collections.nlp.models.enc_dec_nlp_model import (
    TokenClassifierConfig,
    TokenizerConfig,
    TransformerDecoderConfig,
    TransformerEmbeddingConfig,
    TransformerEncoderConfig,
    TransformerEncoderDefaultConfig,
)
from nemo.collections.nlp.models.machine_translation.mt_enc_dec_model import (
    MTEncDecModel,
    MTEncDecModelConfig,
    MTOptimConfig,
    MTSchedConfig,
    TranslationDataConfig,
)
from nemo.core.config import hydra_runner
from nemo.core.config.pytorch_lightning import TrainerConfig
from nemo.utils import logging
from nemo.utils.exp_manager import ExpManagerConfig, exp_manager


@dataclass
class MTTransformerBase(ModelConfig):
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
    # encoder: TransformerEncoderConfig = TransformerEncoderConfig(
    #     hidden_size=512,
    #     inner_size=2048,
    #     num_layers=6,
    #     num_attention_heads=8,
    #     ffn_dropout=0.1,
    #     attn_score_dropout=0.1,
    #     attn_layer_dropout=0.1,
    # )
    encoder: TransformerEncoderConfig = TransformerEncoderDefaultConfig()
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
    max_generation_delta: int = 10
    label_smoothing: Optional[float] = 0.0
