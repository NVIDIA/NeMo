import math
from abc import abstractmethod
from dataclasses import MISSING, asdict, dataclass
from typing import Dict, Optional, Union

from omegaconf.dictconfig import DictConfig
from pytorch_lightning.trainer.trainer import Trainer

from nemo.collections.nlp.models.nlp_model import NLPModel
from nemo.collections.nlp.modules.common.tokenizer_utils import get_tokenizer
from nemo.core.config.pytorch_lightning import TrainerConfig


@dataclass
class TokenizerConfig:
    tokenizer_name: str = MISSING
    tokenizer_model: Optional[str] = None
    vocab_file: Optional[str] = None
    special_tokens: Optional[Dict[str, str]] = None


@dataclass
class EmbeddingConfig:
    vocab_size: int = MISSING
    hidden_size: int = MISSING
    max_sequence_length: int = 512
    num_token_types: int = 2
    embedding_dropout: float = 0.0
    learn_positional_encodings: bool = False


@dataclass
class TransformerEmbeddingConfig(EmbeddingConfig):
    _target_: str = "nemo.collections.nlp.modules.common.transformer.TransformerEmbedding"


@dataclass
class EncoderConfig:
    hidden_size: int = MISSING


@dataclass
class DecoderConfig:
    hidden_size: int = MISSING


@dataclass
class TransformerEncoderConfig(EncoderConfig):
    inner_size: int = MISSING
    num_layers: int = MISSING
    _target_: str = 'nemo.collections.nlp.modules.common.transformer.TransformerEncoder'
    num_attention_heads: int = 1
    ffn_dropout: float = 0.0
    attn_score_dropout: float = 0.0
    attn_layer_dropout: float = 0.0
    hidden_act: str = 'relu'
    mask_future: bool = False


@dataclass
class TransformerDecoderConfig(DecoderConfig):
    inner_size: int = MISSING
    num_layers: int = MISSING
    _target_: str = 'nemo.collections.nlp.modules.common.transformer.TransformerDecoder'
    num_attention_heads: int = 1
    ffn_dropout: float = 0.0
    attn_score_dropout: float = 0.0
    attn_layer_dropout: float = 0.0
    hidden_act: str = 'relu'


@dataclass
class HeadConfig:
    num_classes: int = MISSING


@dataclass
class TokenClassifierConfig(HeadConfig):
    hidden_size: int = MISSING
    num_classes: int = MISSING
    _target_: str = 'nemo.collections.nlp.modules.common.token_classifier.TokenClassifier'
    num_layers: int = 1
    activation: str = 'relu'
    log_softmax: bool = True
    dropout: float = 0.0
    use_transformer_init: bool = True


@dataclass
class SchedConfig:
    name: str = None


@dataclass
class OptimConfig:
    name: str = MISSING
    lr: float = MISSING
    sched: Optional[SchedConfig] = None


@dataclass
class EncDecNLPModelConfig:
    encoder_tokenizer: TokenizerConfig = MISSING
    decoder_tokenizer: TokenizerConfig = MISSING
    encoder_embedding: EmbeddingConfig = MISSING
    decoder_embedding: EmbeddingConfig = MISSING
    encoder: EncoderConfig = MISSING
    decoder: DecoderConfig = MISSING
    head: HeadConfig = MISSING
    optim: OptimConfig = None
    vocab_divisibile_by_eight: bool = True  # TODO: should this go somewhere else?


class EncDecNLPModel(NLPModel):
    """Base class for encoder-decoder NLP models.
    """

    def __init__(self, cfg: EncDecNLPModelConfig, trainer: Trainer = None):
        super().__init__(cfg=cfg, trainer=trainer)

    @property
    def encoder_vocab_size(self):
        return self._encoder_vocab_size

    @encoder_vocab_size.setter
    def encoder_vocab_size(self, size: int):
        self._encoder_vocab_size = size

    @property
    def decoder_vocab_size(self):
        return self._decoder_vocab_size

    @decoder_vocab_size.setter
    def decoder_vocab_size(self, size: int):
        self._decoder_vocab_size = size

    @property
    def encoder_tokenizer(self):
        return self._encoder_tokenizer

    @encoder_tokenizer.setter
    def encoder_tokenizer(self, tokenizer):
        self._encoder_tokenizer = tokenizer

    @property
    def decoder_tokenizer(self):
        return self._decoder_tokenizer

    @decoder_tokenizer.setter
    def decoder_tokenizer(self, tokenizer):
        self._decoder_tokenizer = tokenizer

    @property
    def encoder_embedding(self):
        return self._encoder_embedding

    @encoder_embedding.setter
    def encoder_embedding(self, embedding):
        self._encoder_embedding = embedding

    @property
    def decoder_embedding(self):
        return self._decoder_embedding

    @decoder_embedding.setter
    def decoder_embedding(self, embedding):
        self._decoder_embedding = embedding

    @property
    def encoder(self):
        return self._encoder

    @encoder.setter
    def encoder(self, encoder):
        self._encoder = encoder

    @property
    def decoder(self):
        return self._decoder

    @decoder.setter
    def decoder(self, decoder):
        self._decoder = decoder

    def setup_enc_dec_tokenizers(self, cfg: EncDecNLPModelConfig):
        self.encoder_tokenizer = get_tokenizer(**cfg.encoder_tokenizer)
        self.decoder_tokenizer = get_tokenizer(**cfg.decoder_tokenizer)
        self.encoder_vocab_size = self.encoder_tokenizer.vocab_size
        self.decoder_vocab_size = self.decoder_tokenizer.vocab_size
