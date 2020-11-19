from dataclasses import dataclass
from nemo.collections.nlp.modules.common.tokenizer_utils import get_tokenizer
from typing import Dict, Optional, Union

from omegaconf.dictconfig import DictConfig
from pytorch_lightning.trainer.trainer import Trainer
from nemo.collections.nlp.models.nlp_model import NLPModel


@dataclass
class TokenizerConfig:
    tokenizer_name: str
    tokenizer_model: Optional[str] = None
    vocab_file: Optional[str] = None
    special_tokens: Optional[Dict[str, str]] = None


@dataclass
class EncDecNLPModelConfig:
    enc_tokenizer: TokenizerConfig
    dec_tokenizer: TokenizerConfig


class EncDecNLPModel(NLPModel):
    """Base class for encoder-decoder NLP models.
    """

    def __init__(self, cfg: Union[EncDecNLPModelConfig, DictConfig], trainer: Trainer = None):
        self._enc_tokenizer = None
        self._dec_tokenizer = None

    @property
    def enc_tokenizer(self):
        return self._enc_tokenizer

    @enc_tokenizer.setter
    def enc_tokenizer(self, tokenizer):
        self._enc_tokenizer = tokenizer

    @property
    def dec_tokenizer(self):
        return self._enc_tokenizer

    @dec_tokenizer.setter
    def dec_tokenizer(self, tokenizer):
        self._dec_tokenizer = tokenizer

    def setup_enc_dec_tokenizers(self, cfg: EncDecNLPModelConfig):
        self.enc_tokenizer = get_tokenizer(**cfg.enc_tokenizer)
        self.dec_tokenizer = get_tokenizer(**cfg.dec_tokenizer)
