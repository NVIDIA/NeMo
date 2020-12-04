from dataclasses import MISSING, dataclass
from typing import Any

from pytorch_lightning.trainer.trainer import Trainer

from nemo.collections.nlp.models.nlp_model import NLPModel
from nemo.collections.nlp.modules.common.tokenizer_utils import get_tokenizer


@dataclass
class EncDecNLPModelConfig:
    encoder_tokenizer: Any = MISSING
    decoder_tokenizer: Any = MISSING
    encoder_embedding: Any = MISSING
    decoder_embedding: Any = MISSING
    encoder: Any = MISSING
    decoder: Any = MISSING
    head: Any = MISSING


class EncDecNLPModel(NLPModel):
    """Base class for encoder-decoder NLP models.
    """

    def __init__(self, cfg: EncDecNLPModelConfig, trainer: Trainer = None):
        super().__init__(cfg=cfg, trainer=trainer)

    @property
    def encoder_vocab_size(self):
        if self.encoder and self.encoder.vocab_size:
            return self.encoder.vocab_size

    @property
    def decoder_vocab_size(self):
        if self.decoder and self.decoder.vocab_size:
            return self.decoder.vocab_size

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
