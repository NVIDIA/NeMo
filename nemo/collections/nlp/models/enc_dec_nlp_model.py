from dataclasses import dataclass
from typing import Any

from omegaconf.omegaconf import MISSING
from pytorch_lightning.trainer.trainer import Trainer

from nemo.collections.nlp.models.nlp_model import NLPModel
from nemo.collections.nlp.modules.common.decoder_module import DecoderModule
from nemo.collections.nlp.modules.common.encoder_module import EncoderModule
from nemo.collections.nlp.modules.common.tokenizer_utils import TokenizerConfig, get_tokenizer
from nemo.core.config.modelPT import ModelConfig


@dataclass
class EncDecNLPModelConfig(ModelConfig):
    encoder_tokenizer: TokenizerConfig = MISSING
    decoder_tokenizer: TokenizerConfig = MISSING
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
        return self.encoder_tokenizer.vocab_size

    @property
    def decoder_vocab_size(self):
        return self.decoder_tokenizer.vocab_size

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
    def encoder(self) -> EncoderModule:
        return self._encoder

    @encoder.setter
    def encoder(self, encoder):
        self._encoder = encoder

    @property
    def decoder(self) -> DecoderModule:
        return self._decoder

    @decoder.setter
    def decoder(self, decoder):
        self._decoder = decoder

    def setup_enc_dec_tokenizers(self, cfg: EncDecNLPModelConfig):
        if cfg.encoder_tokenizer.vocab_file is not None or cfg.decoder_tokenizer.vocab_file is not None:
            raise NotImplemented(
                f'Vocab files are currently not supported. Please use tokenizer name and model instead'
            )

        if cfg.encoder_tokenizer.tokenizer_name != 'yttm' or cfg.decoder_tokenizer.tokenizer_name != 'yttm':
            raise NotImplemented(f"Currently we only support yttm tokenizer.")
        self.encoder_tokenizer = get_tokenizer(
            tokenizer_name=cfg.encoder_tokenizer.tokenizer_name,
            tokenizer_model=self.register_artifact(
                "cfg.encoder_tokenizer.tokenizer_model", cfg.encoder_tokenizer.tokenizer_model
            ),
            bpe_dropout=cfg.encoder_tokenizer.bpe_dropout if hasattr(cfg.encoder_tokenizer, 'bpe_dropout') else 0.0,
        )
        self.decoder_tokenizer = get_tokenizer(
            tokenizer_name=cfg.decoder_tokenizer.tokenizer_name,
            tokenizer_model=self.register_artifact(
                "cfg.decoder_tokenizer.tokenizer_model", cfg.decoder_tokenizer.tokenizer_model
            ),
            bpe_dropout=cfg.decoder_tokenizer.bpe_dropout if hasattr(cfg.decoder_tokenizer, 'bpe_dropout') else 0.0,
        )
