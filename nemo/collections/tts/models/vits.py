from dataclasses import dataclass
from itertools import chain
from typing import Any, Dict

import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import MISSING, DictConfig, OmegaConf
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import LoggerCollection, TensorBoardLogger
import math
import torch
from torch import nn
from torch.nn import functional as F

import commons
import modules
import attentions
import monotonic_align

from torch.nn import Conv1d, ConvTranspose1d, AvgPool1d, Conv2d
from torch.nn.utils import weight_norm, remove_weight_norm, spectral_norm
from commons import init_weights, get_padding

from nemo.collections.asr.data.audio_to_text import FastPitchDataset
from nemo.collections.common.parts.preprocessing import parsers
from nemo.collections.tts.helpers.helpers import plot_spectrogram_to_numpy, regulate_len
from nemo.collections.tts.models.base import TextToWaveform
from nemo.core.classes.common import PretrainedModelInfo, typecheck
from nemo.core.neural_types.elements import (
    MelSpectrogramType,
    RegressionValuesType,
    TokenDurationType,
    TokenIndex,
    TokenLogDurationType,
)
from nemo.core.neural_types.neural_type import NeuralType
from nemo.core.optim.lr_scheduler import NoamAnnealing
from nemo.utils import logging
from nemo.collections.tts.models.base import TextToWaveform
from nemo.collections.tts.losses.vits_losses import DiscriminatorLoss, FeatureLoss, GeneratorLoss, KlLoss
import nemo.collections.tts.modules.vits_modules as modules


@dataclass
class VitsConfig:
    parser: Dict[Any, Any] = MISSING
    preprocessor: Dict[Any, Any] = MISSING
    input_fft: Dict[Any, Any] = MISSING
    output_fft: Dict[Any, Any] = MISSING
    duration_predictor: Dict[Any, Any] = MISSING
    pitch_predictor: Dict[Any, Any] = MISSING


class Vits(TextToWaveform):
    def __init__(self, cfg: DictConfig, trainer: 'Trainer' = None):
        if isinstance(cfg, dict):
            cfg = OmegaConf.create(cfg)

        self._parser = parsers.make_parser(
            labels=cfg.labels,
            name='en',
            unk_id=-1,
            blank_id=-1,
            do_normalize=True,
            abbreviation_version="fastpitch",
            make_table=False,
        )

        super().__init__(cfg=cfg, trainer=trainer)

        schema = OmegaConf.structured(VitsConfig)
        # ModelPT ensures that cfg is a DictConfig, but do this second check in case ModelPT changes
        if isinstance(cfg, dict):
            cfg = OmegaConf.create(cfg)
        elif not isinstance(cfg, DictConfig):
            raise ValueError(f"cfg was type: {type(cfg)}. Expected either a dict or a DictConfig")
        # Ensure passed cfg is compliant with schema
        OmegaConf.merge(cfg, schema)

        self.preprocessor = instantiate(cfg.preprocessor)
        self.melspec_fn = instantiate(cfg.preprocessor, highfreq=None, use_grads=True)

        self.encoder = instantiate(cfg.input_fft)
        self.duration_predictor = instantiate(cfg.duration_predictor)
        self.pitch_predictor = instantiate(cfg.pitch_predictor)

        self.generator = instantiate(cfg.generator)
        self.multiperioddisc = MultiPeriodDiscriminator()
        self.feat_matching_loss = FeatureLoss()
        self.disc_loss = DiscriminatorLoss()
        self.gen_loss = GeneratorLoss()

        self.max_token_duration = cfg.max_token_duration

        self.pitch_emb = torch.nn.Conv1d(
            1,
            cfg.symbols_embedding_dim,
            kernel_size=cfg.pitch_embedding_kernel_size,
            padding=int((cfg.pitch_embedding_kernel_size - 1) / 2),
        )

        # Store values precomputed from training data for convenience
        self.register_buffer('pitch_mean', torch.zeros(1))
        self.register_buffer('pitch_std', torch.zeros(1))

        self.mel_loss_coeff = cfg.mel_loss_coeff

        self.log_train_images = False
        self.logged_real_samples = False
        self._tb_logger = None
        self.hann_window = None
        self.splice_length = cfg.splice_length
        self.sample_rate = cfg.sample_rate
        self.hop_size = cfg.hop_size

    def parse(self, str_input: str) -> torch.tensor:
        # TODO: Implement
        pass

    def configure_optimizers(self):
        # TODO: Implement
        pass

    @staticmethod
    def _loader(cfg):
        try:
            _ = cfg.dataset.manifest_filepath
        except omegaconf.errors.MissingMandatoryValue:
            logging.warning("manifest_filepath was skipped. No dataset for this model.")
            return None

        dataset = instantiate(cfg.dataset)
        return torch.utils.data.DataLoader(  # noqa
            dataset=dataset, collate_fn=dataset.collate_fn, **cfg.dataloader_params,
        )

    def setup_training_data(self, cfg):
        self._train_dl = self._loader(cfg)

    def setup_validation_data(self, cfg):
        self._validation_dl = self._loader(cfg)

    def setup_test_data(self, cfg):
        """Omitted."""
        pass

    def list_available_models(cls) -> 'List[PretrainedModelInfo]':
        list_of_models = []
        # TODO: List available models??
        return list_of_models

    def convert_text_to_waveform(self, *, tokens):
        #  TODO: Convert text to waveforms
        pass

