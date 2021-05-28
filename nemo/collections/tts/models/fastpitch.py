# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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
from typing import Any, Dict, List

import torch
from hydra.utils import instantiate
from omegaconf import MISSING, DictConfig, OmegaConf
from pytorch_lightning import Trainer

from nemo.collections.asr.data.audio_to_text import FastPitchDataset
from nemo.collections.asr.parts import parsers
from nemo.collections.tts.losses.fastpitchloss import FastPitchLoss
from nemo.collections.tts.models.base import SpectrogramGenerator
from nemo.collections.tts.modules.fastpitch import FastPitchModule
from nemo.core.classes.common import PretrainedModelInfo, typecheck
from nemo.core.neural_types.elements import MelSpectrogramType, RegressionValuesType, TokenDurationType, TokenIndex
from nemo.core.neural_types.neural_type import NeuralType


@dataclass
class FastPitchConfig:
    parser: Dict[Any, Any] = MISSING
    preprocessor: Dict[Any, Any] = MISSING
    input_fft: Dict[Any, Any] = MISSING
    output_fft: Dict[Any, Any] = MISSING
    duration_predictor: Dict[Any, Any] = MISSING
    pitch_predictor: Dict[Any, Any] = MISSING


class FastPitchModel(SpectrogramGenerator):
    """FastPitch Model that is used to generate mel spectrograms from text"""

    def __init__(self, cfg: DictConfig, trainer: Trainer = None):
        if isinstance(cfg, dict):
            cfg = OmegaConf.create(cfg)

        self._parser = None
        super().__init__(cfg=cfg, trainer=trainer)

        schema = OmegaConf.structured(FastPitchConfig)
        # ModelPT ensures that cfg is a DictConfig, but do this second check in case ModelPT changes
        if isinstance(cfg, dict):
            cfg = OmegaConf.create(cfg)
        elif not isinstance(cfg, DictConfig):
            raise ValueError(f"cfg was type: {type(cfg)}. Expected either a dict or a DictConfig")
        # Ensure passed cfg is compliant with schema
        OmegaConf.merge(cfg, schema)

        self.preprocessor = instantiate(self._cfg.preprocessor)

        input_fft = instantiate(self._cfg.input_fft)
        output_fft = instantiate(self._cfg.output_fft)
        duration_predictor = instantiate(self._cfg.duration_predictor)
        pitch_predictor = instantiate(self._cfg.pitch_predictor)

        self.fastpitch = FastPitchModule(
            input_fft,
            output_fft,
            duration_predictor,
            pitch_predictor,
            cfg.n_speakers,
            cfg.symbols_embedding_dim,
            cfg.pitch_embedding_kernel_size,
            cfg.n_mel_channels,
        )
        self.loss = FastPitchLoss()

    @property
    def parser(self):
        if self._parser is not None:
            return self._parser

        self._parser = parsers.make_parser(
            labels=self._cfg.labels,
            name='en',
            unk_id=-1,
            blank_id=-1,
            do_normalize=True,
            abbreviation_version="fastpitch",
            make_table=False,
        )
        return self._parser

    def parse(self, str_input: str) -> torch.tensor:
        if str_input[-1] not in [".", "!", "?"]:
            str_input = str_input + "."

        tokens = self.parser(str_input)

        x = torch.tensor(tokens).unsqueeze_(0).long().to(self.device)
        return x

    @typecheck(
        input_types={
            "text": NeuralType(('B', 'T'), TokenIndex()),
            "durs": NeuralType(('B', 'T'), TokenDurationType(), optional=True),
            "pitch": NeuralType(('B', 'T'), RegressionValuesType(), optional=True),
            "speaker": NeuralType(optional=True),  # NeuralType(('B'), IntType(), optional=True),
            "pace": NeuralType(optional=True),
        }
    )
    def forward(self, *, text, durs=None, pitch=None, speaker=0, pace=1.0):
        return self.fastpitch(text=text, durs=durs, pitch=pitch, speaker=speaker, pace=pace)

    @typecheck(output_types={"spect": NeuralType(('B', 'C', 'T'), MelSpectrogramType())})
    def generate_spectrogram(self, tokens: 'torch.tensor', speaker: int = 0, pace: float = 1.0) -> torch.tensor:
        self.eval()
        spect, *_ = self(text=tokens, durs=None, pitch=None, speaker=speaker, pace=pace)
        return spect.transpose(1, 2)

    def training_step(self, batch, batch_idx):
        audio, audio_lens, text, text_lens, durs, pitch, speakers = batch
        mels, mel_lens = self.preprocessor(input_signal=audio, length=audio_lens)

        mels_pred, mel_lens, _, _, log_durs_pred, pitch_pred = self(
            text=text, durs=durs, pitch=pitch, speaker=speakers, pace=1.0
        )

        loss, mel_loss, dur_loss, pitch_loss = self.loss(
            spect_predicted=mels_pred,
            log_durs_predicted=log_durs_pred,
            pitch_predicted=pitch_pred,
            spect_tgt=mels,
            durs_tgt=durs,
            dur_lens=text_lens,
            pitch_tgt=pitch,
        )

        losses = {
            "mel_loss": mel_loss,
            "dur_loss": dur_loss,
            "pitch_loss": pitch_loss,
        }
        all_losses = {"loss": loss, **losses}
        return {**all_losses, "progress_bar": losses, "log": all_losses}

    def validation_step(self, batch, batch_idx):
        audio, audio_lens, text, text_lens, durs, pitch, speakers = batch
        mels, mel_lens = self.preprocessor(input_signal=audio, length=audio_lens)

        # Calculate val loss on ground truth durations to better align L2 loss in time
        mels_pred, mel_lens, _, _, log_durs_pred, pitch_pred = self(
            text=text, durs=durs, pitch=None, speaker=speakers, pace=1.0
        )

        loss, mel_loss, dur_loss, pitch_loss = self.loss(
            spect_predicted=mels_pred,
            log_durs_predicted=log_durs_pred,
            pitch_predicted=pitch_pred,
            spect_tgt=mels,
            durs_tgt=durs,
            dur_lens=text_lens,
            pitch_tgt=pitch,
        )

        ret = {
            "loss": loss,
            "mel_loss": mel_loss,
            "dur_loss": dur_loss,
            "pitch_loss": pitch_loss,
        }
        return {**ret, "progress_bar": ret}

    def validation_epoch_end(self, outputs):
        collect = lambda key: torch.stack([x[key] for x in outputs]).mean()
        tb_logs = {
            'val_loss': collect('loss'),
            'val_mel_loss': collect('mel_loss'),
            'val_dur_loss': collect('dur_loss'),
            'val_pitch_loss': collect('pitch_loss'),
        }
        return {'val_loss': tb_logs['val_loss'], 'log': tb_logs}

    def _loader(self, cfg):
        dataset = FastPitchDataset(
            manifest_filepath=cfg['manifest_filepath'],
            parser=self.parser,
            sample_rate=cfg['sample_rate'],
            int_values=cfg.get('int_values', False),
            max_duration=cfg.get('max_duration', None),
            min_duration=cfg.get('min_duration', None),
            max_utts=cfg.get('max_utts', 0),
            trim=cfg.get('trim_silence', True),
        )

        return torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=cfg['batch_size'],
            collate_fn=dataset.collate_fn,
            drop_last=cfg.get('drop_last', True),
            shuffle=cfg['shuffle'],
            num_workers=cfg.get('num_workers', 16),
        )

    def setup_training_data(self, cfg):
        self._train_dl = self._loader(cfg)

    def setup_validation_data(self, cfg):
        self._validation_dl = self._loader(cfg)

    def setup_test_data(self, cfg):
        """Omitted."""
        pass

    @classmethod
    def list_available_models(cls) -> 'List[PretrainedModelInfo]':
        """
        This method returns a list of pre-trained model which can be instantiated directly from NVIDIA's NGC cloud.
        Returns:
            List of available pre-trained models.
        """
        list_of_models = []
        model = PretrainedModelInfo(
            pretrained_model_name="tts_en_fastpitch",
            location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/tts_en_fastpitch/versions/1.0.0/files/tts_en_fastpitch.nemo",
            description="This model is trained on LJSpeech sampled at 22050Hz with and can be used to generate female English voices with an American accent.",
            class_=cls,
        )
        list_of_models.append(model)

        return list_of_models
