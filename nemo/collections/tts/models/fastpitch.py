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
from typing import Any, Dict

import torch
from hydra.utils import instantiate
from omegaconf import MISSING, DictConfig, OmegaConf, open_dict
from pytorch_lightning import Trainer

from nemo.collections.asr.parts import parsers
from nemo.collections.tts.losses.aligner_loss import BinLoss, ForwardSumLoss
from nemo.collections.tts.losses.fastpitchloss import MelLoss, PitchLoss, DurationLoss
from nemo.collections.tts.models.base import SpectrogramGenerator
from nemo.collections.tts.modules.fastpitch import FastPitchModule
from nemo.core.classes.common import PretrainedModelInfo, typecheck
from nemo.core.neural_types.elements import MelSpectrogramType, RegressionValuesType, TokenDurationType, TokenIndex
from nemo.core.neural_types.neural_type import NeuralType
from nemo.utils import logging


@dataclass
class FastPitchConfig:
    parser: Dict[Any, Any] = MISSING
    preprocessor: Dict[Any, Any] = MISSING
    input_fft: Dict[Any, Any] = MISSING
    output_fft: Dict[Any, Any] = MISSING
    duration_predictor: Dict[Any, Any] = MISSING
    pitch_predictor: Dict[Any, Any] = MISSING


def average_pitch(pitch, durs):
    durs_cums_ends = torch.cumsum(durs, dim=1).long()
    durs_cums_starts = torch.functional.pad(durs_cums_ends[:, :-1], (1, 0))
    pitch_nonzero_cums = torch.functional.pad(torch.cumsum(pitch != 0.0, dim=2), (1, 0))
    pitch_cums = torch.functional.pad(torch.cumsum(pitch, dim=2), (1, 0))

    bs, l = durs_cums_ends.size()
    n_formants = pitch.size(1)
    dcs = durs_cums_starts[:, None, :].expand(bs, n_formants, l)
    dce = durs_cums_ends[:, None, :].expand(bs, n_formants, l)

    pitch_sums = (torch.gather(pitch_cums, 2, dce) - torch.gather(pitch_cums, 2, dcs)).float()
    pitch_nelems = (torch.gather(pitch_nonzero_cums, 2, dce) - torch.gather(pitch_nonzero_cums, 2, dcs)).float()

    pitch_avg = torch.where(pitch_nelems == 0.0, pitch_nelems, pitch_sums / pitch_nelems)
    return pitch_avg


class FastPitchModel(SpectrogramGenerator):
    """FastPitch Model that is used to generate mel spectrograms from text"""

    def __init__(self, cfg: DictConfig, trainer: Trainer = None):
        if isinstance(cfg, dict):
            cfg = OmegaConf.create(cfg)

        self.learn_aligntment = False
        if "learn_alignment" in cfg:
            self.learn_aligntment = cfg.learn_alignment
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

        self.aligner = None
        self.mel_loss = MelLoss()
        self.pitch_loss = PitchLoss()
        self.duration_loss = DurationLoss()
        if self.learn_aligntment:
            self.aligner = instantiate(self._cfg.alignment_module)
            self.forward_sum_loss = ForwardSumLoss()
            self.bin_loss = BinLoss()

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
            self.aligner,
            cfg.n_speakers,
            cfg.symbols_embedding_dim,
            cfg.pitch_embedding_kernel_size,
            cfg.n_mel_channels,
        )

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
        attn_prior, durs, speakers = None, None, None
        if self.learn_aligntment:
            audio, audio_lens, text, text_lens, attn_prior, pitch = batch
        else:
            audio, audio_lens, text, text_lens, durs, pitch, speakers = batch
        mels, spec_len = self.preprocessor(input_signal=audio, length=audio_lens)

        mels_pred, _, log_durs_pred, pitch_pred, attn_soft, attn_logprob, attn_hard, attn_hard_dur = self(
            text=text,
            durs=durs,
            pitch=pitch,
            speaker=speakers,
            pace=1.0,
            spec=mels,
            attn_prior=attn_prior,
            mel_lens=spec_len,
            input_lens=text_lens,
        )
        if durs is None:
            durs = attn_hard_dur

        mel_loss = self.mel_loss(spect_predicted=mels_pred, spect_tgt=mels)
        dur_loss = self.duration_loss(log_durs_predicted=log_durs_pred, durs_tgt=durs, len=text_lens)
        loss = mel_loss + dur_loss
        if self.learn_aligntment:
            ctc_loss = self.forward_sum_loss(attn_logprob, text_lens, spec_len)
            bin_loss = self.bin_loss(attn_hard, attn_soft)
            loss += ctc_loss + bin_loss
            pitch = average_pitch(pitch, attn_hard_dur)

        pitch_loss = self.pitch_loss(pitch_predicted=pitch_pred, pitch_tgt=pitch, len=text_lens)
        loss += pitch_loss

        return loss

    def validation_step(self, batch, batch_idx):
        durs, speakers = None, None
        if self.learn_aligntment:
            audio, audio_lens, text, text_lens, _, pitch = batch
        else:
            audio, audio_lens, text, text_lens, durs, pitch, speakers = batch
        mels, _ = self.preprocessor(input_signal=audio, length=audio_lens)

        # Calculate val loss on ground truth durations to better align L2 loss in time
        mels_pred, _, log_durs_pred, pitch_pred, _, _, _, attn_hard_dur = self(
            text=text, durs=durs, pitch=None, speaker=speakers, pace=1.0
        )

        mel_loss = self.mel_loss(spect_predicted=mels_pred, spect_tgt=mels)
        dur_loss = self.duration_loss(log_durs_predicted=log_durs_pred, durs_tgt=durs, len=text_lens)
        loss = mel_loss + dur_loss
        if self.learn_aligntment:
            pitch = average_pitch(pitch, attn_hard_dur)

        pitch_loss = self.pitch_loss(pitch_predicted=pitch_pred, pitch_tgt=pitch, len=text_lens)
        loss += pitch_loss

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

    def __setup_dataloader_from_config(self, cfg, shuffle_should_be: bool = True, name: str = "train"):
        if "dataset" not in cfg or not isinstance(cfg.dataset, DictConfig):
            raise ValueError(f"No dataset for {name}")
        if "dataloader_params" not in cfg or not isinstance(cfg.dataloader_params, DictConfig):
            raise ValueError(f"No dataloder_params for {name}")
        if shuffle_should_be:
            if 'shuffle' not in cfg.dataloader_params:
                logging.warning(
                    f"Shuffle should be set to True for {self}'s {name} dataloader but was not found in its "
                    "config. Manually setting to True"
                )
                with open_dict(cfg.dataloader_params):
                    cfg.dataloader_params.shuffle = True
            elif not cfg.dataloader_params.shuffle:
                logging.error(f"The {name} dataloader for {self} has shuffle set to False!!!")
        elif not shuffle_should_be and cfg.dataloader_params.shuffle:
            logging.error(f"The {name} dataloader for {self} has shuffle set to True!!!")

        kwargs_dict = {}
        if cfg.dataset._target_ == "nemo.collections.asr.data.audio_to_text.FastPitchDataset":
            kwargs_dict["parser"] = self.parser
        dataset = instantiate(cfg.dataset, **kwargs_dict)
        return torch.utils.data.DataLoader(dataset, collate_fn=dataset.collate_fn, **cfg.dataloader_params)

    def setup_training_data(self, cfg):
        self._train_dl = self.__setup_dataloader_from_config(cfg)

    def setup_validation_data(self, cfg):
        self._validation_dl = self.__setup_dataloader_from_config(cfg, shuffle_should_be=False, name="val")

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
        # model = PretrainedModelInfo(
        #     pretrained_model_name="",
        #     location="",
        #     description="",
        #     class_=cls,
        # )
        # list_of_models.append(model)

        return list_of_models

