# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

from abc import ABC

import numpy as np
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger

from nemo.collections.tts.parts.utils.helpers import (
    mel_to_audio,
    plot_alignment_to_numpy,
    plot_multipitch_to_numpy,
    plot_pitch_to_numpy,
    plot_spectrogram_to_numpy,
    tensor_to_wav,
)
from nemo.utils import logging
from nemo.utils.loggers.clearml_logger import HAVE_CLEARML_LOGGER, ClearMLLogger

HAVE_WANDB = True
try:
    import wandb
except ModuleNotFoundError:
    HAVE_WANDB = False


__all__ = ["TTSValLogger"]


SUPPORTED_LOGGERS = (
    TensorBoardLogger,
    WandbLogger,
    ClearMLLogger,
)


class TTSValLogger(ABC):
    """Class for logging val samples of TTS models."""

    def _val_log_image(self, title: str, name: str, image: np.ndarray, wandb_array: list):
        for logger in self.loggers:
            if isinstance(logger, TensorBoardLogger):
                logger.experiment.add_image(
                    name, image, self.global_step, dataformats="HWC",
                )
            elif isinstance(logger, WandbLogger) and HAVE_WANDB:
                wandb_array.append(wandb.Image(image, caption=name))
            elif isinstance(logger, ClearMLLogger) and HAVE_CLEARML_LOGGER:
                logger.clearml_task.logger.report_image(
                    image=image, series=name, title=title, iteration=self.global_step,
                )

    def _val_log_audio(self, title: str, name: str, audio: np.ndarray, wandb_array: list):
        if len(audio.shape) > 1:
            audio = mel_to_audio(
                audio,
                sr=self.sample_rate,
                n_fft=self._cfg.n_fft,
                n_mels=self._cfg.n_mel_channels,
                fmax=self._cfg.highfreq,
            )
        for logger in self.loggers:
            if isinstance(logger, TensorBoardLogger):
                logger.experiment.add_audio(
                    name, audio, self.global_step, sample_rate=self.sample_rate,
                )
            elif isinstance(logger, WandbLogger) and HAVE_WANDB:
                wandb_array.append(wandb.Audio(audio, caption=name, sample_rate=self.sample_rate))
            elif isinstance(logger, ClearMLLogger) and HAVE_CLEARML_LOGGER:
                logger.clearml_task.logger.report_media(
                    stream=tensor_to_wav(audio, self.sample_rate),
                    title=title,
                    series=name,
                    file_extension="wav",
                    iteration=self.global_step,
                )

    def _val_log_spects(
        self, wandb_logs: dict, spects: dict = None,
    ):
        if spects:
            wandb_spects = []

            for name, spect_array in spects.items():
                for i in range(len(spect_array)):
                    self._val_log_image(
                        title="spect",
                        name=f"{name}_{i}",
                        image=plot_spectrogram_to_numpy(spect_array[i]),
                        wandb_array=wandb_spects,
                    )

            if len(wandb_spects):
                wandb_logs["spects"] = wandb_spects

    def _val_log_pitches(
        self, wandb_logs: dict, pitches_target: list = None, pitches_pred: list = None,
    ):
        if isinstance(pitches_target, list) and isinstance(pitches_pred, list):
            wandb_pitches = []

            if len(pitches_target) == len(pitches_pred) and len(pitches_pred) > 0:
                for i, (pitch_target, pitch_pred) in enumerate(zip(pitches_target, pitches_pred)):
                    self._val_log_image(
                        title="pitch",
                        name=f"val_pitch_{i}",
                        image=plot_multipitch_to_numpy(pitch_target, pitch_pred),
                        wandb_array=wandb_pitches,
                    )

            else:
                if len(pitches_target) > 0:
                    for i, pitch_target in enumerate(pitches_target):
                        self._val_log_image(
                            title="pitch",
                            name=f"val_pitch_target_{i}",
                            image=plot_pitch_to_numpy(pitch_target),
                            wandb_array=wandb_pitches,
                        )

                if len(pitches_pred) > 0:
                    for i, pitch_pred in enumerate(pitches_pred):
                        self._val_log_image(
                            title="pitch",
                            name=f"val_pitch_pred_{i}",
                            image=plot_pitch_to_numpy(pitch_pred),
                            wandb_array=wandb_pitches,
                        )

            if len(wandb_pitches):
                wandb_logs["pitches"] = wandb_pitches

    def _val_log_audios(
        self, wandb_logs: dict, audios: dict = None,
    ):
        if audios:
            wandb_audios = []

            for name, audio_array in audios.items():
                for i in range(len(audio_array)):
                    self._val_log_audio(
                        title="audio", name=f"{name}_{i}", audio=audio_array[i], wandb_array=wandb_audios,
                    )

            if len(wandb_audios):
                wandb_logs["audios"] = wandb_audios

    def _val_log_alignments(
        self, wandb_logs: dict, alignments: dict = None,
    ):
        if alignments:
            wandb_alignments = []

            for name, alignment_array in alignments.items():
                for i in range(len(alignment_array)):
                    self._val_log_image(
                        title="alignment",
                        name=f"{name}_{i}",
                        image=plot_alignment_to_numpy(alignment_array[i]),
                        wandb_array=wandb_alignments,
                    )

            if len(wandb_alignments):
                wandb_logs["alignments"] = wandb_alignments

    def val_check_loggers(self):
        for logger in self.loggers:
            if not isinstance(logger, SUPPORTED_LOGGERS):
                logging.warning(f"Logger {logger} is not supported for logging validation samples!")

    def val_log(
        self,
        spects: dict = None,
        pitches_target: list = None,
        pitches_pred: list = None,
        audios: dict = None,
        alignments: dict = None,
    ):
        wandb_logs = {}

        self._val_log_spects(wandb_logs, spects)
        self._val_log_pitches(wandb_logs, pitches_target, pitches_pred)
        self._val_log_audios(wandb_logs, audios)
        self._val_log_alignments(wandb_logs, alignments)

        for logger in self.loggers:
            if isinstance(logger, WandbLogger):
                logger.experiment.log(wandb_logs)
                break
