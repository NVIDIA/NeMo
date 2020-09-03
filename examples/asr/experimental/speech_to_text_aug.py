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

import copy
import os
import random
import subprocess
from tempfile import NamedTemporaryFile

import numpy as np
import pytorch_lightning as pl
import soundfile as sf
import torch
from omegaconf import DictConfig, OmegaConf

from nemo.collections.asr.models import EncDecCTCModel
from nemo.collections.asr.parts import perturb
from nemo.collections.asr.parts.segment import AudioSegment
from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.exp_manager import exp_manager

class RirAndNoisePerturbation(perturb.Perturbation):
    def __init__(
        self,
        rir_manifest_path=None,
        rir_prob=0.5,
        noise_manifest_paths=None,
        min_snr_db=0,
        max_snr_db=50,
        rir_tar_filepaths=None,
        rir_shuffle_n=100,
        noise_tar_filepaths=None,
        apply_noise_rir=False,
        orig_sampling_rate=None,
        max_additions=5,
        max_duration=2.0,
        bg_noise_manifest_paths=None,
        bg_min_snr_db=10,
        bg_max_snr_db=50,
        bg_noise_tar_filepaths=None,
        bg_orig_sampling_rate=None,
    ):
        """
        RIR augmentation with additive foreground and background noise.
        In this implementation audio data is augmented by first convolving the audio with a Room Impulse Response
        and then adding foreground noise and background noise at various SNRs. RIR, foreground and background noises
        should either be supplied with a manifest file or as tarred audio files (faster).

        Different sets of noise audio files based on the original sampling rate of the noise. This is useful while
        training a mixed sample rate model. For example, when training a mixed model with 8 kHz and 16 kHz audio with a
        target sampling rate of 16 kHz, one would want to augment 8 kHz data with 8 kHz noise rather than 16 kHz noise.

        Args:
            rir_manifest_path: manifest file for RIRs
            rir_tar_filepaths: tar files, if RIR audio files are tarred
            rir_prob: probability of applying a RIR
            noise_manifest_paths: foreground noise manifest path
            min_snr_db: min SNR for foreground noise
            max_snr_db: max SNR for background noise,
            noise_tar_filepaths: tar files, if noise files are tarred
            apply_noise_rir: whether to convolve foreground noise with a a random RIR
            orig_sampling_rate: original sampling rate of foreground noise audio
            max_additions: max number of times foreground noise is added to an utterance,
            max_duration: max duration of foreground noise
            bg_noise_manifest_paths: background noise manifest path
            bg_min_snr_db: min SNR for background noise
            bg_max_snr_db: max SNR for background noise
            bg_noise_tar_filepaths: tar files, if noise files are tarred
            bg_orig_sampling_rate: original sampling rate of background noise audio

        """
        logging.info("Called Rir aug init")
        self._rir_prob = rir_prob
        self._rng = random.Random()
        self._rir_perturber = perturb.ImpulsePerturbation(
            manifest_path=rir_manifest_path,
            audio_tar_filepaths=rir_tar_filepaths,
            shuffle_n=rir_shuffle_n,
            shift_impulse=True
        )
        self._fg_noise_perturbers = {}
        self._bg_noise_perturbers = {}
        if noise_manifest_paths:
            for i in range(len(noise_manifest_paths)):
                if orig_sampling_rate is None:
                    orig_sr = 16000
                else:
                    orig_sr = orig_sampling_rate[i]
                self._fg_noise_perturbers[orig_sr] = perturb.NoisePerturbation(
                    manifest_path=noise_manifest_paths[i],
                    min_snr_db=min_snr_db[i],
                    max_snr_db=max_snr_db[i],
                    audio_tar_filepaths=noise_tar_filepaths[i],
                    orig_sr=orig_sr,
                )
        self._max_additions = max_additions
        self._max_duration = max_duration
        if bg_noise_manifest_paths:
            for i in range(len(bg_noise_manifest_paths)):
                if bg_orig_sampling_rate is None:
                    orig_sr = 16000
                else:
                    orig_sr = bg_orig_sampling_rate[i]
                self._bg_noise_perturbers[orig_sr] = perturb.NoisePerturbation(
                    manifest_path=bg_noise_manifest_paths[i],
                    min_snr_db=bg_min_snr_db[i],
                    max_snr_db=bg_max_snr_db[i],
                    audio_tar_filepaths=bg_noise_tar_filepaths[i],
                    orig_sr=orig_sr,
                )

        self._apply_noise_rir = apply_noise_rir

    def perturb(self, data):
        prob = self._rng.uniform(0.0, 1.0)

        if prob < self._rir_prob:
            self._rir_perturber.perturb(data)

        orig_sr = data.orig_sr
        if orig_sr not in self._fg_noise_perturbers:
            orig_sr = max(self._fg_noise_perturbers.keys())
        fg_perturber = self._fg_noise_perturbers[orig_sr]

        orig_sr = data.orig_sr
        if orig_sr not in self._bg_noise_perturbers:
            orig_sr = max(self._bg_noise_perturbers.keys())
        bg_perturber = self._bg_noise_perturbers[orig_sr]

        data_rms = data.rms_db
        noise = fg_perturber.get_one_noise_sample(data.sample_rate)
        if self._apply_noise_rir:
            self._rir_perturber.perturb(noise)
        fg_perturber.perturb_with_foreground_noise(
            data, noise, data_rms=data_rms, max_noise_dur=self._max_duration, max_additions=self._max_additions
        )
        noise = bg_perturber.get_one_noise_sample(data.sample_rate)
        bg_perturber.perturb_with_input_noise(data, noise, data_rms=data_rms)


class TranscodePerturbation(perturb.Perturbation):
    def __init__(self, rng=None):
        """
        Audio codec augmentation. This implementation uses sox to transcode audio with low rate audio codecs,
        so users need to make sure that the installed sox version supports the codecs used here (G711 and amr-nb).

        """
        self._rng = np.random.RandomState() if rng is None else rng
        self._codecs = ["g711", "amr-nb"]

    def perturb(self, data):
        att_factor = 0.8
        max_level = np.max(np.abs(data._samples))
        norm_factor = att_factor / max_level
        norm_samples = norm_factor * data._samples
        orig_f = NamedTemporaryFile(suffix=".wav")
        sf.write(orig_f.name, norm_samples.transpose(), 16000)

        codec_ind = random.randint(0, len(self._codecs) - 1)
        if self._codecs[codec_ind] == "amr-nb":
            transcoded_f = NamedTemporaryFile(suffix="_amr.wav")
            rates = list(range(0, 8))
            rate = rates[random.randint(0, len(rates) - 1)]
            _ = subprocess.check_output(
                f"sox {orig_f.name} -V0 -C {rate} -t amr-nb - | sox -t amr-nb - -V0 -b 16 -r 16000 {transcoded_f.name}",
                shell=True,
            )
        elif self._codecs[codec_ind] == "g711":
            transcoded_f = NamedTemporaryFile(suffix="_g711.wav")
            _ = subprocess.check_output(
                f"sox {orig_f.name} -V0  -r 8000 -c 1 -e a-law {transcoded_f.name}", shell=True
            )

        new_data = AudioSegment.from_file(transcoded_f.name, target_sr=16000)
        data._samples = new_data._samples[0 : data._samples.shape[0]]
        return


def get_augmentor_dict():
    audio_augmentations = dict(
        rir_noise_aug=dict(
            prob=0.5,
            rir_manifest_path="/data/datasets/freesound_20s/rir_noises_tarred/rir_tarred/tarred/tarred_audio_manifest.json",
            rir_tar_filepaths="/data/datasets/freesound_20s/rir_noises_tarred/rir_tarred/tarred/audio_{0..1}.tar",
            rir_prob=0.5,
            noise_manifest_paths=[
                "/data/datasets/freesound_20s/rir_noises_tarred/noises_20s_tarred/tarred_audio_manifest.json",
                "/data/datasets/freesound_20s/rir_noises_tarred/noises_20s_tarred_8k/tarred_audio_manifest.json",
            ],
            noise_tar_filepaths=[
                "/data/datasets/freesound_20s/rir_noises_tarred/noises_20s_tarred/audio_{0..63}.tar",
                "/data/datasets/freesound_20s/rir_noises_tarred/noises_20s_tarred_8k/audio_{0..63}.tar",
            ],
            min_snr_db=[0, 0],
            max_snr_db=[30, 30],
            orig_sampling_rate=[16000, 8000],
            bg_noise_manifest_paths=[
                "/data/datasets/freesound_20s/rir_noises_tarred/noises_20s_tarred/tarred_audio_manifest.json",
                "/data/datasets/freesound_20s/rir_noises_tarred/noises_20s_tarred_8k/tarred_audio_manifest.json",
            ],
            bg_noise_tar_filepaths=[
                "/data/datasets/freesound_20s/rir_noises_tarred/noises_20s_tarred/audio_{0..63}.tar",
                "/data/datasets/freesound_20s/rir_noises_tarred/noises_20s_tarred_8k/audio_{0..63}.tar",
            ],
            bg_min_snr_db=[10, 10],
            bg_max_snr_db=[40, 40],
            bg_orig_sampling_rate=[16000, 8000],
        ),
        transcode_aug=dict(prob=0.5,),
    )
    return audio_augmentations



@hydra_runner(config_path="../conf", config_name="quartznet_15x5")
def main(cfg):
    perturb.register_perturbation(name='rir_noise_aug', perturbation=RirAndNoisePerturbation)
    perturb.register_perturbation(name='transcode_aug', perturbation=TranscodePerturbation)

    OmegaConf.set_struct(cfg, False)
    if cfg.trainer.gpus > 0 and "num_workers" not in cfg.model.train_ds:
        cfg.model.train_ds.num_workers = os.cpu_count()
    cfg.model.train_ds.augmentor = OmegaConf.create(get_augmentor_dict())
    OmegaConf.set_struct(cfg, True)

    trainer = pl.Trainer(**cfg.trainer)
    exp_manager(trainer, cfg.get("exp_manager", None))
    asr_model = EncDecCTCModel(cfg=cfg.model, trainer=trainer)
    trainer.fit(asr_model)


if __name__ == '__main__':
    main()  # noqa pylint: disable=no-value-for-parameter
