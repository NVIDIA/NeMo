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
        noise_manifest_paths=None,
        min_snr_db=10,
        max_snr_db=50,
        rir_prob=0.5,
        max_gain_db=300.0,
        rng=None,
        rir_tar_filepaths=None,
        rir_shuffle_n=100,
        noise_tar_filepaths=None,
        noise_shuffle_n=None,
        apply_noise_rir=False,
        max_frequency=None,
        max_additions=5,
        max_duration=2.0,
        bg_noise_manifest_paths=None,
        bg_min_snr_db=10,
        bg_max_snr_db=50,
        bg_max_gain_db=300.0,
        bg_noise_tar_filepaths=None,
        bg_noise_shuffle_n=None,
        bg_max_frequency=None,
    ):
        logging.info("Called init")
        self._rir_prob = rir_prob
        self._rng = random.Random() if rng is None else rng
        self._rir_perturber = perturb.ImpulsePerturbation(
            manifest_path=rir_manifest_path, rng=rng, audio_tar_filepaths=rir_tar_filepaths, shuffle_n=rir_shuffle_n
        )
        self._fg_noise_perturbers = {}
        self._bg_noise_perturbers = {}
        if noise_manifest_paths:
            for i in range(len(noise_manifest_paths)):
                if noise_shuffle_n is None:
                    shuffle_n = 100
                else:
                    shuffle_n = noise_shuffle_n[i]
                if max_frequency is None:
                    max_freq = 16000
                else:
                    max_freq = max_frequency[i]
                self._fg_noise_perturbers[max_freq] = perturb.NoisePerturbation(
                    manifest_path=noise_manifest_paths[i],
                    min_snr_db=min_snr_db[i],
                    max_snr_db=max_snr_db[i],
                    max_gain_db=max_gain_db[i],
                    rng=rng,
                    audio_tar_filepaths=noise_tar_filepaths[i],
                    shuffle_n=shuffle_n,
                    max_freq=max_freq,
                )
        self._max_additions = max_additions
        self._max_duration = max_duration
        if bg_noise_manifest_paths:
            for i in range(len(bg_noise_manifest_paths)):
                if noise_shuffle_n is None:
                    shuffle_n = 100
                else:
                    shuffle_n = noise_shuffle_n[i]
                if bg_max_frequency is None:
                    max_freq = 16000
                else:
                    max_freq = bg_max_frequency[i]
                self._bg_noise_perturbers[max_freq] = perturb.NoisePerturbation(
                    manifest_path=bg_noise_manifest_paths[i],
                    min_snr_db=bg_min_snr_db[i],
                    max_snr_db=bg_max_snr_db[i],
                    max_gain_db=bg_max_gain_db[i],
                    rng=rng,
                    audio_tar_filepaths=bg_noise_tar_filepaths[i],
                    shuffle_n=shuffle_n,
                    max_freq=max_freq,
                )

        # self._fg_noise_perturber = NoisePerturbation(manifest_path=noise_manifest_path, min_snr_db=min_snr_db,
        #                                             max_snr_db=max_snr_db, max_gain_db=max_gain_db, rng=rng,
        #                                             audio_tar_filepaths=noise_tar_filepaths, shuffle_n=noise_shuffle_n,
        #                                             max_freq=16000)
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
        fg_perturber.perturb_with_point_noise(
            data, noise, data_rms=data_rms, max_noise_dur=self._max_duration, max_additions=self._max_additions
        )
        noise = bg_perturber.get_one_noise_sample(data.sample_rate)
        bg_perturber.perturb_with_input_noise(data, noise, data_rms=data_rms)


class TranscodePerturbation(perturb.Perturbation):
    def __init__(self, rng=None):
        self._rng = np.random.RandomState() if rng is None else rng
        self._codecs = ["g711", "amr-nb"]

    def perturb(self, data):
        orig_f = NamedTemporaryFile(suffix=".wav")
        sf.write(orig_f.name, data._samples.transpose(), 16000)

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
                "/data/datasets/freesound_20s/rir_noises_tarred/noises_20s_tarred/tarred_audio_manifest.json",
            ],
            noise_tar_filepaths=[
                "/data/datasets/freesound_20s/rir_noises_tarred/noises_20s_tarred/audio_{0..63}.tar",
                "/data/datasets/freesound_20s/rir_noises_tarred/noises_20s_tarred/audio_{0..63}.tar",
            ],
            min_snr_db=[0, 0],
            max_snr_db=[30, 30],
            max_gain_db=[300.0, 300],
            max_frequency=[16000, 8000],
            bg_noise_manifest_paths=[
                "/data/datasets/freesound_20s/rir_noises_tarred/noises_20s_tarred/tarred_audio_manifest.json",
                "/data/datasets/freesound_20s/rir_noises_tarred/noises_20s_tarred/tarred_audio_manifest.json",
            ],
            bg_noise_tar_filepaths=[
                "/data/datasets/freesound_20s/rir_noises_tarred/noises_20s_tarred/audio_{0..63}.tar",
                "/data/datasets/freesound_20s/rir_noises_tarred/noises_20s_tarred/audio_{0..63}.tar",
            ],
            bg_min_snr_db=[10, 10],
            bg_max_snr_db=[40, 40],
            bg_max_gain_db=[300.0, 300],
            bg_max_frequency=[16000, 8000],
        ),
        transcode_aug=dict(prob=0.5,),
    )
    return audio_augmentations


"""
Basic run (on CPU for 50 epochs):
    python examples/asr/speech_to_text.py \
        model.train_ds.manifest_filepath="/Users/okuchaiev/Data/an4_dataset/an4_train.json" \
        model.validation_ds.manifest_filepath="/Users/okuchaiev/Data/an4_dataset/an4_val.json" \
        hydra.run.dir="." \
        trainer.gpus=0 \
        trainer.max_epochs=50


Add PyTorch Lightning Trainer arguments from CLI:
    python speech_to_text.py \
        ... \
        +trainer.fast_dev_run=true

Hydra logs will be found in "$(./outputs/$(date +"%y-%m-%d")/$(date +"%H-%M-%S")/.hydra)"
PTL logs will be found in "$(./outputs/$(date +"%y-%m-%d")/$(date +"%H-%M-%S")/lightning_logs)"

Override some args of optimizer:
    python speech_to_text.py \
    model.train_ds.manifest_filepath="./an4/train_manifest.json" \
    model.validation_ds.manifest_filepath="./an4/test_manifest.json" \
    hydra.run.dir="." \
    trainer.gpus=2 \
    trainer.max_epochs=2 \
    model.optim.args.params.betas=[0.8,0.5] \
    model.optim.args.params.weight_decay=0.0001

Overide optimizer entirely
    python speech_to_text.py \
    model.train_ds.manifest_filepath="./an4/train_manifest.json" \
    model.validation_ds.manifest_filepath="./an4/test_manifest.json" \
    hydra.run.dir="." \
    trainer.gpus=2 \
    trainer.max_epochs=2 \
    model.optim.name=adamw \
    model.optim.lr=0.001 \
    ~model.optim.args \
    +model.optim.args.betas=[0.8,0.5]\
    +model.optim.args.weight_decay=0.0005

"""


@hydra_runner(config_path="../conf", config_name="quartznet_15x5")
def main(cfg):
    perturb.register_perturbation(name='rir_noise_aug', perturbation=RirAndNoisePerturbation)
    perturb.register_perturbation(name='transcode_aug', perturbation=TranscodePerturbation)

    OmegaConf.set_struct(cfg, False)
    if cfg.trainer.gpus > 0 and "num_workers" not in cfg.model.train_ds:
        total_cpus = os.cpu_count()
        cfg.model.train_ds.num_workers = max(int(total_cpus / cfg.trainer.gpus), 1)
    cfg.model.train_ds.augmentor = OmegaConf.create(get_augmentor_dict())

    # Lets see the new optim config
    print("New Config: ")
    print(OmegaConf.to_yaml(cfg.model.train_ds.augmentor))
    OmegaConf.set_struct(cfg, True)

    trainer = pl.Trainer(**cfg.trainer)
    exp_manager(trainer, cfg.get("exp_manager", None))
    asr_model = EncDecCTCModel(cfg=cfg.model, trainer=trainer)

    encoder_path = "/home/jbalam/noiseaug_exp/models/asrset1.2/qn600/JasperEncoder-STEP-284700.pt"
    decoder_path = "/home/jbalam/noiseaug_exp/models/asrset1.2/qn600/JasperDecoderForCTC-STEP-284700.pt"
    encoder = torch.load(encoder_path, map_location=torch.device('cpu'))
    decoder = torch.load(decoder_path, map_location=torch.device('cpu'))
    asr_model.encoder.load_state_dict(encoder)
    asr_model.decoder.load_state_dict(decoder)

    trainer.fit(asr_model)

    # trainer.test(asr_model, ckpt_path=None)


if __name__ == '__main__':
    main()  # noqa pylint: disable=no-value-for-parameter
