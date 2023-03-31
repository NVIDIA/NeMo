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

"""
This script computes features for TTS models prior to training, such as pitch and energy.
The resulting features will be stored in the provided 'sup_data_path'.

$ python <nemo_root_path>/scripts/dataset_processing/tts/compute_features.py \
    --feature_config_path=<nemo_root_path>/examples/tts/conf/features/feature_22050.yaml \
    --manifest_path=<data_root_path>/manifest.json \
    --audio_path=<data_root_path>/audio \
    --sup_data_path=<data_root_path>/sup_data \
    --num_workers=1
"""

import argparse
import os
from pathlib import Path

import librosa
import torch
from hydra.utils import instantiate
from joblib import Parallel, delayed
from omegaconf import OmegaConf
from tqdm import tqdm

from nemo.collections.asr.parts.utils.manifest_utils import read_manifest
from nemo.collections.tts.parts.preprocessing.features import (
    MelSpectrogramFeaturizer,
    PitchFeaturizer,
    TTSFeature,
    compute_energy,
)
from nemo.collections.tts.parts.utils.tts_dataset_utils import get_audio_paths, get_sup_data_file_name


def get_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter, description="Compute TTS features.",
    )
    parser.add_argument(
        "--feature_config_path", required=True, type=Path, help="Path to feature config file.",
    )
    parser.add_argument(
        "--manifest_path", required=True, type=Path, help="Path to training manifest.",
    )
    parser.add_argument(
        "--audio_path", required=True, type=Path, help="Path to base directory with audio data.",
    )
    parser.add_argument(
        "--sup_data_path", required=True, type=Path, help="Path to directory where supplementary data will be stored.",
    )
    parser.add_argument(
        "--save_pitch", default=True, type=bool, help="Whether to save pitch features.",
    )
    parser.add_argument(
        "--save_voiced", default=True, type=bool, help="Whether to save voiced mask.",
    )
    parser.add_argument(
        "--save_energy", default=True, type=bool, help="Whether to save energy features.",
    )
    parser.add_argument(
        "--num_workers", default=1, type=int, help="Number of parallel threads to use. If -1 all CPUs are used."
    )
    args = parser.parse_args()
    return args


def _create_feature_dir(base_path: Path, save_feature: bool, feature_name: str):
    if not save_feature:
        return None

    feature_path = base_path / feature_name
    feature_path.mkdir(exist_ok=True)
    return feature_path


def _process_entry(
    entry: dict,
    audio_base_path: Path,
    pitch_base_path: Path,
    voiced_base_path: Path,
    energy_base_path: Path,
    sample_rate: int,
    mel_featurizer: MelSpectrogramFeaturizer,
    pitch_featurizer: PitchFeaturizer,
) -> None:
    audio_filepath = Path(entry["audio_filepath"])

    audio_path, audio_path_rel = get_audio_paths(audio_path=audio_filepath, base_path=audio_base_path)

    audio, _ = librosa.load(path=audio_path, sr=sample_rate)
    sup_data_file_name = get_sup_data_file_name(audio_path_rel)

    if pitch_base_path or voiced_base_path:
        pitch, voiced, _ = pitch_featurizer.compute_pitch(audio)

        if pitch_base_path:
            pitch_path = pitch_base_path / sup_data_file_name
            pitch_tensor = torch.tensor(pitch, dtype=torch.float32)
            torch.save(pitch_tensor, pitch_path)

        if voiced_base_path:
            voiced_path = voiced_base_path / sup_data_file_name
            voiced_tensor = torch.tensor(voiced, dtype=torch.bool)
            torch.save(voiced_tensor, voiced_path)

    if energy_base_path:
        energy_path = energy_base_path / sup_data_file_name
        spec = mel_featurizer.compute_mel_spectrogram(audio)
        energy = compute_energy(spec)
        energy_tensor = torch.tensor(energy, dtype=torch.float32)
        torch.save(energy_tensor, energy_path)

    return


def main():
    args = get_args()
    feature_config_path = args.feature_config_path
    manifest_path = args.manifest_path
    audio_base_path = args.audio_path
    sup_base_path = args.sup_data_path
    save_pitch = args.save_pitch
    save_voiced = args.save_voiced
    save_energy = args.save_energy
    num_workers = args.num_workers

    if not os.path.exists(manifest_path):
        raise ValueError(f"Manifest {manifest_path} does not exist.")

    if not os.path.exists(audio_base_path):
        raise ValueError(f"Audio directory {audio_base_path} does not exist.")

    sup_base_path.mkdir(exist_ok=True, parents=True)
    os.makedirs(sup_base_path, exist_ok=True)

    pitch_base_path = _create_feature_dir(
        base_path=sup_base_path, save_feature=save_pitch, feature_name=TTSFeature.PITCH.value
    )
    voiced_base_path = _create_feature_dir(
        base_path=sup_base_path, save_feature=save_voiced, feature_name=TTSFeature.VOICED.value
    )
    energy_base_path = _create_feature_dir(
        base_path=sup_base_path, save_feature=save_energy, feature_name=TTSFeature.ENERGY.value
    )

    feature_config = OmegaConf.load(feature_config_path)
    feature_config = instantiate(feature_config)

    sample_rate = feature_config.sample_rate
    mel_featurizer = feature_config.mel
    pitch_featurizer = feature_config.pitch

    entries = read_manifest(manifest_path)

    Parallel(n_jobs=num_workers)(
        delayed(_process_entry)(
            entry=entry,
            audio_base_path=audio_base_path,
            pitch_base_path=pitch_base_path,
            voiced_base_path=voiced_base_path,
            energy_base_path=energy_base_path,
            sample_rate=sample_rate,
            mel_featurizer=mel_featurizer,
            pitch_featurizer=pitch_featurizer,
        )
        for entry in tqdm(entries)
    )


if __name__ == "__main__":
    main()
