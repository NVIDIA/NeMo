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
This script computes audio tokens and stores them for TTS training.

$ python /NeMo/scripts/dataset_processing/tts/compute_audio_tokens.py \
    --manifest_path=train_manifest.json \
    --audio_dir=/data/audio \
    --feature_dir=/data/features \
    --feature_name="audio_tokens" \
    --model_path=/models/SpeechCodec.nemo \
    --volume_norm \
    --device=cuda:0 \
    --batch_size=16
"""

import argparse
import json
import torch
import numpy as np
from einops import rearrange
from pathlib import Path
from tqdm import tqdm

from nemo.collections.tts.models.audio_codec import AudioCodecModel
from nemo.collections.tts.parts.preprocessing.features import save_numpy_feature, _features_exists
from nemo.collections.tts.parts.utils.helpers import load_model
from nemo.collections.tts.parts.utils.tts_dataset_utils import load_audio, stack_tensors


def get_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter, description="Compute TTS features.",
    )
    parser.add_argument(
        "--manifest_path", required=True, type=Path, help="Path to training manifest.",
    )
    parser.add_argument(
        "--audio_dir", required=True, type=Path, help="Path to base directory with audio data.",
    )
    parser.add_argument(
        "--feature_dir", required=True, type=Path, help="Path to feature directory where tokens will be stored.",
    )
    parser.add_argument(
        "--feature_name",
        type=str,
        default="audio_tokens",
        help="Name (directory) to store tokens under.",
    )
    parser.add_argument(
        "--model_name", type=Path, help="Name of NGC model to load.",
    )
    parser.add_argument(
        "--model_path", type=Path, help="Path to checkpoint to load.",
    )
    parser.add_argument(
        "--volume_norm",
        action=argparse.BooleanOptionalAction,
        help="Whether to normalize volume of audio before computing tokens.",
    )
    parser.add_argument(
        "--device", default="cpu", type=str, help="Device to run model on.",
    )
    parser.add_argument(
        "--batch_size", required=True, type=int, help="Batch size to user during inference.",
    )
    parser.add_argument(
        "--overwrite", action=argparse.BooleanOptionalAction, help="Whether to overwrite existing feature files.",
    )
    args = parser.parse_args()
    return args


def _process_batch(entries, audio_codec, audio_dir, feature_dir, feature_name, volume_norm):
    audio_list = []
    audio_len_list = []
    for entry in entries:
        audio_array, _, _ = load_audio(
            manifest_entry=entry,
            audio_dir=audio_dir,
            sample_rate=audio_codec.sample_rate,
            volume_norm=volume_norm,
        )
        audio = torch.from_numpy(audio_array)
        audio_list.append(audio)
        audio_len_list.append(audio.shape[0])

    max_len = max(audio_len_list)
    audio = stack_tensors(audio_list, max_lens=[max_len]).to(audio_codec.device)
    audio_len = torch.IntTensor(audio_len_list).to(audio_codec.device)

    with torch.no_grad():
        # [batch_size, num_codebook, T_token]
        tokens, token_lens = audio_codec.encode(audio=audio, audio_len=audio_len)
        tokens = tokens.cpu().numpy().astype(np.int32)
        token_lens = token_lens.cpu().numpy()

    for i in range(len(audio_list)):
        entry = entries[i]
        token_len = token_lens[i]
        # [num_codebook, T_token]
        token = tokens[i, :, :token_len]
        token = rearrange(token, 'C T -> T C')
        save_numpy_feature(
            feature_name=feature_name,
            features=token,
            manifest_entry=entry,
            audio_dir=audio_dir,
            feature_dir=feature_dir,
        )


def get_entries_sorted_by_duration(manifest_path):
    from collections import defaultdict
    audio_dur_map = defaultdict(float)
    with open(manifest_path, "r", encoding="utf-8") as input_f:
        for line in input_f:
            entry = json.loads(line)
            audio_filepath = entry["audio_filepath"]
            duration = entry["duration"]
            audio_dur_map[audio_filepath] += duration

    audio_dur_list = list(audio_dur_map.items())
    audio_dur_list.sort(key=lambda x: x[1], reverse=True)
    entry_list = [{"audio_filepath": x[0]} for x in audio_dur_list]
    return entry_list


def filter_existing_entries(entries, feature_name, audio_dir, feature_dir):
    filtered_entries = []
    for entry in entries:
        if not _features_exists(
            feature_names=[feature_name],
            manifest_entry=entry,
            audio_dir=audio_dir,
            feature_dir=feature_dir,
        ):
            filtered_entries.append(entry)
    return filtered_entries


def main():
    args = get_args()
    manifest_path = args.manifest_path
    audio_dir = args.audio_dir
    feature_dir = args.feature_dir
    feature_name = args.feature_name
    model_name = args.model_name
    model_path = args.model_path
    volume_norm = args.volume_norm
    device = args.device
    batch_size = args.batch_size
    overwrite = args.overwrite

    if not manifest_path.exists():
        raise ValueError(f"Manifest {manifest_path} does not exist.")

    if not audio_dir.exists():
        raise ValueError(f"Audio directory {audio_dir} does not exist.")

    audio_codec = load_model(
        model_type=AudioCodecModel,
        device=device,
        model_name=model_name,
        checkpoint_path=model_path
    )

    print(f"Reading manifest file {manifest_path}")
    entries = get_entries_sorted_by_duration(manifest_path)
    num_entries = len(entries)
    print(f"Found {num_entries} files.")
    if not overwrite:
        entries = filter_existing_entries(
            entries=entries,
            audio_dir=audio_dir,
            feature_dir=feature_dir,
            feature_name=feature_name,
        )
        print(f"Ignoring {num_entries - len(entries)} files with existing features.")

    batch_list = []
    for entry in tqdm(entries):
        batch_list.append(entry)
        if len(batch_list) == batch_size:
            _process_batch(
                entries=batch_list,
                audio_codec=audio_codec,
                audio_dir=audio_dir,
                feature_dir=feature_dir,
                feature_name=feature_name,
                volume_norm=volume_norm)
            batch_list = []

    if batch_list:
        _process_batch(
            entries=batch_list,
            audio_codec=audio_codec,
            audio_dir=audio_dir,
            feature_dir=feature_dir,
            feature_name=feature_name,
            volume_norm=volume_norm
        )


if __name__ == "__main__":
    main()
