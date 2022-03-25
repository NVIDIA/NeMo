# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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

import argparse
import json
import random
import shutil
import urllib.request
from pathlib import Path

from nemo_text_processing.text_normalization.normalize import Normalizer
from tqdm import tqdm


def get_args():
    parser = argparse.ArgumentParser(
        description='Download HUI-Audio-Corpus-German and create manifests with predefined split'
    )

    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument(
        "--speaker",
        default="Karlsson",
        choices=["Bernd_Ungerer", "Hokuspokus", "Friedrich", "Karlsson", "Eva_K"],
        type=str,
    )
    parser.add_argument("--set-type", default="clean", choices=["full", "clean"], type=str)

    parser.add_argument("--min-duration", default=0.1, type=float)
    parser.add_argument("--max-duration", default=15, type=float)
    parser.add_argument("--val-size", default=100, type=int)
    parser.add_argument("--test-size", default=200, type=int)
    parser.add_argument(
        "--seed-for-ds-split",
        default=100,
        type=float,
        help="Seed for deterministic split of train/dev/test, NVIDIA's default is 100",
    )

    args = parser.parse_args()
    return args


URLS_FULL = {
    'Bernd_Ungerer': "https://opendata.iisys.de/systemintegration/Datasets/HUI-Audio-Corpus-German/dataset_full/Bernd_Ungerer.zip",
    'Hokuspokus': "https://opendata.iisys.de/systemintegration/Datasets/HUI-Audio-Corpus-German/dataset_full/Hokuspokus.zip",
    'Friedrich': "https://opendata.iisys.de/systemintegration/Datasets/HUI-Audio-Corpus-German/dataset_full/Friedrich.zip",
    'Karlsson': "https://opendata.iisys.de/systemintegration/Datasets/HUI-Audio-Corpus-German/dataset_full/Karlsson.zip",
    'Eva_K': "https://opendata.iisys.de/systemintegration/Datasets/HUI-Audio-Corpus-German/dataset_full/Eva_K.zip",
}
URL_STATS_FULL = "https://opendata.iisys.de/systemintegration/Datasets/HUI-Audio-Corpus-German/datasetStatistic.zip"

URLS_CLEAN = {
    'Bernd_Ungerer': "https://opendata.iisys.de/systemintegration/Datasets/HUI-Audio-Corpus-German/dataset_clean/Bernd_Ungerer_Clean.zip",
    'Hokuspokus': "https://opendata.iisys.de/systemintegration/Datasets/HUI-Audio-Corpus-German/dataset_clean/Hokuspokus_Clean.zip",
    'Friedrich': "https://opendata.iisys.de/systemintegration/Datasets/HUI-Audio-Corpus-German/dataset_clean/Friedrich_Clean.zip",
    'Karlsson': "https://opendata.iisys.de/systemintegration/Datasets/HUI-Audio-Corpus-German/dataset_clean/Karlsson_Clean.zip",
    'Eva_K': "https://opendata.iisys.de/systemintegration/Datasets/HUI-Audio-Corpus-German/dataset_clean/Eva_K_Clean.zip",
}
URL_STATS_CLEAN = (
    "https://opendata.iisys.de/systemintegration/Datasets/HUI-Audio-Corpus-German/datasetStatisticClean.zip"
)


def __maybe_download_file(source_url, destination_path):
    if not destination_path.exists():
        tmp_file_path = destination_path.with_suffix('.tmp')
        urllib.request.urlretrieve(source_url, filename=str(tmp_file_path))
        tmp_file_path.rename(destination_path)


def __extract_file(filepath, data_dir):
    try:
        shutil.unpack_archive(filepath, data_dir)
    except Exception:
        print(f"Error while extracting {filepath}. Already extracted?")


def __process_data(dataset_path, stat_path, min_duration, max_duration, val_size, test_size, seed_for_ds_split):
    # Create normalizer
    text_normalizer = Normalizer(
        lang="de", input_case="cased", overwrite_cache=True, cache_dir=str(dataset_path / "cache_dir"),
    )
    text_normalizer_call_kwargs = {"punct_pre_process": True, "punct_post_process": True}
    normalizer_call = lambda x: text_normalizer.normalize(x, **text_normalizer_call_kwargs)

    entries = []
    with open(stat_path) as f:
        # Let's skip the header
        f.readline()
        for line in tqdm(f):
            file_stem, duration, *_, text = line.strip().split("|")
            duration = float(duration)

            # file_stem -> dir_name (e.g. maerchen_01_f000051 -> maerchen, ber_psychoanalyse_01_f000046 -> ber_psychoanalyse)
            dir_name = "_".join(file_stem.split("_")[:-2])
            audio_path = dataset_path / dir_name / "wavs" / f"{file_stem}.wav"

            if min_duration <= duration <= max_duration:
                normalized_text = normalizer_call(text)
                entry = {
                    'audio_filepath': str(audio_path),
                    'duration': duration,
                    'text': text,
                    'normalized_text': normalized_text,
                }
                entries.append(entry)

    random.Random(seed_for_ds_split).shuffle(entries)
    train_size = len(entries) - val_size - test_size

    assert train_size > 0, "Not enough data for train, val and test"

    def save(p, data):
        with open(p, 'w') as f:
            for d in data:
                f.write(json.dumps(d) + '\n')

    save(dataset_path / "train_manifest.json", entries[:train_size])
    save(dataset_path / "val_manifest.json", entries[train_size : train_size + val_size])
    save(dataset_path / "test_manifest.json", entries[train_size + val_size :])


def main():
    args = get_args()

    speaker = args.speaker
    set_type = args.set_type

    dataset_root = args.data_root / "HUI-Audio-Corpus-German"
    dataset_root.mkdir(parents=True, exist_ok=True)

    speaker_data_source = URLS_FULL[speaker] if set_type == "full" else URLS_CLEAN[speaker]
    stats_source = URL_STATS_FULL if set_type == "full" else URL_STATS_CLEAN

    zipped_speaker_data_path = dataset_root / Path(speaker_data_source).name
    zipped_stats_path = dataset_root / Path(stats_source).name

    __maybe_download_file(speaker_data_source, zipped_speaker_data_path)
    __maybe_download_file(stats_source, zipped_stats_path)

    __extract_file(zipped_speaker_data_path, dataset_root)
    __extract_file(zipped_stats_path, dataset_root)

    # Rename unzipped speaker data folder which has `speaker` name to `Path(speaker_data_source).stem` to avoid name conflicts between full and clean
    speaker_data_path = dataset_root / speaker
    speaker_data_path = speaker_data_path.rename(dataset_root / Path(speaker_data_source).stem)

    stats_path = dataset_root / Path(stats_source).stem / "speacker" / speaker / "overview.csv"

    __process_data(
        speaker_data_path,
        stats_path,
        args.min_duration,
        args.max_duration,
        args.val_size,
        args.test_size,
        args.seed_for_ds_split,
    )


if __name__ == "__main__":
    main()
