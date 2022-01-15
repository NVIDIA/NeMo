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
import glob
import json
import os
import shutil
from dataclasses import dataclass, is_dataclass
from typing import Optional
from omegaconf import OmegaConf, open_dict
import hydra
from hydra.core.config_store import ConfigStore

from datasets import load_dataset, Dataset, IterableDataset, Audio
import librosa
import soundfile
import tqdm


@dataclass
class HFDatasetConvertionConfig:
    # HF Dataset info
    path: str
    output_dir: str

    name: Optional[str] = None
    split: Optional[str] = None

    # NeMo dataset conversion
    sampling_rate: int = 16000
    streaming: bool = False

    # Placeholders. Generated internally.
    resolved_output_dir: str = ''
    split_output_dir: Optional[str] = None


def prepare_output_dirs(cfg: HFDatasetConvertionConfig):
    output_dir = os.path.abspath(cfg.output_dir)
    output_dir = os.path.join(output_dir, cfg.path)

    if cfg.name is not None:
        output_dir = os.path.join(output_dir, cfg.name)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    cfg.resolved_output_dir = output_dir
    cfg.split_output_dir = None


def build_map_dataset_to_nemo_func(cfg: HFDatasetConvertionConfig, basedir):
    def map_dataset_to_nemo(batch):
        # Write audio file to correct path
        if cfg.streaming:
            batch['audio_filepath'] = batch['audio']['path'].split("::")[0].replace("zip://", "")
        else:
            segments = []
            segment, path = os.path.split(batch['audio']['path'])
            segments.insert(0, path)
            while segment != os.path.sep:
                segment, path = os.path.split(segment)
                segments.insert(0, path)

            index_of_basedir = segments.index("extracted")
            segments = segments[(index_of_basedir + 1 + 1) :]  # skip .../extracted/{hash}/
            audio_filepath = os.path.join(*segments)
            batch['audio_filepath'] = audio_filepath

        batch['audio_filepath'] = os.path.abspath(os.path.join(basedir, batch['audio_filepath']))
        audio_filepath = batch['audio_filepath']

        audio_basefilepath = os.path.split(audio_filepath)[0]
        if not os.path.exists(audio_basefilepath):
            os.makedirs(audio_basefilepath, exist_ok=True)

        if os.path.exists(audio_filepath):
            os.remove(audio_filepath)

        soundfile.write(audio_filepath, batch['audio']['array'], samplerate=cfg.sampling_rate)

        batch['duration'] = librosa.get_duration(batch['audio']['array'], sr=batch['audio']['sampling_rate'])
        return batch

    return map_dataset_to_nemo


def process_dataset(dataset: IterableDataset, cfg: HFDatasetConvertionConfig):
    dataset = dataset.cast_column("audio", Audio(cfg.sampling_rate, mono=True))

    if cfg.split_output_dir is None:
        basedir = cfg.resolved_output_dir
        split = None
        manifest_filename = f"{cfg.path.replace('/', '_')}_manifest.json"
    else:
        basedir = cfg.split_output_dir
        split = os.path.split(cfg.split_output_dir)[-1]
        manifest_filename = f"{split}_{cfg.path.replace('/', '_')}_manifest.json"

    manifest_filepath = os.path.abspath(os.path.join(basedir, manifest_filename))

    if cfg.streaming:
        convert_streaming_dataset_to_nemo(dataset, cfg, basedir=basedir, manifest_filepath=manifest_filepath)
    else:
        convert_offline_dataset_to_nemo(dataset, cfg, basedir=basedir, manifest_filepath=manifest_filepath)

    print()
    print("Dataset conversion finished !")


def convert_offline_dataset_to_nemo(
    dataset: IterableDataset, cfg: HFDatasetConvertionConfig, basedir: str, manifest_filepath: str
):
    # Disable until fix https://github.com/huggingface/datasets/pull/3556 is merged
    dataset = dataset.map(build_map_dataset_to_nemo_func(cfg, basedir))
    ds_iter = iter(dataset)

    with open(manifest_filepath, 'w') as manifest_f:
        for idx, sample in enumerate(
            tqdm.tqdm(ds_iter, desc=f'Processing {cfg.path}:', total=len(dataset), unit=' samples')
        ):
            # remove large components from sample
            del sample['audio']
            if 'file' in sample:
                del sample['file']

            manifest_f.write(f"{json.dumps(sample)}\n")


def convert_streaming_dataset_to_nemo(
    dataset: IterableDataset, cfg: HFDatasetConvertionConfig, basedir: str, manifest_filepath: str
):
    # Disable until fix https://github.com/huggingface/datasets/pull/3556 is merged
    # dataset = dataset.map(build_map_dataset_to_nemo_func(cfg, basedir))

    ds_iter = iter(dataset)

    with open(manifest_filepath, 'w') as manifest_f:
        for idx, sample in enumerate(tqdm.tqdm(ds_iter, desc=f'Processing {cfg.path}:', unit=' samples')):

            audio_filepath = sample['audio']['path'].split("::")[0].replace("zip://", "")
            audio_filepath = os.path.abspath(os.path.join(basedir, audio_filepath))

            audio_basefilepath = os.path.split(audio_filepath)[0]
            if not os.path.exists(audio_basefilepath):
                os.makedirs(audio_basefilepath, exist_ok=True)

            if os.path.exists(audio_filepath):
                os.remove(audio_filepath)

            soundfile.write(audio_filepath, sample['audio']['array'], samplerate=cfg.sampling_rate)

            manifest_line = {
                'audio_filepath': audio_filepath,
                'text': sample['text'],
                'duration': librosa.get_duration(sample['audio']['array'], sr=cfg.sampling_rate),
            }

            # remove large components from sample
            del sample['audio']
            del sample['text']
            if 'file' in sample:
                del sample['file']

            manifest_line.update(sample)

            manifest_f.write(f"{json.dumps(manifest_line)}\n")


@hydra.main(config_name='hfds_config', config_path=None)
def main(cfg: HFDatasetConvertionConfig):
    if is_dataclass(cfg):
        cfg = OmegaConf.structured(cfg)

    prepare_output_dirs(cfg)

    # Load dataset in streaming mode
    dataset = load_dataset(
        path=cfg.path,
        name=cfg.name,
        split=cfg.split,
        cache_dir=None,
        streaming=cfg.streaming,
        data_dir=cfg.resolved_output_dir,
    )

    if isinstance(dataset, dict):
        print("Multiple splits found for dataset", cfg.path, ":", list(dataset.keys()))

    else:
        print("Single split found for dataset", cfg.path)

        process_dataset(dataset, cfg)


ConfigStore.instance().store(name='hfds_config', node=HFDatasetConvertionConfig)

if __name__ == '__main__':
    cfg = HFDatasetConvertionConfig(
        path='timit_asr', name=None, split='train', output_dir='/media/smajumdar/data/Datasets/Timit'
    )

    main(cfg)
