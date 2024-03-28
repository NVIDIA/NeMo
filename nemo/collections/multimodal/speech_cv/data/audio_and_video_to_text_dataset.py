# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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

import random
from math import isclose
from typing import Optional, Union

import torch
from omegaconf import DictConfig
from omegaconf.listconfig import ListConfig
from torch.utils.data import ChainDataset

from nemo.collections.asr.data import audio_to_text
from nemo.collections.asr.data.audio_to_text_dataset import convert_to_config_list, get_chain_dataset
from nemo.collections.asr.parts.preprocessing.perturb import process_augmentations
from nemo.collections.multimodal.speech_cv.data import audio_and_video_to_text
from nemo.utils import logging


def get_audio_and_video_to_text_bpe_dataset_from_config(
    config,
    local_rank: int,
    global_rank: int,
    world_size: int,
    tokenizer,
    preprocessor_cfg: Optional[DictConfig] = None,
):
    """
    Construct Audio-And-Video-To-Text BPE dataset from a config.
    Args:
        config: BPE dataset config
        local_rank: model local rank
        global_rank: model global rand
        world_size: world size
        tokenizer: BPE tokenizer
        preprocessor_cfg: preprocessor config, for DALI BPE dataset

    Returns:
        constructed dataset or None if dataset config is invalid or nothing to load
    """
    if 'augmentor' in config:
        augmentor = process_augmentations(config['augmentor'])
    else:
        augmentor = None

    is_concat = config.get('is_concat', False)
    if is_concat:
        if 'concat_sampling' in config and config['concat_sampling'] is None:
            logging.warning(f"Concat dataset requires `concat_sampling` but it was not provided. Config: {config}")
            return None

        if not 'concat_probabilities' in config:
            logging.warning(
                f"Concat dataset requires `concat_probabilities` list but it was not provided. Config: {config}"
            )
            return None
        else:
            if not isclose(sum(config['concat_probabilities']), 1, abs_tol=1e-6):
                logging.warning(f"`concat_probabilities` need to sum to 1. Config: {config}")
                return None

    shuffle = config['shuffle']
    device = 'gpu' if torch.cuda.is_available() else 'cpu'
    if config.get('use_dali', False):
        raise Exception("get_dali_bpe_dataset method not implemented")

    # Instantiate tarred dataset loader or normal dataset loader
    if config.get('is_tarred', False):
        if ('tarred_audio_filepaths' in config and config['tarred_audio_filepaths'] is None) or (
            'manifest_filepath' in config and config['manifest_filepath'] is None
        ):
            logging.warning(
                "Could not load dataset as `manifest_filepath` was None or "
                f"`tarred_audio_filepaths` is None. Provided config : {config}"
            )
            return None

        shuffle_n = config.get('shuffle_n', 4 * config['batch_size']) if shuffle else 0
        if is_concat:
            raise Exception("get_concat_tarred_dataset method not implemented")
        else:
            dataset = get_tarred_dataset(
                config=config,
                tokenizer=tokenizer,
                shuffle_n=shuffle_n,
                global_rank=global_rank,
                world_size=world_size,
                augmentor=augmentor,
            )
    else:
        if 'manifest_filepath' in config and config['manifest_filepath'] is None:
            logging.warning(f"Could not load dataset as `manifest_filepath` was None. Provided config : {config}")
            return None
        if is_concat:
            raise Exception("get_concat_bpe_dataset method not implemented")
        else:
            dataset = get_bpe_dataset(config=config, tokenizer=tokenizer, augmentor=augmentor)
    return dataset


def get_audio_and_video_to_text_char_dataset_from_config(
    config,
    local_rank: int,
    global_rank: int,
    world_size: int,
    audio_preprocessor_cfg: Optional[DictConfig] = None,
    video_preprocessor_cfg: Optional[DictConfig] = None,
):
    """
    Construct Audio-And-Video-To-Text Char dataset from a config.
    Args:
        config: dataset config
        local_rank: model local rank
        global_rank: model global rand
        world_size: world size
        preprocessor_cfg: preprocessor config, for DALI dataset

    Returns:
        constructed dataset or None if dataset config is invalid or nothing to load
    """
    if 'augmentor' in config:
        augmentor = process_augmentations(config['augmentor'])
    else:
        augmentor = None

    is_concat = config.get('is_concat', False)
    if is_concat:
        if 'concat_sampling' in config and config['concat_sampling'] is None:
            logging.warning(f"Concat dataset requires `concat_sampling` but it was not provided. Config: {config}")
            return None

        if not 'concat_probabilities' in config:
            logging.warning(
                f"Concat dataset requires `concat_probabilities` list but it was not provided. Config: {config}"
            )
            return None
        else:
            if not isclose(sum(config['concat_probabilities']), 1, abs_tol=1e-6):
                logging.warning(f"`concat_probabilities` need to sum to 1. Config: {config}")
                return None

    shuffle = config['shuffle']
    device = 'gpu' if torch.cuda.is_available() else 'cpu'
    if config.get('use_dali', False):
        raise Exception("get_dali_char_dataset method not implemented")

    # Instantiate tarred dataset loader or normal dataset loader
    if config.get('is_tarred', False):
        if ('tarred_audio_filepaths' in config and config['tarred_audio_filepaths'] is None) or (
            'manifest_filepath' in config and config['manifest_filepath'] is None
        ):
            logging.warning(
                "Could not load dataset as `manifest_filepath` was None or "
                f"`tarred_audio_filepaths` is None. Provided config : {config}"
            )
            return None

        shuffle_n = config.get('shuffle_n', 4 * config['batch_size']) if shuffle else 0
        if is_concat:
            raise Exception("get_concat_tarred_dataset method not implemented")
        else:
            dataset = get_tarred_dataset(
                config=config,
                shuffle_n=shuffle_n,
                global_rank=global_rank,
                world_size=world_size,
                augmentor=augmentor,
            )
    else:
        if 'manifest_filepath' in config and config['manifest_filepath'] is None:
            logging.warning(f"Could not load dataset as `manifest_filepath` was None. Provided config : {config}")
            return None
        if is_concat:
            raise Exception("get_concat_char_dataset method not implemented")
        else:
            dataset = get_char_dataset(config=config, augmentor=augmentor)
    return dataset


def get_bpe_dataset(
    config: dict, tokenizer: 'TokenizerSpec', augmentor: Optional['AudioAugmentor'] = None
) -> audio_and_video_to_text.AudioAndVideoToBPEDataset:
    """
    Instantiates a Byte Pair Encoding / Word Piece Encoding based AudioAndVideoToBPEDataset.

    Args:
        config: Config of the AudioAndVideoToBPEDataset.
        tokenizer: An instance of a TokenizerSpec object.
        augmentor: Optional AudioAugmentor object for augmentations on audio data.

    Returns:
        An instance of AudioToBPEDataset.
    """
    dataset = audio_and_video_to_text.AudioAndVideoToBPEDataset(
        manifest_filepath=config['manifest_filepath'],
        tokenizer=tokenizer,
        sample_rate=config['sample_rate'],
        int_values=config.get('int_values', False),
        augmentor=augmentor,
        max_duration=config.get('max_duration', None),
        min_duration=config.get('min_duration', None),
        max_utts=config.get('max_utts', 0),
        trim=config.get('trim_silence', False),
        use_start_end_token=config.get('use_start_end_token', True),
        return_sample_id=config.get('return_sample_id', False),
        channel_selector=config.get('channel_selector', None),
    )
    return dataset


def get_char_dataset(
    config: dict, augmentor: Optional['AudioAugmentor'] = None
) -> audio_and_video_to_text.AudioAndVideoToCharDataset:
    """
    Instantiates a Character Encoding based AudioAndVideoToCharDataset.

    Args:
        config: Config of the AudioAndVideoToCharDataset.
        augmentor: Optional AudioAugmentor object for augmentations on audio data.

    Returns:
        An instance of AudioAndVideoToCharDataset.
    """
    if 'labels' not in config:
        logging.warning(f"dataset does not have explicitly defined labels")

    dataset = audio_and_video_to_text.AudioAndVideoToCharDataset(
        manifest_filepath=config['manifest_filepath'],
        labels=config.get('labels', None),
        sample_rate=config['sample_rate'],
        int_values=config.get('int_values', False),
        augmentor=augmentor,
        max_duration=config.get('max_duration', None),
        min_duration=config.get('min_duration', None),
        max_utts=config.get('max_utts', 0),
        blank_index=config.get('blank_index', -1),
        unk_index=config.get('unk_index', -1),
        normalize=config.get('normalize_transcripts', False),
        trim=config.get('trim_silence', False),
        parser=config.get('parser', 'en'),
        return_sample_id=config.get('return_sample_id', False),
        channel_selector=config.get('channel_selector', None),
    )
    return dataset


def get_tarred_dataset(
    config: dict,
    shuffle_n: int,
    global_rank: int,
    world_size: int,
    tokenizer: Optional['TokenizerSpec'] = None,
    augmentor: Optional['AudioAugmentor'] = None,
) -> Union[
    audio_and_video_to_text.TarredAudioAndVideoToBPEDataset, audio_and_video_to_text.TarredAudioAndVideoToCharDataset
]:
    """
    Instantiates a Word Piece/BPE Encoding based TarredAudioAndVideoToBPEDataset or a char based TarredAudioAndVideoToCharDataset.

    Args:
        config: Config of the TarredAudioAndVideoToBPEDataset or TarredAudioAndVideoToCharDataset.
        shuffle_n: How many samples to look ahead and load to be shuffled.
            See WebDataset documentation for more details.
        tokenizer: An instance of a TokenizerSpec object if BPE dataset is needed.
        global_rank: Global rank of this device.
        world_size: Global world size in the training method.
            Passsing None would return a char-based dataset.
        augmentor: Optional AudioAugmentor object for augmentations on audio data.

    Returns:
        An instance of TarredAudioAndVideoToBPEDataset or TarredAudioAndVideoToCharDataset.
    """
    tarred_audio_filepaths = config['tarred_audio_filepaths']
    manifest_filepaths = config['manifest_filepath']
    datasets = []
    tarred_audio_filepaths = convert_to_config_list(tarred_audio_filepaths)
    manifest_filepaths = convert_to_config_list(manifest_filepaths)

    bucketing_weights = config.get('bucketing_weights', None)  # For upsampling buckets
    if bucketing_weights:
        for idx, weight in enumerate(bucketing_weights):
            if not isinstance(weight, int) or weight <= 0:
                raise ValueError(f"bucket weights must be positive integers")

    if len(manifest_filepaths) != len(tarred_audio_filepaths):
        raise ValueError(
            f"manifest_filepaths (length={len(manifest_filepaths)}) and tarred_audio_filepaths (length={len(tarred_audio_filepaths)}) need to have the same number of buckets."
        )

    if 'labels' not in config:
        logging.warning(f"dataset does not have explicitly defined labels")

    if 'max_utts' in config:
        raise ValueError('"max_utts" parameter is not supported for tarred datasets')

    for dataset_idx, (tarred_audio_filepath, manifest_filepath) in enumerate(
        zip(tarred_audio_filepaths, manifest_filepaths)
    ):
        if len(tarred_audio_filepath) == 1:
            tarred_audio_filepath = tarred_audio_filepath[0]
        if tokenizer is None:
            dataset = audio_and_video_to_text.TarredAudioAndVideoToCharDataset(
                audio_tar_filepaths=tarred_audio_filepath,
                manifest_filepath=manifest_filepath,
                labels=config.get('labels', None),
                sample_rate=config['sample_rate'],
                int_values=config.get('int_values', False),
                augmentor=augmentor,
                shuffle_n=shuffle_n,
                max_duration=config.get('max_duration', None),
                min_duration=config.get('min_duration', None),
                blank_index=config.get('blank_index', -1),
                unk_index=config.get('unk_index', -1),
                normalize=config.get('normalize_transcripts', False),
                trim=config.get('trim_silence', False),
                parser=config.get('parser', 'en'),
                shard_strategy=config.get('tarred_shard_strategy', 'scatter'),
                global_rank=global_rank,
                world_size=world_size,
                return_sample_id=config.get('return_sample_id', False),
                # Align Audio to Video by pading one of the two signals for features fusion
                align_audio_video=config.get('align_audio_video', True),
                spec_pad_to=config.get('spec_pad_to', 0),  # Spectrogram padding during audio preprocessing
                hop_length=config.get('hop_length', 160),  # hop_length of audio preprocessing
                spec_down_factor=config.get('spec_down_factor', 2 * 2),  # Downsampling factor of audio encoder
            )
        else:
            dataset = audio_and_video_to_text.TarredAudioAndVideoToBPEDataset(
                audio_tar_filepaths=tarred_audio_filepath,
                manifest_filepath=manifest_filepath,
                tokenizer=tokenizer,
                sample_rate=config['sample_rate'],
                int_values=config.get('int_values', False),
                augmentor=augmentor,
                shuffle_n=shuffle_n,
                max_duration=config.get('max_duration', None),
                min_duration=config.get('min_duration', None),
                trim=config.get('trim_silence', False),
                use_start_end_token=config.get('use_start_end_token', True),
                shard_strategy=config.get('tarred_shard_strategy', 'scatter'),
                global_rank=global_rank,
                world_size=world_size,
                return_sample_id=config.get('return_sample_id', False),
                # Align Audio to Video by pading one of the two signals for features fusion
                align_audio_video=config.get('align_audio_video', True),
                spec_pad_to=config.get('spec_pad_to', 0),  # Spectrogram padding during audio preprocessing
                hop_length=config.get('hop_length', 160),  # hop_length of audio preprocessing
                spec_down_factor=config.get('spec_down_factor', 2 * 2),  # Downsampling factor of audio encoder
            )
        if bucketing_weights:
            [datasets.append(dataset) for _ in range(bucketing_weights[dataset_idx])]
        else:
            datasets.append(dataset)

    return get_chain_dataset(datasets=datasets, ds_config=config)
