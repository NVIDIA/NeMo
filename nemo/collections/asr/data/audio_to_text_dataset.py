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

from typing import Optional

import torch
from omegaconf import DictConfig, open_dict

from nemo.collections.asr.data import audio_to_text, audio_to_text_dali
from nemo.utils import logging


def inject_dataloader_value_from_model_config(model_cfg: dict, dataloader_cfg: dict, key: str):
    """
    Extracts the label set provided at the top level of the model, and propagates it to the dataloader
    config.

    Args:
        model_cfg: A DictConfig representing the model's config.
        dataloader_cfg: A DictConfig representing the individual data loader
        key: A str value representing a key in the model_cfg whose value will be propagated to the
            dataloader config.
    """
    if key not in model_cfg:
        logging.info(
            f"Model level config does not container `{key}`, please explicitly provide `{key}` to the dataloaders."
        )
        return

    # If key exists in the data loader config (either set explicitly or as a placeholder (via None))
    if key in dataloader_cfg:
        # Dataloader `labels` is provided and is non-null
        if dataloader_cfg[key] is not None and model_cfg[key] != dataloader_cfg[key]:
            # Model level `labels` dont match Dataloader level `labels`
            logging.warning(
                f'`{key}` is explicitly provided to the data loader, and is different from '
                f'the `{key}` provided at the model level config.\n'
                f'If this is incorrect, please set the dataloader\'s `{key}` to None.'
            )

        else:
            # Dataloader `key` is None or values match
            # Propagate from model level `key` (even if they match)
            with open_dict(dataloader_cfg):
                dataloader_cfg[key] = model_cfg[key]

    else:
        # If key key doesnt even exist in dataloader_cfg, inject it explicitly
        with open_dict(dataloader_cfg):
            dataloader_cfg[key] = model_cfg[key]


def get_char_dataset(config: dict, augmentor: Optional['AudioAugmentor'] = None) -> audio_to_text.AudioToCharDataset:
    """
    Instantiates a Character Encoding based AudioToCharDataset.

    Args:
        config: Config of the AudioToCharDataset.
        augmentor: Optional AudioAugmentor object for augmentations on audio data.

    Returns:
        An instance of AudioToCharDataset.
    """
    dataset = audio_to_text.AudioToCharDataset(
        manifest_filepath=config['manifest_filepath'],
        labels=config['labels'],
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
    )
    return dataset


def get_bpe_dataset(
    config: dict, tokenizer: 'TokenizerSpec', augmentor: Optional['AudioAugmentor'] = None
) -> audio_to_text.AudioToBPEDataset:
    """
    Instantiates a Byte Pair Encoding / Word Piece Encoding based AudioToBPEDataset.

    Args:
        config: Config of the AudioToBPEDataset.
        tokenizer: An instance of a TokenizerSpec object.
        augmentor: Optional AudioAugmentor object for augmentations on audio data.

    Returns:
        An instance of AudioToBPEDataset.
    """
    dataset = audio_to_text.AudioToBPEDataset(
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
    )
    return dataset


def get_tarred_char_dataset(
    config: dict, shuffle_n: int, global_rank: int, world_size: int, augmentor: Optional['AudioAugmentor'] = None
) -> audio_to_text.TarredAudioToCharDataset:
    """
    Instantiates a Character Encoding based TarredAudioToCharDataset.

    Args:
        config: Config of the TarredAudioToCharDataset.
        shuffle_n: How many samples to look ahead and load to be shuffled.
            See WebDataset documentation for more details.
        global_rank: Global rank of this device.
        world_size: Global world size in the training method.
        augmentor: Optional AudioAugmentor object for augmentations on audio data.

    Returns:
        An instance of TarredAudioToCharDataset.
    """
    dataset = audio_to_text.TarredAudioToCharDataset(
        audio_tar_filepaths=config['tarred_audio_filepaths'],
        manifest_filepath=config['manifest_filepath'],
        labels=config['labels'],
        sample_rate=config['sample_rate'],
        int_values=config.get('int_values', False),
        augmentor=augmentor,
        shuffle_n=shuffle_n,
        max_duration=config.get('max_duration', None),
        min_duration=config.get('min_duration', None),
        max_utts=config.get('max_utts', 0),
        blank_index=config.get('blank_index', -1),
        unk_index=config.get('unk_index', -1),
        normalize=config.get('normalize_transcripts', False),
        trim=config.get('trim_silence', False),
        parser=config.get('parser', 'en'),
        shard_strategy=config.get('tarred_shard_strategy', 'scatter'),
        global_rank=global_rank,
        world_size=world_size,
    )
    return dataset


def get_tarred_bpe_dataset(
    config: dict,
    tokenizer: 'TokenizerSpec',
    shuffle_n: int,
    global_rank: int,
    world_size: int,
    augmentor: Optional['AudioAugmentor'] = None,
) -> audio_to_text.TarredAudioToBPEDataset:
    """
    Instantiates a Byte Pair Encoding / Word Piece Encoding based TarredAudioToBPEDataset.

    Args:
        config: Config of the TarredAudioToBPEDataset.
        tokenizer: An instance of a TokenizerSpec object.
        shuffle_n: How many samples to look ahead and load to be shuffled.
            See WebDataset documentation for more details.
        global_rank: Global rank of this device.
        world_size: Global world size in the training method.
        augmentor: Optional AudioAugmentor object for augmentations on audio data.

    Returns:
        An instance of TarredAudioToBPEDataset.
    """
    dataset = audio_to_text.TarredAudioToBPEDataset(
        audio_tar_filepaths=config['tarred_audio_filepaths'],
        manifest_filepath=config['manifest_filepath'],
        tokenizer=tokenizer,
        sample_rate=config['sample_rate'],
        int_values=config.get('int_values', False),
        augmentor=augmentor,
        shuffle_n=shuffle_n,
        max_duration=config.get('max_duration', None),
        min_duration=config.get('min_duration', None),
        max_utts=config.get('max_utts', 0),
        trim=config.get('trim_silence', False),
        use_start_end_token=config.get('use_start_end_token', True),
        shard_strategy=config.get('tarred_shard_strategy', 'scatter'),
        global_rank=global_rank,
        world_size=world_size,
    )
    return dataset


def get_dali_char_dataset(
    config: dict,
    shuffle: bool,
    device_id: int,
    global_rank: int,
    world_size: int,
    preprocessor_cfg: Optional[DictConfig] = None,
) -> audio_to_text_dali.AudioToCharDALIDataset:
    """
    Instantiates a Character Encoding based AudioToCharDALIDataset.

    Args:
        config: Config of the AudioToCharDALIDataset.
        shuffle: Bool flag whether to shuffle the dataset.
        device_id: Index of the GPU to be used (local_rank). Only applicable when device == 'gpu'. Defaults to 0.
        global_rank: Global rank of this device.
        world_size: Global world size in the training method.
        augmentor: Optional AudioAugmentor object for augmentations on audio data.

    Returns:
        An instance of AudioToCharDALIDataset.
    """
    device = 'gpu' if torch.cuda.is_available() else 'cpu'
    dataset = audio_to_text_dali.AudioToCharDALIDataset(
        manifest_filepath=config['manifest_filepath'],
        device=device,
        batch_size=config['batch_size'],
        labels=config['labels'],
        sample_rate=config['sample_rate'],
        max_duration=config.get('max_duration', None),
        min_duration=config.get('min_duration', None),
        blank_index=config.get('blank_index', -1),
        unk_index=config.get('unk_index', -1),
        normalize=config.get('normalize_transcripts', False),
        trim=config.get('trim_silence', False),
        parser=config.get('parser', 'en'),
        shuffle=shuffle,
        device_id=device_id,
        global_rank=global_rank,
        world_size=world_size,
        preprocessor_cfg=preprocessor_cfg,
    )
    return dataset


def get_dali_bpe_dataset(
    config: dict,
    tokenizer,
    shuffle: bool,
    device_id: int,
    global_rank: int,
    world_size: int,
    preprocessor_cfg: Optional[DictConfig] = None,
) -> audio_to_text_dali.AudioToCharDALIDataset:
    """
    Instantiates a Subword Encoding based AudioToBPEDALIDataset.

    Args:
        config: Config of the AudioToBPEDALIDataset.
        tokenizer: An implementation of NeMo TokenizerSpec.
        shuffle: Bool flag whether to shuffle the dataset.
        device_id: Index of the GPU to be used (local_rank). Only applicable when device == 'gpu'. Defaults to 0.
        global_rank: Global rank of this device.
        world_size: Global world size in the training method.
        augmentor: Optional AudioAugmentor object for augmentations on audio data.

    Returns:
        An instance of AudioToCharDALIDataset.
    """
    device = 'gpu' if torch.cuda.is_available() else 'cpu'
    dataset = audio_to_text_dali.AudioToBPEDALIDataset(
        manifest_filepath=config['manifest_filepath'],
        tokenizer=tokenizer,
        device=device,
        batch_size=config['batch_size'],
        sample_rate=config['sample_rate'],
        max_duration=config.get('max_duration', None),
        min_duration=config.get('min_duration', None),
        trim=config.get('trim_silence', False),
        use_start_end_token=config.get('use_start_end_token', True),
        shuffle=shuffle,
        device_id=device_id,
        global_rank=global_rank,
        world_size=world_size,
        preprocessor_cfg=preprocessor_cfg,
    )
    return dataset
