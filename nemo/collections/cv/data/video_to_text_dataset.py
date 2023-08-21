import random
from math import isclose
from typing import Optional

import torch
from omegaconf import DictConfig
from omegaconf.listconfig import ListConfig
from torch.utils.data import ChainDataset

from nemo.collections.asr.data import audio_to_text
from nemo.collections.cv.data import video_to_text
from nemo.utils import logging


def get_video_to_text_bpe_dataset_from_config(
    config,
    local_rank: int,
    global_rank: int,
    world_size: int,
    tokenizer,
    preprocessor_cfg: Optional[DictConfig] = None,
):
    """
    Construct Video-To-Text BPE dataset from a config.
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
                config=config, tokenizer=tokenizer, shuffle_n=shuffle_n, global_rank=global_rank, world_size=world_size
            )
    else:
        if 'manifest_filepath' in config and config['manifest_filepath'] is None:
            logging.warning(f"Could not load dataset as `manifest_filepath` was None. Provided config : {config}")
            return None
        if is_concat:
            raise Exception("get_concat_bpe_dataset method not implemented")
        else:
            dataset = get_bpe_dataset(config=config, tokenizer=tokenizer)
    return dataset


def get_video_to_text_char_dataset_from_config(
    config, local_rank: int, global_rank: int, world_size: int, preprocessor_cfg: Optional[DictConfig] = None
):
    """
    Construct Video-To-Text Char dataset from a config.
    Args:
        config: dataset config
        local_rank: model local rank
        global_rank: model global rand
        world_size: world size
        preprocessor_cfg: preprocessor config, for DALI dataset

    Returns:
        constructed dataset or None if dataset config is invalid or nothing to load
    """

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
                config=config, shuffle_n=shuffle_n, global_rank=global_rank, world_size=world_size,
            )
    else:
        if 'manifest_filepath' in config and config['manifest_filepath'] is None:
            logging.warning(f"Could not load dataset as `manifest_filepath` was None. Provided config : {config}")
            return None
        if is_concat:
            raise Exception("get_concat_char_dataset method not implemented")
        else:
            dataset = get_char_dataset(config=config)
    return dataset


def get_bpe_dataset(config: dict, tokenizer: 'TokenizerSpec') -> video_to_text.VideoToBPEDataset:
    """
    Instantiates a Byte Pair Encoding / Word Piece Encoding based VideoToBPEDataset.

    Args:
        config: Config of the VideoToBPEDataset.
        tokenizer: An instance of a TokenizerSpec object.

    Returns:
        An instance of VideoToBPEDataset.
    """
    dataset = video_to_text.VideoToBPEDataset(
        manifest_filepath=config['manifest_filepath'],
        tokenizer=tokenizer,
        int_values=config.get('int_values', False),
        max_duration=config.get('max_duration', None),
        min_duration=config.get('min_duration', None),
        max_utts=config.get('max_utts', 0),
        trim=config.get('trim_silence', False),
        use_start_end_token=config.get('use_start_end_token', True),
        return_sample_id=config.get('return_sample_id', False),
        channel_selector=config.get('channel_selector', None),
    )
    return dataset


def get_char_dataset(config: dict) -> video_to_text.VideoToCharDataset:
    """
    Instantiates a Character Encoding based VideoToCharDataset.

    Args:
        config: Config of the VideoToCharDataset.

    Returns:
        An instance of VideoToCharDataset.
    """
    if 'labels' not in config:
        logging.warning(f"dataset does not have explicitly defined labels")

    dataset = video_to_text.VideoToCharDataset(
        manifest_filepath=config['manifest_filepath'],
        labels=config.get('labels', None),
        int_values=config.get('int_values', False),
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
    config: dict, shuffle_n: int, global_rank: int, world_size: int, tokenizer: Optional['TokenizerSpec'] = None,
) -> video_to_text.TarredVideoToBPEDataset:
    """
    Instantiates a Word Piece/BPE Encoding based TarredVideoToBPEDataset or a char based TarredVideoToCharDataset.

    Args:
        config: Config of the TarredAudioToBPEDataset or TarredAudioToCharDataset.
        shuffle_n: How many samples to look ahead and load to be shuffled.
            See WebDataset documentation for more details.
        tokenizer: An instance of a TokenizerSpec object if BPE dataset is needed.
        global_rank: Global rank of this device.
        world_size: Global world size in the training method.
            Passsing None would return a char-based dataset.

    Returns:
        An instance of TarredVideoToBPEDataset or TarredVideoToCharDataset.
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
            raise Exception("video_to_text.TarredVideoToCharDataset class not Implemented")
        else:
            dataset = video_to_text.TarredVideoToBPEDataset(
                audio_tar_filepaths=tarred_audio_filepath,
                manifest_filepath=manifest_filepath,
                tokenizer=tokenizer,
                int_values=config.get('int_values', False),
                shuffle_n=shuffle_n,
                max_duration=config.get('max_duration', None),
                min_duration=config.get('min_duration', None),
                trim=config.get('trim_silence', False),
                use_start_end_token=config.get('use_start_end_token', True),
                shard_strategy=config.get('tarred_shard_strategy', 'scatter'),
                global_rank=global_rank,
                world_size=world_size,
                return_sample_id=config.get('return_sample_id', False),
            )
        if bucketing_weights:
            [datasets.append(dataset) for _ in range(bucketing_weights[dataset_idx])]
        else:
            datasets.append(dataset)

    return get_chain_dataset(datasets=datasets, ds_config=config)


def convert_to_config_list(initial_list):
    if type(initial_list) is str:
        initial_list = initial_list.split(",")
    if initial_list is None or initial_list == []:
        raise ValueError("manifest_filepaths and tarred_audio_filepaths must not be empty.")
    if not isinstance(initial_list, ListConfig):
        initial_list = ListConfig([initial_list])

    for list_idx, list_val in enumerate(initial_list):
        if type(list_val) != type(initial_list[0]):
            raise ValueError(
                "manifest_filepaths and tarred_audio_filepaths need to be a list of lists for bucketing or just a list of strings"
            )
    if type(initial_list[0]) is not ListConfig:
        initial_list = ListConfig([initial_list])
    return initial_list


def get_chain_dataset(datasets, ds_config):
    if len(datasets) > 1:
        if ds_config.get('bucketing_batch_size', None) is not None:
            bucketing_batch_sizes = calc_bucketing_batch_sizes(ds_config, len(datasets))
            logging.info(
                f"Batch bucketing is enabled for {len(datasets)} buckets with adaptive batch sizes of {bucketing_batch_sizes}!"
            )
            for idx, dataset in enumerate(datasets):
                datasets[idx] = audio_to_text.BucketingDataset(
                    dataset=dataset, bucketing_batch_size=bucketing_batch_sizes[idx]
                )
        else:
            logging.info(
                f"Batch bucketing is enabled for {len(datasets)} buckets with fixed batch size of {ds_config['batch_size']}!"
            )

    if len(datasets) == 1:
        return datasets[0]
    bucketing_strategy = ds_config.get('bucketing_strategy', 'synced_randomized')
    if bucketing_strategy == 'fixed_order':
        return ChainDataset(datasets)
    elif bucketing_strategy == 'synced_randomized':
        return audio_to_text.RandomizedChainDataset(datasets=datasets, rnd_seed=0)
    elif bucketing_strategy == 'fully_randomized':
        return audio_to_text.RandomizedChainDataset(datasets=datasets, rnd_seed=random.randint(0, 30000))
    else:
        raise ValueError(
            f'bucketing_strategy={bucketing_strategy} is not supported! Supported strategies are [fixed_order, fully_randomized, synced_randomized].'
        )


def calc_bucketing_batch_sizes(ds_config, datasets_len):
    bucketing_batch_size = ds_config['bucketing_batch_size']
    bucketing_weights = ds_config.get('bucketing_weights', None)  # To adjust for upsampled buckets

    bucketing_batch_sizes = []

    if ds_config['batch_size'] != 1:
        raise ValueError(
            f"batch_size should be set to one when bucketing_batch_size is set and adaptive bucketing is enabled (batch_size={ds_config['batch_size']}!"
        )
    if type(bucketing_batch_size) == int:  # linear scaling
        if bucketing_weights:  # Want same batchsize for the same duplicated bucket
            for idx, weight in enumerate(bucketing_weights):
                scale_factor = datasets_len - idx
                [bucketing_batch_sizes.append(scale_factor * bucketing_batch_size) for _ in range(weight)]
        else:
            for idx in range(datasets_len):
                scale_factor = datasets_len - idx
                bucketing_batch_sizes.append(scale_factor * bucketing_batch_size)
    elif isinstance(bucketing_batch_size, ListConfig) or isinstance(
        bucketing_batch_size, list
    ):  # assigned bucket sizes
        if bucketing_weights:  # Want same batchsize for same duplicated bucket
            for idx, weight in enumerate(bucketing_weights):
                [bucketing_batch_sizes.append(bucketing_batch_size[idx]) for _ in range(weight)]
        else:
            bucketing_batch_sizes = bucketing_batch_size
    else:
        raise ValueError(
            f"bucketing_batch_size should be an integer or a list (bucketing_batch_size={bucketing_batch_size})!"
        )

    if len(bucketing_batch_sizes) != datasets_len:
        raise ValueError(
            f"batch_size should have the same length as the number of buckets ({len(bucketing_batch_sizes)}!={datasets_len}) "
        )
    return bucketing_batch_sizes
