from typing import Optional

from nemo.collections.asr.data.feature_to_text import FeatureToBPEDataset, FeatureToCharDataset
from nemo.utils import logging


def get_char_dataset(config: dict, augmentor: Optional['AudioAugmentor'] = None) -> FeatureToCharDataset:
    """
    Instantiates a Character Encoding based AudioToCharDataset.

    Args:
        config: Config of the AudioToCharDataset.
        augmentor: Optional AudioAugmentor object for augmentations on audio data.

    Returns:
        An instance of AudioToCharDataset.
    """
    if 'labels' not in config:
        logging.warning(f"dataset does not have explicitly defined labels")

    dataset = FeatureToCharDataset(
        manifest_filepath=config['manifest_filepath'],
        labels=config.get('labels', None),
        normalize=config.get('normalize', 'post_norm'),
        use_rttm=config.get('use_rttm', False),
        frame_unit_time_secs=config.get('frame_unit_time_secs', 0.01),
        sample_rate=config.get('sample_rate', 16000),
        int_values=config.get('int_values', False),
        augmentor=augmentor,
        max_duration=config.get('max_duration', None),
        min_duration=config.get('min_duration', None),
        max_utts=config.get('max_utts', 0),
        blank_index=config.get('blank_index', -1),
        unk_index=config.get('unk_index', -1),
        trim=config.get('trim_silence', False),
        parser=config.get('parser', 'en'),
        return_sample_id=config.get('return_sample_id', False),
        channel_selector=config.get('channel_selector', None),
    )
    return dataset


def get_bpe_dataset(
    config: dict, tokenizer: 'TokenizerSpec', augmentor: Optional['AudioAugmentor'] = None
) -> FeatureToBPEDataset:
    """
    Instantiates a Byte Pair Encoding / Word Piece Encoding based AudioToBPEDataset.

    Args:
        config: Config of the AudioToBPEDataset.
        tokenizer: An instance of a TokenizerSpec object.
        augmentor: Optional AudioAugmentor object for augmentations on audio data.

    Returns:
        An instance of AudioToBPEDataset.
    """
    dataset = FeatureToBPEDataset(
        manifest_filepath=config['manifest_filepath'],
        tokenizer=tokenizer,
        normalize=config.get('normalize', 'post_norm'),
        use_rttm=config.get('use_rttm', False),
        frame_unit_time_secs=config.get('frame_unit_time_secs', 0.01),
        sample_rate=config.get('sample_rate', 16000),
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
