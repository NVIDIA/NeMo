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

from nemo.collections.asr.data import audio_to_label


def get_classification_label_dataset(featurizer, config: dict) -> audio_to_label.AudioToClassificationLabelDataset:
    """
    Instantiates a Classification AudioLabelDataset.

    Args:
        config: Config of the AudioToClassificationLabelDataset.

    Returns:
        An instance of AudioToClassificationLabelDataset.
    """
    dataset = audio_to_label.AudioToClassificationLabelDataset(
        manifest_filepath=config['manifest_filepath'],
        labels=config['labels'],
        featurizer=featurizer,
        max_duration=config.get('max_duration', None),
        min_duration=config.get('min_duration', None),
        trim=config.get('trim_silence', False),
        is_regression_task=config.get('is_regression_task', False),
    )
    return dataset


def get_speech_label_dataset(featurizer, config: dict) -> audio_to_label.AudioToSpeechLabelDataset:
    """
    Instantiates a Speech Label (e.g. VAD, speaker recognition) AudioLabelDataset.

    Args:
        config: Config of the AudioToSpeechLabelDataSet.

    Returns:
        An instance of AudioToSpeechLabelDataset.
    """
    dataset = audio_to_label.AudioToSpeechLabelDataset(
        manifest_filepath=config['manifest_filepath'],
        labels=config['labels'],
        featurizer=featurizer,
        max_duration=config.get('max_duration', None),
        min_duration=config.get('min_duration', None),
        trim=config.get('trim_silence', False),
        time_length=config.get('time_length', 0.31),
        shift_length=config.get('shift_length', 0.01),
        normalize_audio=config.get('normalize_audio', False),
    )
    return dataset


def get_tarred_classification_label_dataset(
    featurizer, config: dict, shuffle_n: int, global_rank: int, world_size: int
) -> audio_to_label.TarredAudioToClassificationLabelDataset:
    """
    Instantiates a Classification TarredAudioLabelDataset.

    Args:
        config: Config of the TarredAudioToClassificationLabelDataset.
        shuffle_n: How many samples to look ahead and load to be shuffled.
            See WebDataset documentation for more details.
        global_rank: Global rank of this device.
        world_size: Global world size in the training method.

    Returns:
        An instance of TarredAudioToClassificationLabelDataset.
    """
    dataset = audio_to_label.TarredAudioToClassificationLabelDataset(
        audio_tar_filepaths=config['tarred_audio_filepaths'],
        manifest_filepath=config['manifest_filepath'],
        labels=config['labels'],
        featurizer=featurizer,
        shuffle_n=shuffle_n,
        max_duration=config.get('max_duration', None),
        min_duration=config.get('min_duration', None),
        trim=config.get('trim_silence', False),
        shard_strategy=config.get('tarred_shard_strategy', 'scatter'),
        global_rank=global_rank,
        world_size=world_size,
        is_regression_task=config.get('is_regression_task', False),
    )
    return dataset


def get_tarred_speech_label_dataset(
    featurizer, config: dict, shuffle_n: int, global_rank: int, world_size: int,
) -> audio_to_label.TarredAudioToSpeechLabelDataset:
    """
    InInstantiates a Speech Label (e.g. VAD, speaker recognition) TarredAudioLabelDataset.

    Args:
        config: Config of the TarredAudioToSpeechLabelDataset.
        shuffle_n: How many samples to look ahead and load to be shuffled.
            See WebDataset documentation for more details.
        global_rank: Global rank of this device.
        world_size: Global world size in the training method.

    Returns:
        An instance of TarredAudioToSpeechLabelDataset.
    """
    dataset = audio_to_label.TarredAudioToSpeechLabelDataset(
        audio_tar_filepaths=config['tarred_audio_filepaths'],
        manifest_filepath=config['manifest_filepath'],
        labels=config['labels'],
        featurizer=featurizer,
        shuffle_n=shuffle_n,
        max_duration=config.get('max_duration', None),
        min_duration=config.get('min_duration', None),
        trim=config.get('trim_silence', False),
        time_length=config.get('time_length', 0.31),
        shift_length=config.get('shift_length', 0.01),
        normalize_audio=config.get('normalize_audio', False),
        shard_strategy=config.get('tarred_shard_strategy', 'scatter'),
        global_rank=global_rank,
        world_size=world_size,
    )
    return dataset
