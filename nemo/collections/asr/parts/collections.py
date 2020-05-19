# Copyright (c) 2019 NVIDIA Corporation
import collections
import json
import os
from typing import Any, Dict, List, Optional, Union

import pandas as pd

from nemo.collections.asr.parts import manifest, parsers
from nemo.utils import logging


class _Collection(collections.UserList):
    """List of parsed and preprocessed data."""

    OUTPUT_TYPE = None  # Single element output type.


class Text(_Collection):
    """Simple list of preprocessed text entries, result in list of tokens."""

    OUTPUT_TYPE = collections.namedtuple('TextEntity', 'tokens')

    def __init__(self, texts: List[str], parser: parsers.CharParser):
        """Instantiates text manifest and do the preprocessing step.

        Args:
            texts: List of raw texts strings.
            parser: Instance of `CharParser` to convert string to tokens.
        """

        data, output_type = [], self.OUTPUT_TYPE
        for text in texts:
            tokens = parser(text)

            if tokens is None:
                logging.warning("Fail to parse '%s' text line.", text)
                continue

            data.append(output_type(tokens))

        super().__init__(data)


class FromFileText(Text):
    """Another form of texts manifest with reading from file."""

    def __init__(self, file: str, parser: parsers.CharParser):
        """Instantiates text manifest and do the preprocessing step.

        Args:
            file: File path to read from.
            parser: Instance of `CharParser` to convert string to tokens.
        """

        texts = self.__parse_texts(file)

        super().__init__(texts, parser)

    @staticmethod
    def __parse_texts(file: str) -> List[str]:
        if not os.path.exists(file):
            raise ValueError('Provided texts file does not exists!')

        _, ext = os.path.splitext(file)
        if ext == '.csv':
            texts = pd.read_csv(file)['transcript'].tolist()
        elif ext == '.json':  # Not really a correct json.
            texts = list(item['text'] for item in manifest.item_iter(file))
        else:
            with open(file, 'r') as f:
                texts = f.readlines()

        return texts


class AudioText(_Collection):
    """List of audio-transcript text correspondence with preprocessing."""

    OUTPUT_TYPE = collections.namedtuple(
        typename='AudioTextEntity', field_names='audio_file duration text_tokens offset',
    )

    def __init__(
        self,
        audio_files: List[str],
        durations: List[float],
        texts: List[str],
        offsets: List[str],
        parser: parsers.CharParser,
        min_duration: Optional[float] = None,
        max_duration: Optional[float] = None,
        max_number: Optional[int] = None,
        do_sort_by_duration: bool = False,
        index_by_file_id: bool = False,
    ):
        """Instantiates audio-text manifest with filters and preprocessing.

        Args:
            audio_files: List of audio files.
            durations: List of float durations.
            texts: List of raw text transcripts.
            offsets: List of duration offsets or None.
            parser: Instance of `CharParser` to convert string to tokens.
            min_duration: Minimum duration to keep entry with (default: None).
            max_duration: Maximum duration to keep entry with (default: None).
            max_number: Maximum number of samples to collect.
            do_sort_by_duration: True if sort samples list by duration. Not compatible with index_by_file_id.
            index_by_file_id: If True, saves a mapping from filename base (ID) to index in data.
        """

        output_type = self.OUTPUT_TYPE
        data, duration_filtered, num_filtered, total_duration = [], 0.0, 0, 0.0
        if index_by_file_id:
            self.mapping = {}

        for audio_file, duration, text, offset in zip(audio_files, durations, texts, offsets):
            # Duration filters.
            if min_duration is not None and duration < min_duration:
                duration_filtered += duration
                num_filtered += 1
                continue

            if max_duration is not None and duration > max_duration:
                duration_filtered += duration
                num_filtered += 1
                continue

            text_tokens = parser(text)
            if text_tokens is None:
                duration_filtered += duration
                num_filtered += 1
                continue

            total_duration += duration

            data.append(output_type(audio_file, duration, text_tokens, offset))
            if index_by_file_id:
                file_id, _ = os.path.splitext(os.path.basename(audio_file))
                self.mapping[file_id] = len(data) - 1

            # Max number of entities filter.
            if len(data) == max_number:
                break

        if do_sort_by_duration:
            if index_by_file_id:
                logging.warning("Tried to sort dataset by duration, but cannot since index_by_file_id is set.")
            else:
                data.sort(key=lambda entity: entity.duration)

        logging.info("Dataset loaded with %d files totalling %.2f hours", len(data), total_duration / 3600)
        logging.info("%d files were filtered totalling %.2f hours", num_filtered, duration_filtered / 3600)

        super().__init__(data)


class ASRAudioText(AudioText):
    """`AudioText` collector from asr structured json files."""

    def __init__(self, manifests_files: Union[str, List[str]], *args, **kwargs):
        """Parse lists of audio files, durations and transcripts texts.

        Args:
            manifests_files: Either single string file or list of such -
                manifests to yield items from.
            *args: Args to pass to `AudioText` constructor.
            **kwargs: Kwargs to pass to `AudioText` constructor.
        """

        audio_files, durations, texts, offsets = [], [], [], []
        for item in manifest.item_iter(manifests_files):
            audio_files.append(item['audio_file'])
            durations.append(item['duration'])
            texts.append(item['text'])
            offsets.append(item['offset'])

        super().__init__(audio_files, durations, texts, offsets, *args, **kwargs)


class SpeechLabel(_Collection):
    """List of audio-label correspondence with preprocessing."""

    OUTPUT_TYPE = collections.namedtuple(typename='SpeechLabelEntity', field_names='audio_file duration label offset',)

    def __init__(
        self,
        audio_files: List[str],
        durations: List[float],
        labels: List[Union[int, str]],
        offsets: List[Optional[float]],
        min_duration: Optional[float] = None,
        max_duration: Optional[float] = None,
        max_number: Optional[int] = None,
        do_sort_by_duration: bool = False,
    ):
        """Instantiates audio-label manifest with filters and preprocessing.

        Args:
            audio_files: List of audio files.
            durations: List of float durations.
            labels: List of labels.
            offsets: List of offsets or None.
            min_duration: Minimum duration to keep entry with (default: None).
            max_duration: Maximum duration to keep entry with (default: None).
            max_number: Maximum number of samples to collect.
            do_sort_by_duration: True if sort samples list by duration.
        """

        output_type = self.OUTPUT_TYPE
        data, duration_filtered = [], 0.0
        for audio_file, duration, command, offset in zip(audio_files, durations, labels, offsets):
            # Duration filters.
            if min_duration is not None and duration < min_duration:
                duration_filtered += duration
                continue

            if max_duration is not None and duration > max_duration:
                duration_filtered += duration
                continue

            data.append(output_type(audio_file, duration, command, offset))

            # Max number of entities filter.
            if len(data) == max_number:
                break

        if do_sort_by_duration:
            data.sort(key=lambda entity: entity.duration)

        logging.info(
            "Filtered duration for loading collection is %f.", duration_filtered,
        )
        self.uniq_labels = sorted(set(map(lambda x: x.label, data)))
        logging.info("# {} files loaded accounting to # {} labels".format(len(data), len(self.uniq_labels)))

        super().__init__(data)


class ASRSpeechLabel(SpeechLabel):
    """`SpeechLabel` collector from structured json files."""

    def __init__(self, manifests_files: Union[str, List[str]], *args, **kwargs):
        """Parse lists of audio files, durations and transcripts texts.

        Args:
            manifests_files: Either single string file or list of such -
                manifests to yield items from.
            *args: Args to pass to `SpeechLabel` constructor.
            **kwargs: Kwargs to pass to `SpeechLabel` constructor.
        """
        audio_files, durations, labels, offsets = [], [], [], []

        for item in manifest.item_iter(manifests_files, parse_func=self.__parse_item):
            audio_files.append(item['audio_file'])
            durations.append(item['duration'])
            labels.append(item['label'])
            offsets.append(item['offset'])

        super().__init__(audio_files, durations, labels, offsets, *args, **kwargs)

    def __parse_item(self, line: str, manifest_file: str) -> Dict[str, Any]:
        item = json.loads(line)

        # Audio file
        if 'audio_filename' in item:
            item['audio_file'] = item.pop('audio_filename')
        elif 'audio_filepath' in item:
            item['audio_file'] = item.pop('audio_filepath')
        else:
            raise ValueError(
                f"Manifest file has invalid json line " f"structure: {line} without proper audio file key."
            )
        item['audio_file'] = os.path.expanduser(item['audio_file'])

        # Duration.
        if 'duration' not in item:
            raise ValueError(f"Manifest file has invalid json line " f"structure: {line} without proper duration key.")

        # Label.
        if 'command' in item:
            item['label'] = item.pop('command')
        elif 'target' in item:
            item['label'] = item.pop('target')
        elif 'label' in item:
            pass
        else:
            raise ValueError(f"Manifest file has invalid json line " f"structure: {line} without proper label key.")

        item = dict(
            audio_file=item['audio_file'],
            duration=item['duration'],
            label=item['label'],
            offset=item.get('offset', None),
        )

        return item
