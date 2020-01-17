# Copyright (c) 2019 NVIDIA Corporation
import json
from typing import Union, Iterator, Dict, Any, List


def item_iter(
    manifests_files: Union[str, List[str]]
) -> Iterator[Dict[str, Any]]:
    """Iterate through json lines of provided manifests.

    Args:
        manifests_files: Either single string file or list of such -
            manifests to yield items from.

    Yields:
        Parsed key to value item dicts.

    Raises:
        ValueError: If met invalid json line structure.
    """

    if isinstance(manifests_files, str):
        manifests_files = [manifests_files]

    for manifest_file in manifests_files:
        with open(manifest_file, 'r') as f:
            for line in f:
                item = json.loads(line)

                # Audio file
                if 'audio_filename' in item:
                    item['audio_file'] = item.pop('audio_filename')
                elif 'audio_filepath' in item:
                    item['audio_file'] = item.pop('audio_filepath')
                else:
                    raise ValueError(
                        f"Manifest file {manifest_file} has invalid json line "
                        f"structure: {line} without proper audio file key."
                    )

                # Duration.
                if 'duration' not in item:
                    raise ValueError(
                        f"Manifest file {manifest_file} has invalid json line "
                        f"structure: {line} without proper duration key."
                    )

                # Text.
                if 'text' in item:
                    pass
                elif 'text_filepath' in item:
                    item['text'] = __load_text(item.pop('text_filepath'))
                else:
                    raise ValueError(
                        f"Manifest file {manifest_file} has invalid json line "
                        f"structure: {line} without proper text key."
                    )

                item = dict(
                    audio_file=item['audio_file'],
                    duration=item['duration'],
                    text=item['text'],
                )

                yield item


def __load_text(file):
    with open(file, 'r') as f:
        return f.read().replace('\n', '')
