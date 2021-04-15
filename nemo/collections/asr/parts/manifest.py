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

import json
from os.path import expanduser, split, join
from typing import Any, Callable, Dict, Iterator, List, Optional, Union


class ManifestBase:
    def __init__(self, *args, **kwargs):
        raise ValueError(
            "This class is deprecated, look at https://github.com/NVIDIA/NeMo/pull/284 for correct behaviour."
        )


class ManifestEN:
    def __init__(self, *args, **kwargs):
        raise ValueError(
            "This class is deprecated, look at https://github.com/NVIDIA/NeMo/pull/284 for correct behaviour."
        )


def item_iter(
    manifests_files: Union[str, List[str]], parse_func: Callable[[str, Optional[str]], Dict[str, Any]] = None
) -> Iterator[Dict[str, Any]]:
    """Iterate through json lines of provided manifests.

    NeMo ASR pipelines often assume certain manifest files structure. In
    particular, each manifest file should consist of line-per-sample files with
    each line being correct json dict. Each such json dict should have a field
    for audio file string, a field for duration float and a field for text
    string. Offset also could be additional field and is set to None by
    default.

    Args:
        manifests_files: Either single string file or list of such -
            manifests to yield items from.

        parse_func: A callable function which accepts as input a single line
            of a manifest and optionally the manifest file itself,
            and parses it, returning a dictionary mapping from str -> Any.

    Yields:
        Parsed key to value item dicts.

    Raises:
        ValueError: If met invalid json line structure.
    """

    if isinstance(manifests_files, str):
        manifests_files = [manifests_files]

    if parse_func is None:
        parse_func = __parse_item
    elif parse_func == "open_stt":
        parse_func = __parse_open_stt

    k = -1
    for manifest_file in manifests_files:
        with open(expanduser(manifest_file), 'r') as f:
            for line in f:
                k += 1
                item = parse_func(line, manifest_file)
                item['id'] = k

                yield item

def __parse_open_stt(line: str, manifest_file: str) -> Dict[str, Any]:
    manifest_path, _ = split(manifest_file)
    line_list = line.split(",")
    assert len(line_list)==3, f"Incorrect fields number: {len(line_list)}"
    item = dict(
        offset=None,
        speaker=None,
        orig_sr=None,
    )

    item['audio_file'] = line_list[0]
    if item['audio_file'][0] != "/":
        item['audio_file'] = join(manifest_path, item['audio_file'])

    text_file = line_list[1]
    if text_file[0] != "/":
        text_file = join(manifest_path, text_file)
    with open(text_file, 'r') as f:
            item['text'] = f.read().replace('\n', '')

    item['duration'] = float(line_list[2])

    return item

def __parse_item(line: str, manifest_file: str) -> Dict[str, Any]:
    item = json.loads(line)
    manifest_path, _ = split(manifest_file)
    # Audio file
    if 'audio_filename' in item:
        item['audio_file'] = item.pop('audio_filename')
    elif 'audio_filepath' in item:
        item['audio_file'] = item.pop('audio_filepath')
    else:
        raise ValueError(
            f"Manifest file {manifest_file} has invalid json line structure: {line} without proper audio file key."
        )
    item['audio_file'] = expanduser(item['audio_file'])
    if item['audio_file'][0] != "/":
        item['audio_file'] = join(manifest_path, item['audio_file'])

    # Duration.
    if 'duration' not in item:
        raise ValueError(
            f"Manifest file {manifest_file} has invalid json line structure: {line} without proper duration key."
        )

    # Text.
    if 'text' in item:
        pass
    elif 'text_filepath' in item:
        if item['text_filepath'][0] != "/":
            item['text_filepath'] = join(manifest_path, item['text_filepath'])
        with open(item.pop('text_filepath'), 'r', encoding='utf8') as f:
            item['text'] = f.read().replace('\n', '')
    elif 'normalized_text' in item:
        item['text'] = item['normalized_text']
    else:
        raise ValueError(
            f"Manifest file {manifest_file} has invalid json line structure: {line} without proper text key."
        )

    item = dict(
        audio_file=item['audio_file'],
        duration=item['duration'],
        text=item['text'],
        offset=item.get('offset', None),
        speaker=item.get('speaker', None),
        orig_sr=item.get('orig_sample_rate', None),
    )

    return item
