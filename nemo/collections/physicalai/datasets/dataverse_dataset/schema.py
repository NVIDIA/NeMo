# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

# pylint: disable=C0115,C0116,C0301

from typing import Optional

import attrs

from imaginaire import config
from imaginaire.config import make_freezable


@make_freezable
@attrs.define(slots=False)
class DatasetInfo:
    object_store_config: config.ObjectStoreConfig  # Object strore config
    wdinfo: list[str]  # List of wdinfo files
    opts: dict = {}  # Additional dataset info args
    per_dataset_keys: list[str] = []  # List of keys per dataset
    source: str = ""  # data source


@make_freezable
@attrs.define(slots=False)
class DatasetConfig:
    keys: list[str]  # List of keys used
    buffer_size: int  # Buffer size used by each worker
    dataset_info: list[DatasetInfo]  # List of dataset info files, one for each dataset
    decoders: list  # List of decoder functions for decoding bytestream
    streaming_download: bool = True  # Whether to use streaming loader
    remove_extension_from_keys: bool = True  # True: objects will have a key of data_type; False: data_type.extension
    sample_keys_full_list_path: Optional[str] = (
        None  # Path to the file containing all keys present in the dataset, e.g., "index"
    )
