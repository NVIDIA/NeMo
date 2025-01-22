# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

import logging
from pathlib import Path
from typing import Any, Dict, Union

import torch
from torch.distributed.checkpoint import FileSystemReader, load
from torch.distributed.checkpoint.metadata import BytesStorageMetadata, TensorStorageMetadata
from torch.distributed.checkpoint.state_dict_loader import load_state_dict

from nemo.export.tarutils import TarPath


LOGGER = logging.getLogger("NeMo")
EXTRA_STATE = "extra_state"


class TarFileSystemReader(FileSystemReader):
    """Reader that accepts both Path and TarPath checkpoint directory.

    The FileSystemReader works with TarPath, but expects a pure Path.
    It's enough to skip the Path check in __init__.
    """

    def __init__(self, path: Union[Path, TarPath]) -> None:
        """Makes sure that super().__init__ gets a pure path as expected."""
        super_path = str(path) if isinstance(path, TarPath) else path
        super().__init__(super_path)
        if isinstance(path, TarPath):
            self.path = path  # overwrites path set in super().__init__ call


def load_sharded_metadata_torch_dist(checkpoint_dir: Union[Path, TarPath]) -> Dict[str, Any]:
    """
    Loads model state dictionary from torch_dist checkpoint.

    Args:
        checkpoint_dir (Path | TarPath): Path to the model weights directory.
    Returns:
        dict: Loaded model state dictionary.
    """
    fs_reader = TarFileSystemReader(checkpoint_dir)
    metadata = fs_reader.read_metadata()

    state_dict = {
        k: torch.empty(tp.size, dtype=tp.properties.dtype)
        for k, tp in metadata.state_dict_metadata.items()
        if isinstance(tp, TensorStorageMetadata)
    }

    state_dict.update(
        {k: [] for k, tp in metadata.state_dict_metadata.items() if isinstance(tp, BytesStorageMetadata)}
    )

    load(state_dict, storage_reader=fs_reader)
    return state_dict
