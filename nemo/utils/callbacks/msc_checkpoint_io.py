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

from pathlib import Path
from typing import Any, Callable, Dict, Optional, Union

from lightning.fabric.plugins.io.checkpoint_io import CheckpointIO
from lightning.fabric.utilities.types import _PATH

import multistorageclient as msc
from multistorageclient.types import MSC_PROTOCOL
from nemo.utils import logging


class MSCCheckpointIO(CheckpointIO):
    """A custom MSCCheckpointIO module that supports checkpoint reading/writing with multi-storage client when filepath
    is a MSC url.
    """

    def __init__(
        self,
        dirpath: str,
    ):
        """
        Initialize the MSC checkpoint IO.

        Args:
            dirpath (str): The directory path for checkpoints
        """
        if not dirpath.startswith(MSC_PROTOCOL):
            raise AssertionError(
                f"Error attempting to initialize an MSCCheckpointIO when {dirpath} is not an MSC url."
            )

        super().__init__()

    def save_checkpoint(
        self, checkpoint: Dict[str, Any], path: Union[str, Path], storage_options: Optional[Any] = None
    ) -> None:
        """Save checkpoint to MSC storage.

        Args:
            checkpoint: dict containing model and trainer state
            path: write-target path
            storage_options: not used in MSCCheckpointIO
        """
        if storage_options is not None and len(storage_options) > 0:
            logging.warning(
                f"{self.__class__.__name__} does not support storage_options, but {storage_options=} was provided."
                f" Ignoring given storage_options"
            )

        msc.torch.save(checkpoint, path)

    def load_checkpoint(
        self, path: Union[str, Path], map_location: Optional[Callable] = lambda storage, loc: storage
    ) -> Dict[str, Any]:
        """Load checkpoint from MSC storage.

        Args:
            path: Path to checkpoint
            map_location: a function, torch.device, string or a dict specifying how to remap storage locations

        Returns:
            The loaded checkpoint
        """
        checkpoint = msc.torch.load(path, map_location=map_location)
        return checkpoint

    def remove_checkpoint(self, path: Union[str, Path]) -> None:
        """Remove checkpoint from MSC storage.

        Args:
            path: Path to checkpoint
        """
        #TODO: add remove function to shortcuts.py then use it here
        pass
