# =============================================================================
# Copyright (c) 2020 NVIDIA. All Rights Reserved.
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
# =============================================================================

from os.path import expanduser
from typing import Any, Dict

import torch

# This should go to settings/NeMo config file.


def save(checkpoint: Dict[str, Any], filename: str) -> None:
    """
    A proxy function that saves the checkpoint to a given file.

    Args:
        checkpoint: Checkpoint to be stored.
        filename: Name of the file containing checkpoint.
    """
    # Get the absolute path and save.
    abs_filename = expanduser(filename)
    torch.save(checkpoint, abs_filename)


def load(filename: str) -> Dict[str, Any]:
    """
    A proxy function that loads checkpoint from a given file.

    Args:
        filename: Name of the file containing checkpoint.
    Returns:
        Loaded checkpoint.
    """
    # Get the absolute path and save.
    abs_filename = expanduser(filename)
    # Use map location to be able to load CUDA-trained modules on CPU.
    return torch.load(abs_filename, map_location=lambda storage, loc: storage)


def get_state_dict(model: torch.nn.Module) -> Dict[str, Any]:
    """
    A proxy function that gets the state dictionary.

    Args:
        model: Torch model.
    Returns:
        State dictionary containing model weights.
    """
    return model.state_dict()


def set_state_dict(model: torch.nn.Module, state_dict: Dict[str, Any]) -> None:
    """
    A proxy function that sets the state dictionary.

    Args:
        model: Torch model.
        state_dict: State dictionary containing model weights.
    """
    model.load_state_dict(state_dict)
