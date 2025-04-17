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

import importlib
import logging

import numpy as np
from dataverse.datasets.base import BaseDataset
from omegaconf.dictconfig import DictConfig
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table

try:
    import torch
except ImportError:
    torch = None

logger = logging.getLogger("dataverse")
logger.setLevel(logging.INFO)
logger.addHandler(RichHandler(markup=True, rich_tracebacks=True, log_time_format="[%m-%d %H:%M:%S]"))


def dataset_from_config(config: DictConfig, **additional_params) -> BaseDataset:
    """
    Instantiates a dataset from a config.
    """
    assert "target" in config, "Expected key `target` to be the class name under dataverse.datasets."

    target = config.target
    params = config.get("params", dict())
    params.update(additional_params)

    module_name, cls_name = target.rsplit(".", 1)

    try:
        dataset_module = importlib.import_module(f"dataverse.datasets.{module_name}", package=None)
    except ImportError:
        raise ImportError(
            f"Failed to import module due to the following reasons, "
            f"which can be potentially resolved by: `pip install dataverse[{module_name}]`."
        )

    assert hasattr(dataset_module, cls_name), f"Class {cls_name} not found in module {module_name}."
    dataset_inst = getattr(dataset_module, cls_name)(**params)

    assert isinstance(dataset_inst, BaseDataset), f"{cls_name} is not a BaseDataset."

    return dataset_inst


def print_rich_dict(data_dict):
    """
    Convert a dictionary into a rich Table with columns:
    Key, Value Type, Value Shape, Device, Mean, Min, and Max (if value is a numpy or pytorch array).

    Args:
        data_dict (dict): Dictionary to convert.
    """
    # Create a Rich Table
    table = Table(title="Dictionary Details")

    # Define the columns
    table.add_column("Key", justify="left", style="cyan", no_wrap=True)
    table.add_column("Value Type", justify="left", style="magenta")
    table.add_column("Value Shape", justify="left", style="green")
    table.add_column("Device", justify="left", style="blue")
    table.add_column("Mean", justify="right", style="yellow")
    table.add_column("Min", justify="right", style="red")
    table.add_column("Max", justify="right", style="red")

    # Iterate through the dictionary
    for key, value in data_dict.items():
        value_type = str(type(value)).lstrip("<class ").rstrip(">").replace("torch.", "").strip("'")
        device = "-"
        mean = min_val = max_val = "-"

        # Check if value is a NumPy array
        if isinstance(value, np.ndarray):
            shape = str(value.shape)
            mean = f"{value.mean():.4f}"
            min_val = f"{value.min():.4f}"
            max_val = f"{value.max():.4f}"

        # Check if value is a PyTorch tensor
        elif torch is not None and isinstance(value, torch.Tensor):
            shape = str(tuple(value.shape))
            device = str(value.device)
            mean = f"{value.float().mean().item():.4f}"
            min_val = f"{value.min().item():.4f}"
            max_val = f"{value.max().item():.4f}"

        else:
            # For other data types
            shape = str(np.shape(value))

        # Add row to the table
        table.add_row(str(key), value_type, shape, device, mean, min_val, max_val)

    # Print the table
    console = Console()
    console.print(table)
