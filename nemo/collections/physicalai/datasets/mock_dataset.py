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

import inspect
from typing import Any, Callable, Dict

import torch
from torch.utils.data import Dataset

MAX_LENGTH = 1 << 15


class LambdaDataset(torch.utils.data.Dataset):
    """
    A dataset that generates items by applying a function. This allows for creating
    dynamic datasets where the items are the result of function calls. The function can optionally
    accept an index argument.

    Attributes:
        length (int): The total number of items in the dataset.
        fn (Callable): The function to generate dataset items.
        is_index_in_params (bool): Flag to determine whether 'index' should be passed
                                   to the function `fn`.
    """

    def __init__(self, fn: Callable, length: int = MAX_LENGTH) -> None:
        """
        Initializes the LambdaDataset with a function and the total length.

        Args:
            fn (Callable): A function that returns a dataset item. It can optionally accept an
                           index argument to generate data items based on their index.
            length (int): The total number of items in the dataset, defaults to MAX_LENGTH.
        """
        self.length = length
        self.fn = fn

        try:
            # Attempt to inspect the function signature to determine if it accepts an 'index' parameter.
            signature = inspect.signature(fn)
            self.is_index_in_params = "index" in signature.parameters
        except ValueError:
            # If the function signature is not inspectable, assume 'index' is not a parameter.
            self.is_index_in_params = False

    def __len__(self) -> int:
        """
        Returns the total length of the dataset.

        Returns:
            int: The number of items in the dataset.
        """
        return self.length

    def __getitem__(self, index: int) -> Any:
        """
        Retrieves an item at a specific index from the dataset by calling the function `fn`.
        Passes the index to `fn` if `fn` is designed to accept an index.

        Args:
            index (int): The index of the item to retrieve.

        Returns:
            Any: The item returned by the function `fn`.
        """
        if self.is_index_in_params:
            return self.fn(index)  # Call fn with index if it accepts an index parameter.
        return self.fn()  # Call fn without any parameters if it does not accept the index.


class RepeatDataset(torch.utils.data.Dataset):
    """
    A dataset wrapper that allows repeating access to items from an underlying dataset.

    This dataset can be used to create an artificial extension of the underlying dataset
    to a specified `length`. Each item from the original dataset can be accessed
    repeatedly up to `num_item` times before it loops back.

    Attributes:
        length (int): The total length of the dataset to be exposed.
        dataset (Dataset): The original dataset.
        num_item (int): Number of times each item is repeated.
        cache_item (dict): Cache to store accessed items to avoid recomputation.
    """

    def __init__(self, dataset: Dataset, length: int = MAX_LENGTH, num_item: int = 1) -> None:
        """
        Initializes the RepeatDataset with a dataset, length, and number of repeats per item.

        Args:
            dataset (Dataset): The dataset to repeat.
            length (int): The total length of the dataset to be exposed. Defaults to MAX_LENGTH.
            num_item (int): The number of times to repeat each item. Defaults to 1.
        """
        self.length = length
        self.dataset = dataset
        self.num_item = num_item
        self.cache_item = {}

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> Any:
        index = index % self.num_item
        if index not in self.cache_item:
            self.cache_item[index] = self.dataset[index]
        return self.cache_item[index]


class CombinedDictDataset(torch.utils.data.Dataset):
    """
    A dataset that wraps multiple PyTorch datasets and returns a dictionary of data items from each dataset for a given index.
    This dataset ensures that all constituent datasets have the same length by setting the length to the minimum length of the datasets provided.

    Parameters:
    -----------
    **datasets : Dict[str, Dataset]
        A dictionary where keys are string identifiers for the datasets and values are the datasets instances themselves.

    Attributes:
    -----------
    datasets : Dict[str, Dataset]
        Stores the input datasets.
    max_length : int
        The minimum length among all provided datasets, determining the length of this combined dataset.

    Examples:
    ---------
    >>> dataset1 = torch.utils.data.TensorDataset(torch.randn(100, 3, 32, 32))
    >>> dataset2 = torch.utils.data.TensorDataset(torch.randn(100, 3, 32, 32))
    >>> combined_dataset = CombinedDictDataset(dataset1=dataset1, dataset2=dataset2)
    >>> print(len(combined_dataset))
    100
    >>> data = combined_dataset[50]
    >>> print(data.keys())
    dict_keys(['dataset1', 'dataset2'])
    """

    def __init__(self, **datasets: Dict[str, Dataset]) -> None:
        """
        Initializes the CombinedDictDataset with multiple datasets.

        Args:
            **datasets (Dict[str, Dataset]): Key-value pairs where keys are dataset names and values
                                             are dataset instances. Each key-value pair adds a dataset
                                             under the specified key.
        """
        self.datasets = datasets
        self.max_length = min([len(dataset) for dataset in datasets.values()])

    def __len__(self) -> int:
        return self.max_length

    def __getitem__(self, index: int) -> Dict[str, Any]:
        """
        Retrieves an item from each dataset at the specified index, combines them into a dictionary,
        and returns the dictionary. Each key in the dictionary corresponds to one of the dataset names provided
        during initialization, and its value is the item from that dataset at the given index.

        Args:
            index (int): The index of the items to retrieve across all datasets.

        Returns:
            Dict[str, Any]: A dictionary containing data items from all datasets for the given index.
                            Each key corresponds to a dataset name, and its value is the data item from that dataset.
        """
        data = {}
        for key, dataset in self.datasets.items():
            data[key] = dataset[index]
        return data
