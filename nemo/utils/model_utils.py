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

import copy
import os
from pathlib import Path

from omegaconf import DictConfig, ListConfig, OmegaConf

from nemo.utils import logging

_VAL_TEST_FASTPATH_KEY = 'ds_item'


def resolve_dataset_name_from_cfg(cfg: DictConfig) -> str:
    """
    Parses items of the provided sub-config to find the first potential key that
    resolves to an existing file or directory.

    # Fast-path Resolution
    In order to handle cases where we need to resolve items that are not paths, a fastpath
    key can be provided as defined in the global `_VAL_TEST_FASTPATH_KEY`.

    This key can be used in two ways :

    ## _VAL_TEST_FASTPATH_KEY points to another key in the config

    If this _VAL_TEST_FASTPATH_KEY points to another key in this config itself,
    then we assume we want to loop through the values of that key.

    This allows for any key in the config to become a fastpath key.

    Example:
    validation_ds:
        splits: "val"
        ...
        <_VAL_TEST_FASTPATH_KEY>: "splits"  <-- this points to the key name "splits"

    Then we can write the following when overriding in hydra:
    ```python
    python train_file.py ... \
        model.validation_ds.splits=[val1, val2, dev1, dev2] ...
    ```

    ## _VAL_TEST_FASTPATH_KEY itself acts as the resolved key

    If this _VAL_TEST_FASTPATH_KEY does not point to another key in the config, then
    it is assumed that the items of this key itself are used for resolution.

    Example:
    validation_ds:
        ...
        <_VAL_TEST_FASTPATH_KEY>: "val"  <-- this points to the key name "splits"

    Then we can write the following when overriding in hydra:
    ```python
    python train_file.py ... \
        model.validation_ds.<_VAL_TEST_FASTPATH_KEY>=[val1, val2, dev1, dev2] ...
    ```

    # IMPORTANT NOTE:
    It <can> potentially mismatch if there exist more than 2 valid paths, and the
    first path does *not* resolve the the path of the data file (but does resolve to
    some other valid path).

    To avoid this side-effect, place the data path as the first item on the config file.

    Args:
        cfg: DictConfig (Sub-config) that should be parsed.

    Returns:
        A str representing the `key` of the config which hosts the filepath(s),
        or None in case path could not be resolved.
    """
    if _VAL_TEST_FASTPATH_KEY in cfg and cfg[_VAL_TEST_FASTPATH_KEY] is not None:
        fastpath_key = cfg[_VAL_TEST_FASTPATH_KEY]

        if isinstance(fastpath_key, str) and fastpath_key in cfg:
            return cfg[fastpath_key]
        else:
            return _VAL_TEST_FASTPATH_KEY

    for key, value in cfg.items():
        if type(value) in [list, tuple, ListConfig]:
            # Count the number of valid paths in the list
            values_are_paths = 0
            for val_i in value:
                val_i = str(val_i)

                if os.path.exists(val_i) or os.path.isdir(val_i):
                    values_are_paths += 1
                else:
                    # reset counter and break inner loop
                    break

            if values_are_paths == len(value):
                return key

        else:
            if os.path.exists(str(value)) or os.path.isdir(str(value)):
                return key

    return None


def parse_dataset_as_name(name: str) -> str:
    """
    Constructs a valid prefix-name from a provided file path.

    Args:
        name: str path to some valid data/manifest file or a python object that
            will be used as a name for the data loader (via str() cast).

    Returns:
        str prefix used to identify uniquely this data/manifest file.
    """
    if os.path.exists(str(name)) or os.path.isdir(str(name)):
        name = Path(name).stem
    else:
        name = str(name)

    # cleanup name
    name = name.replace('-', '_')

    if 'manifest' in name:
        name = name.replace('manifest', '')

    if 'dataset' in name:
        name = name.replace('dataset', '')

    if '_' != name[-1]:
        name = name + '_'

    return name


def resolve_validation_dataloaders(model: 'ModelPT'):
    """
    Helper method that operates on the ModelPT class to automatically support
    multiple dataloaders for the validation set.

    It does so by first resolving the path to one/more data files via `resolve_dataset_name_from_cfg()`.
    If this resolution fails, it assumes the data loader is prepared to manually support / not support
    multiple data loaders and simply calls the appropriate setup method.

    If resolution succeeds:
        Checks if provided path is to a single file or a list of files.
        If a single file is provided, simply tags that file as such and loads it via the setup method.
        If multiple files are provided:
            Inject a new manifest path at index "i" into the resolved key.
            Calls the appropriate setup method to set the data loader.
            Collects the initialized data loader in a list and preserves it.
            Once all data loaders are processed, assigns the list of loaded loaders to the ModelPT.
            Finally assigns a list of unique names resolved from the file paths to the ModelPT.

    Args:
        model: ModelPT subclass, which requires >=1 Validation Dataloaders to be setup.
    """
    cfg = copy.deepcopy(model._cfg)
    dataloaders = []

    # process val_loss_idx
    if 'val_loss_idx' in cfg.validation_ds:
        cfg = OmegaConf.to_container(cfg)
        val_loss_idx = cfg['validation_ds'].pop('val_loss_idx')
        cfg = OmegaConf.create(cfg)
    else:
        val_loss_idx = 0

    # Set val_loss_idx
    model._validation_loss_idx = val_loss_idx

    ds_key = resolve_dataset_name_from_cfg(cfg.validation_ds)

    if ds_key is None:
        logging.debug(
            "Could not resolve file path from provided config - {}. "
            "Disabling support for multi-dataloaders.".format(cfg.validation_ds)
        )

        model.setup_validation_data(cfg.validation_ds)
        return

    ds_values = cfg.validation_ds[ds_key]

    if isinstance(ds_values, (list, tuple, ListConfig)):

        for ds_value in ds_values:
            cfg.validation_ds[ds_key] = ds_value
            model.setup_validation_data(cfg.validation_ds)
            dataloaders.append(model._validation_dl)

        model._validation_dl = dataloaders
        model._validation_names = [parse_dataset_as_name(ds) for ds in ds_values]

        # In fast-dev-run, only one data loader is used
        if model._trainer.fast_dev_run:
            model._validation_dl = model._validation_dl[:1]
            model._validation_names = model._validation_names[:1]

        return

    else:
        model.setup_validation_data(cfg.validation_ds)
        model._validation_names = [parse_dataset_as_name(ds_values)]


def resolve_test_dataloaders(model: 'ModelPT'):
    """
    Helper method that operates on the ModelPT class to automatically support
    multiple dataloaders for the test set.

    It does so by first resolving the path to one/more data files via `resolve_dataset_name_from_cfg()`.
    If this resolution fails, it assumes the data loader is prepared to manually support / not support
    multiple data loaders and simply calls the appropriate setup method.

    If resolution succeeds:
        Checks if provided path is to a single file or a list of files.
        If a single file is provided, simply tags that file as such and loads it via the setup method.
        If multiple files are provided:
            Inject a new manifest path at index "i" into the resolved key.
            Calls the appropriate setup method to set the data loader.
            Collects the initialized data loader in a list and preserves it.
            Once all data loaders are processed, assigns the list of loaded loaders to the ModelPT.
            Finally assigns a list of unique names resolved from the file paths to the ModelPT.

    Args:
        model: ModelPT subclass, which requires >=1 Test Dataloaders to be setup.
    """
    cfg = copy.deepcopy(model._cfg)
    dataloaders = []

    # process test_loss_idx
    if 'test_loss_idx' in cfg.test_ds:
        cfg = OmegaConf.to_container(cfg)
        test_loss_idx = cfg['test_ds'].pop('test_loss_idx')
        cfg = OmegaConf.create(cfg)
    else:
        test_loss_idx = 0

    # Set val_loss_idx
    model._test_loss_idx = test_loss_idx

    ds_key = resolve_dataset_name_from_cfg(cfg.test_ds)

    if ds_key is None:
        logging.debug(
            "Could not resolve file path from provided config - {}. "
            "Disabling support for multi-dataloaders.".format(cfg.test_ds)
        )

        model.setup_test_data(cfg.test_ds)
        return

    ds_values = cfg.test_ds[ds_key]

    if isinstance(ds_values, (list, tuple, ListConfig)):

        for ds_value in ds_values:
            cfg.test_ds[ds_key] = ds_value
            model.setup_test_data(cfg.test_ds)
            dataloaders.append(model._test_dl)

        model._test_dl = dataloaders
        model._test_names = [parse_dataset_as_name(ds) for ds in ds_values]

        # In fast-dev-run, only one data loader is used
        if model._trainer.fast_dev_run:
            model._test_dl = model._test_dl[:1]
            model._test_names = model._test_names[:1]

    else:
        model.setup_test_data(cfg.test_ds)
        model._test_names = [parse_dataset_as_name(ds_values)]
