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

from nemo import logging


def resolve_filepath_from_cfg(cfg: DictConfig) -> str:
    """
    Parses items of the provided sub-config to find the first potential key that
    resolves to an existing file or directory.

    NOTE:
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
    if hasattr(cfg, 'manifest_filepath'):
        return 'manifest_filepath'

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


def parse_filepath_as_name(filepath: str) -> str:
    """
    Constructs a valid prefix-name from a provided file path.

    Args:
        filepath: str path to some valid data/manifest file.

    Returns:
        str prefix used to identify uniquely this data/manifest file.
    """
    filename = Path(filepath).stem

    # cleanup name
    filename = filename.replace('-', '_')

    if 'manifest' in filename:
        filename = filename.replace('manifest', '')

    if '_' != filename[-1]:
        filename = filename + '_'

    return filename


def resolve_validation_dataloaders(model: 'ModelPT'):
    """
    Helper method that operates on the ModelPT class to automatically support
    multiple dataloaders for the validation set.

    It does so by first resolving the path to one/more data files via `resolve_filepath_from_cfg()`.
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
    if hasattr(cfg.validation_ds, 'val_loss_idx'):
        cfg = OmegaConf.to_container(cfg)
        val_loss_idx = cfg['validation_ds'].pop('val_loss_idx')
        cfg = OmegaConf.create(cfg)
    else:
        val_loss_idx = 0

    # Set val_loss_idx
    model._validation_loss_idx = val_loss_idx

    filepath_key = resolve_filepath_from_cfg(cfg.validation_ds)

    if filepath_key is None:
        logging.debug(
            "Could not resolve file path from provided config - {}. "
            "Disabling support for multi-dataloaders.".format(cfg.validation_ds)
        )

        model.setup_validation_data(cfg.validation_ds)
        model._validation_filenames = ["validation_"]
        return

    manifest_paths = cfg.validation_ds[filepath_key]

    if type(manifest_paths) in (list, tuple, ListConfig):

        for filepath_val in manifest_paths:
            cfg.validation_ds[filepath_key] = filepath_val
            model.setup_validation_data(cfg.validation_ds)
            dataloaders.append(model._validation_dl)

        model._validation_dl = dataloaders
        model._validation_filenames = [parse_filepath_as_name(fp) for fp in manifest_paths]

        # In fast-dev-run, only one data loader is used
        if model._trainer.fast_dev_run:
            model._validation_dl = model._validation_dl[:1]
            model._validation_filenames = model._validation_filenames[:1]

        return

    else:
        model.setup_validation_data(cfg.validation_ds)
        model._validation_filenames = [parse_filepath_as_name(manifest_paths)]


def resolve_test_dataloaders(model: 'ModelPT'):
    """
    Helper method that operates on the ModelPT class to automatically support
    multiple dataloaders for the test set.

    It does so by first resolving the path to one/more data files via `resolve_filepath_from_cfg()`.
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
    if hasattr(cfg.test_ds, 'test_loss_idx'):
        cfg = OmegaConf.to_container(cfg)
        test_loss_idx = cfg['test_ds'].pop('test_loss_idx')
        cfg = OmegaConf.create(cfg)
    else:
        test_loss_idx = 0

    # Set val_loss_idx
    model._test_loss_idx = test_loss_idx

    filepath_key = resolve_filepath_from_cfg(cfg.test_ds)

    if filepath_key is None:
        logging.debug(
            "Could not resolve file path from provided config - {}. "
            "Disabling support for multi-dataloaders.".format(cfg.test_ds)
        )

        model.setup_test_data(cfg.test_ds)
        model._test_filenames = ["test_"]
        return

    manifest_paths = cfg.test_ds[filepath_key]

    if type(manifest_paths) in (list, tuple, ListConfig):

        for filepath_val in manifest_paths:
            cfg.test_ds[filepath_key] = filepath_val
            model.setup_test_data(cfg.test_ds)
            dataloaders.append(model._test_dl)

        model._test_dl = dataloaders
        model._test_filenames = [parse_filepath_as_name(fp) for fp in manifest_paths]

        # In fast-dev-run, only one data loader is used
        if model._trainer.fast_dev_run:
            model._test_dl = model._test_dl[:1]
            model._test_filenames = model._test_filenames[:1]

    else:
        model.setup_test_data(cfg.test_ds)
        model._test_filenames = [parse_filepath_as_name(manifest_paths)]
