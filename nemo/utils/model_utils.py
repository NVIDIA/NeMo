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
from pathlib import Path

from omegaconf import ListConfig
from nemo import logging


def parse_filepath_as_name(filepath: str) -> str:
    filename = Path(filepath).stem

    # cleanup name
    filename = filename.replace('-', '_')

    if 'manifest' in filename:
        filename = filename.replace('manifest', '')

    if '_' != filename[-1]:
        filename = filename + '_'

    return filename


def resolve_validation_dataloaders(model: 'ModelPT'):
    cfg = copy.deepcopy(model._cfg)
    dataloaders = []

    if hasattr(cfg.validation_ds, 'manifest_filepath'):
        manifest_paths = cfg.validation_ds.manifest_filepath

        if type(manifest_paths) in (list, tuple, ListConfig):

            for filepath in manifest_paths:
                cfg.validation_ds['manifest_filepath'] = filepath
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
            model._validation_filenames = [parse_filepath_as_name(model._cfg.validation_ds.manifest_filepath)]

    else:
        # no manifest filepath, cannot resolve multi paths
        logging.warning(
            'Cannot resolve multi-dataloader path since `validation_ds` does not contain `manifest_filepath`'
        )

        model.setup_validation_data(cfg.validation_ds)
        model._validation_filenames = ['validation_']


def resolve_test_dataloaders(model: 'ModelPT'):
    cfg = copy.deepcopy(model._cfg)
    dataloaders = []

    if hasattr(cfg.test_ds, 'manifest_filepath'):
        manifest_paths = cfg.test_ds.manifest_filepath

        if type(manifest_paths) in (list, tuple, ListConfig):

            for filepath in manifest_paths:
                cfg.test_ds['manifest_filepath'] = filepath
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
            model._test_filenames = [parse_filepath_as_name(model._cfg.test_ds.manifest_filepath)]

    else:
        # no manifest filepath, cannot resolve multi paths
        logging.warning('Cannot resolve multi-dataloader path since `test_ds` does not contain `manifest_filepath`')

        model.setup_test_data(cfg.test_ds)
        model._test_filenames = ['test_']
