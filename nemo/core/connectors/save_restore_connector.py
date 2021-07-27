# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
# Copyright 2015 and onwards Google, Inc.
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

from nemo.utils.model_utils import import_class_by_path
import os
import tarfile
import tempfile
from os import path
from typing import Optional, Union

import torch
from omegaconf import DictConfig, OmegaConf

from nemo.utils import logging
from nemo.utils.app_state import AppState

# TODO: Rip as much save/restore from modelpt and put in the connector
# TODO: Prioritize adding artifacts
# TODO: Try to create connector methods for as much as possible, espcially anything I/O


class SaveRestoreConnector:
    def _default_save_to(self, model, save_path: str):
        """
			Saves model instance (weights and configuration) into .nemo file.
			You can use "restore_from" method to fully restore instance from .nemo file.

			.nemo file is an archive (tar.gz) with the following:
				model_config.yaml - model configuration in .yaml format. You can deserialize this into cfg argument for model's constructor
				model_wights.chpt - model checkpoint

			Args:
				save_path: Path to .nemo file where model instance should be saved

		"""
        app_state = AppState()

        with tempfile.TemporaryDirectory() as tmpdir:
            config_yaml = path.join(tmpdir, app_state.model_config_yaml)
            model_weights = path.join(tmpdir, app_state.model_weights_ckpt)
            model.to_config_file(path2yaml_file=config_yaml)
            if hasattr(model, 'artifacts') and model.artifacts is not None:
                model._handle_artifacts(nemo_file_folder=tmpdir)
                # We should not update self._cfg here - the model can still be in use
                model._update_artifact_paths(path2yaml_file=config_yaml)
            # TODO: add connector method for saving weights
            torch.save(model.state_dict(), model_weights)
            self._make_nemo_file_from_folder(filename=save_path, source_dir=tmpdir)

    def _default_restore_from(
        self,
        restore_path: str,
        override_config_path: Optional[Union[OmegaConf, str]] = None,
        map_location: Optional[torch.device] = None,
        strict: bool = True,
        return_config: bool = False,
    ):
        """
		Restores model instance (weights and configuration) into .nemo file
		Args:
		restore_path: path to .nemo file from which model should be instantiated
		override_config_path: path to a yaml config that will override the internal
			config file or an OmegaConf / DictConfig object representing the model config.
		map_location: Optional torch.device() to map the instantiated model to a device.
			By default (None), it will select a GPU if available, falling back to CPU otherwise.
		strict: Passed to load_state_dict. By default True
		return_config: If set to true, will return just the underlying config of the restored
			model as an OmegaConf DictConfig object without instantiating the model.

		Example:
			```
			model = nemo.collections.asr.models.EncDecCTCModel.restore_from('asr.nemo')
			assert isinstance(model, nemo.collections.asr.models.EncDecCTCModel)
			```

		Returns:
		An instance of type cls or its underlying config (if return_config is set).
		"""
        app_state = AppState()

        # Get path where the command is executed - the artifacts will be "retrieved" there
        # (original .nemo behavior)
        cwd = os.getcwd()

        if map_location is None:
            if torch.cuda.is_available():
                map_location = torch.device('cuda')
            else:
                map_location = torch.device('cpu')

        with tempfile.TemporaryDirectory() as tmpdir:
            try:
                self._unpack_nemo_file(path2file=restore_path, out_folder=tmpdir)
                os.chdir(tmpdir)
                if override_config_path is None:
                    config_yaml = path.join(tmpdir, app_state.model_config_yaml)
                else:
                    # can be str path or OmegaConf / DictConfig object
                    config_yaml = override_config_path
                if not isinstance(config_yaml, (OmegaConf, DictConfig)):
                    conf = OmegaConf.load(config_yaml)
                else:
                    conf = config_yaml
                    if override_config_path is not None:
                        # Resolve the override config
                        conf = OmegaConf.to_container(conf, resolve=True)
                        conf = OmegaConf.create(conf)
                # If override is top level config, extract just `model` from it
                if 'model' in conf:
                    conf = conf.model

                if return_config:
                    instance = conf
                else:
                    app_state = AppState()
                if app_state.model_parallel_rank is not None:
                    model_weights = path.join(
                        tmpdir, f'mp_rank_{app_state.model_parallel_rank:02}', app_state.model_weights_ckpt
                    )
                else:
                    model_weights = path.join(tmpdir, app_state.model_weights_ckpt)
                OmegaConf.set_struct(conf, True)
                os.chdir(cwd)
                # get the class
                class_ = import_class_by_path(conf.target)
                class_._set_model_restore_state(is_being_restored=True, folder=tmpdir)
                instance = class_.from_config_dict(config=conf)
                instance = instance.to(map_location)
                instance.load_state_dict(torch.load(model_weights, map_location=map_location), strict=strict)

                logging.info(f'Model {instance.__class__.__name__} was successfully restored from {restore_path}.')
            finally:
                class_._set_model_restore_state(is_being_restored=False)
                os.chdir(cwd)

        return instance

    @staticmethod
    def _make_nemo_file_from_folder(filename, source_dir):
        with tarfile.open(filename, "w:gz") as tar:
            tar.add(source_dir, arcname=".")

    @staticmethod
    def _unpack_nemo_file(path2file: str, out_folder: str) -> str:
        if not path.exists(path2file):
            raise FileNotFoundError(f"{path2file} does not exist")
        tar = tarfile.open(path2file, "r:gz")
        tar.extractall(path=out_folder)
        tar.close()
        return out_folder
