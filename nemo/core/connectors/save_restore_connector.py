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

import os
import pickle
import shutil
import tarfile
import tempfile
import uuid
from os import path
from typing import Optional, Union

import torch
from omegaconf import DictConfig, OmegaConf
from omegaconf.omegaconf import open_dict

from nemo.utils import app_state, logging, model_utils
from nemo.utils.app_state import AppState
from nemo.utils.model_utils import import_class_by_path


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
                self._handle_artifacts(model, nemo_file_folder=tmpdir)
                # We should not update self._cfg here - the model can still be in use
                self._update_artifact_paths(model, path2yaml_file=config_yaml)
            self._save_state_dict_to_disk(model.state_dict(), model_weights)
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
                # add load_state_dict override
                instance.load_state_dict(
                    self._load_state_dict_from_disk(model_weights, map_location=map_location), strict=strict
                )

                logging.info(f'Model {instance.__class__.__name__} was successfully restored from {restore_path}.')
                instance._set_model_restore_state(is_being_restored=False)
            finally:
                os.chdir(cwd)

        return instance

    def _extract_state_dict_from(self, restore_path: str, save_dir: str, split_by_module: bool = False):

        app_state = AppState()

        cwd = os.getcwd()

        save_dir = os.path.abspath(save_dir)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)

        with tempfile.TemporaryDirectory() as tmpdir:
            try:
                self._unpack_nemo_file(path2file=restore_path, out_folder=tmpdir)
                os.chdir(tmpdir)
                model_weights = path.join(tmpdir, app_state.model_weights_ckpt)
                state_dict = self._load_state_dict_from_disk(model_weights)

                if not split_by_module:
                    filepath = os.path.join(save_dir, app_state.model_weights_ckpt)
                    self._save_state_dict_to_disk(state_dict, filepath)

                else:
                    key_set = set([key.split(".")[0] for key in state_dict.keys()])
                    for primary_key in key_set:
                        inner_keys = [key for key in state_dict.keys() if key.split(".")[0] == primary_key]
                        state_dict_subset = {
                            ".".join(inner_key.split(".")[1:]): state_dict[inner_key] for inner_key in inner_keys
                        }
                        filepath = os.path.join(save_dir, f"{primary_key}.ckpt")
                        self._save_state_dict_to_disk(state_dict_subset, filepath)

                logging.info(f'Checkpoints from {restore_path} were successfully extracted into {save_dir}.')
            finally:
                os.chdir(cwd)

        return state_dict

    def _register_artifact(self, model, config_path: str, src: str, verify_src_exists: bool = True):

        app_state = AppState()

        artifact_item = model_utils.ArtifactItem()

        # This is for backward compatibility, if the src objects exists simply inside of the tarfile
        # without its key having been overriden, this pathway will be used.
        src_obj_name = os.path.basename(src)
        if app_state.nemo_file_folder is not None:
            src_obj_path = os.path.abspath(os.path.join(app_state.nemo_file_folder, src_obj_name))
        else:
            src_obj_path = src_obj_name

        # src is a local existing path - register artifact and return exact same path for usage by the model
        if os.path.exists(os.path.abspath(src)):
            return_path = os.path.abspath(src)
            artifact_item.path_type = model_utils.ArtifactPathType.LOCAL_PATH

        # this is the case when artifact must be retried from the nemo file
        # we are assuming that the location of the right nemo file is available from _MODEL_RESTORE_PATH
        elif src.startswith("nemo:"):
            return_path = os.path.abspath(os.path.join(app_state.nemo_file_folder, src[5:]))
            artifact_item.path_type = model_utils.ArtifactPathType.TAR_PATH

        # backward compatibility implementation
        elif os.path.exists(src_obj_path):
            return_path = src_obj_path
            artifact_item.path_type = model_utils.ArtifactPathType.TAR_PATH
        else:
            if verify_src_exists:
                raise FileNotFoundError(
                    f"src path does not exist or it is not a path in nemo file. src value I got was: {src}. Absolute: {os.path.abspath(src)}"
                )
            else:
                # artifact is optional and we simply return None
                return None

        assert os.path.exists(return_path)

        artifact_item.path = os.path.abspath(src)
        model.artifacts[config_path] = artifact_item
        # we were called by ModelPT
        if hasattr(model, "cfg"):
            with open_dict(model._cfg):
                OmegaConf.update(model.cfg, config_path, return_path)
        return return_path

    def _handle_artifacts(self, model, nemo_file_folder):
        tarfile_artifacts = []
        app_state = AppState()
        for conf_path, artiitem in model.artifacts.items():
            if artiitem.path_type == model_utils.ArtifactPathType.LOCAL_PATH:
                if not os.path.exists(artiitem.path):
                    raise FileNotFoundError(f"Artifact {conf_path} not found at location: {artiitem.path}")

                # Generate new uniq artifact name and copy it to nemo_file_folder
                # Note uuid.uuid4().hex is guaranteed to be 32 character long
                artifact_base_name = os.path.basename(artiitem.path)
                artifact_uniq_name = f"{uuid.uuid4().hex}_{artifact_base_name}"
                shutil.copy2(artiitem.path, os.path.join(nemo_file_folder, artifact_uniq_name))

                # Update artifacts registry
                artiitem.hashed_path = "nemo:" + artifact_uniq_name
                model.artifacts[conf_path] = artiitem

            elif artiitem.path_type == model_utils.ArtifactPathType.TAR_PATH:
                # process all tarfile artifacts in one go, so preserve key-value pair
                tarfile_artifacts.append((conf_path, artiitem))

            else:
                raise ValueError(f"Directly referencing artifacts from other nemo files isn't supported yet")

        # Process current tarfile artifacts by unpacking the previous tarfile and extract the artifacts
        # that are currently required.
        model_metadata = app_state.get_model_metadata_from_guid(model.model_guid)
        if len(tarfile_artifacts) > 0 and model_metadata.restoration_path is not None:
            # Need to step into nemo archive to extract file
            # Get path where the command is executed - the artifacts will be "retrieved" there
            # (original .nemo behavior)
            cwd = os.getcwd()
            try:
                # Step into the nemo archive to try and find the file
                with tempfile.TemporaryDirectory() as archive_dir:
                    self._unpack_nemo_file(path2file=model_metadata.restoration_path, out_folder=archive_dir)
                    os.chdir(archive_dir)
                    for conf_path, artiitem in tarfile_artifacts:
                        # Get basename and copy it to nemo_file_folder
                        if 'nemo:' in artiitem.path:
                            artifact_base_name = artiitem.path.split('nemo:')[1]
                        else:
                            artifact_base_name = os.path.basename(artiitem.path)
                        # no need to hash here as we are in tarfile_artifacts which are already hashed
                        artifact_uniq_name = artifact_base_name
                        shutil.copy2(artifact_base_name, os.path.join(nemo_file_folder, artifact_uniq_name))

                        # Update artifacts registry
                        new_artiitem = model_utils.ArtifactItem()
                        new_artiitem.path = "nemo:" + artifact_uniq_name
                        new_artiitem.path_type = model_utils.ArtifactPathType.TAR_PATH
                        model.artifacts[conf_path] = new_artiitem
            finally:
                # change back working directory
                os.chdir(cwd)

    def _update_artifact_paths(self, model, path2yaml_file):
        if model.artifacts is not None and len(model.artifacts) > 0:
            conf = OmegaConf.load(path2yaml_file)
            for conf_path, item in model.artifacts.items():
                if item.hashed_path is None:
                    OmegaConf.update(conf, conf_path, item.path)
                else:
                    OmegaConf.update(conf, conf_path, item.hashed_path)
            with open(path2yaml_file, 'w') as fout:
                OmegaConf.save(config=conf, f=fout, resolve=True)

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

    @staticmethod
    def _save_state_dict_to_disk(state_dict, filepath):
        torch.save(state_dict, filepath)

    @staticmethod
    def _load_state_dict_from_disk(model_weights, map_location=None):
        return torch.load(model_weights, map_location=map_location)
