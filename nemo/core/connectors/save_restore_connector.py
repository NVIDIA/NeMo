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


import tarfile
from nemo.core.classes.modelPT import ModelPT
from os import path
import tempfile

import torch
from nemo.utils import app_state

from nemo.utils.app_state import AppState


class SaveRestoreConnector:
    def __init__(self, model: ModelPT):
        self._model = model

    def _default_save_to(self, save_path: str):
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
            self._model.to_config_file(path2yaml_file=config_yaml)
            if hasattr(self._model, 'artifacts') and self._model.artifacts is not None:
                self._model._handle_artifacts(nemo_file_folder=tmpdir)
                # We should not update self._cfg here - the model can still be in use
                self._model._update_artifact_paths(path2yaml_file=config_yaml)
                torch.save(self._model.state_dict(), model_weights)
                self._make_nemo_file_from_folder(filename=save_path, source_dir=tmpdir)

    @staticmethod
    def _make_nemo_file_from_folder(filename, source_dir):
        with tarfile.open(filename, "w:gz") as tar:
            tar.add(source_dir, arcname=".")

