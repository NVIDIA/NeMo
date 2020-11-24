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
import os
import shutil
import tempfile
from typing import Set

import pytest
from omegaconf import DictConfig, OmegaConf

from nemo.collections.asr.models import EncDecCTCModel, EncDecCTCModelBPE
from nemo.core.classes import ModelPT


def getattr2(object, attr):
    if not '.' in attr:
        return getattr(object, attr)
    else:
        arr = attr.split('.')
        return getattr2(getattr(object, arr[0]), '.'.join(arr[1:]))


class TestSaveRestore:
    def __test_restore_elsewhere(self, model: ModelPT, attr_for_eq_check: Set[str] = None):
        # Create a new temporary directory
        with tempfile.TemporaryDirectory() as restore_folder:
            with tempfile.TemporaryDirectory() as save_folder:
                save_folder_path = save_folder
                # Where model will be saved
                model_save_path = os.path.join(save_folder, f"{model.__class__.__name__}.nemo")
                model.save_to(save_path=model_save_path)
                # Where model will be restored from
                model_restore_path = os.path.join(restore_folder, f"{model.__class__.__name__}.nemo")
                shutil.copy(model_save_path, model_restore_path)
            # at this point save_folder should not exist
            assert save_folder_path is not None and not os.path.exists(save_folder_path)
            assert not os.path.exists(model_save_path)
            assert os.path.exists(model_restore_path)
            # attempt to restore
            model_copy = model.__class__.restore_from(restore_path=model_restore_path)
            if attr_for_eq_check is not None and len(attr_for_eq_check) > 0:
                for attr in attr_for_eq_check:
                    assert getattr2(model, attr) == getattr2(model_copy, attr)

    @pytest.mark.unit
    def test_EncDecCTCModel(self):
        # TODO: Switch to using named configs because here we don't really care about weights
        qn = EncDecCTCModel.from_pretrained(model_name="QuartzNet15x5Base-En")
        self.__test_restore_elsewhere(model=qn, attr_for_eq_check=set(["decoder._feat_in", "decoder._num_classes"]))

    @pytest.mark.unit
    def test_EncDecCTCModelBPE(self):
        # TODO: Switch to using named configs because here we don't really care about weights
        cn = EncDecCTCModelBPE.from_pretrained(model_name="ContextNet-192-WPE-1024-8x-Stride")
        self.__test_restore_elsewhere(model=cn)
