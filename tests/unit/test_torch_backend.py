# ! /usr/bin/python
# -*- coding: utf-8 -*-

# =============================================================================
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
# =============================================================================

import pytest
from numpy import array_equal

from nemo.backends import get_state_dict, load, save, set_state_dict
from nemo.backends.pytorch.tutorials import TaylorNet


@pytest.mark.usefixtures("neural_factory")
class TestTorchBackend:
    @pytest.mark.unit
    def test_state_dict(self):
        """
            Tests whether the get/set_state_dict proxy functions work properly.
        """
        # Module.
        fx = TaylorNet(dim=4)

        # Get state dict.
        state_dict1 = get_state_dict(fx)

        # Set state dict.
        set_state_dict(fx, state_dict1)

        # Compare state dicts.
        state_dict2 = get_state_dict(fx)
        for key in state_dict1.keys():
            assert array_equal(state_dict1[key].cpu().numpy(), state_dict2[key].cpu().numpy())

    @pytest.mark.unit
    def test_save_load(self, tmpdir):
        """
            Tests whether the save and load proxy functions work properly.

            Args:
                tmpdir: Fixture which will provide a temporary directory.
        """
        # Module.
        fx = TaylorNet(dim=4)

        # Generate filename in the temporary directory.
        tmp_file_name = str(tmpdir.join("tsl_taylornet.chkpt"))

        # Save.
        weights = get_state_dict(fx)
        save(weights, tmp_file_name)

        # Load.
        loaded_weights = load(tmp_file_name)

        # Compare state dicts.
        for key in weights:
            assert array_equal(weights[key].cpu().numpy(), loaded_weights[key].cpu().numpy())
