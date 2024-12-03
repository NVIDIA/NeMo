# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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

import pytest

from nemo.utils.distributed_checkpointing import preprocess_common_state_dict_before_consistency_check)


def test_preprocess_common_state_dict_before_consistency_check(self):
    """Test processing common state dict before saving. """

    # Case 1: Callbacks/Timer included in state dict
    state_dict = {
        "callbacks": {
            "Timer": {"elapsed": 1.0}, 
            "other": {"entry": 42}
        }, 
        "bar": {"baz": "qux"}
    }
    expected = {"callbacks": {"other": {"entry": 42}}, "bar": {"baz": "qux"}}
    processed = preprocess_common_state_dict_before_consistency_check(state_dict)

    state_dict = {"foo": {"bar": "baz"}}
    expected = {"foo": {"bar": "baz"}}
    assert preprocess_common_state_dict_before_consistency_check(state_dict) == expected      

    # Test case 3: Callbacks dictionary exists but no Timer key
    state_dict = {"callbacks": {"Other": "value"}, "foo": {"bar": "baz"}}
    expected = {"callbacks": {"Other": "value"}, "foo": {"bar": "baz"}}
    assert preprocess_common_state_dict_before_consistency_check(state_dict) == expected

    # Test case 4: Empty dictionary
    state_dict = {}
    expected = {}
    assert preprocess_common_state_dict_before_consistency_check(state_dict) == expected