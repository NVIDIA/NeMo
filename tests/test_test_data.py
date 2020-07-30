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


from os.path import join, exists

import pytest

class TestData:
    @pytest.mark.unit
    def test_test_data_download(self, test_dir):
        """" Just a dummy tests showing how to use the test_dir fixture. """
        # test_dir contains the absolute path to nemo -> tests/.data
        assert exists(test_dir)
        assert exists(join(test_dir, "test_data.tar.gz"))
        
