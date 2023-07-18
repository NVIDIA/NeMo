# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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
import urllib.request as req
from pathlib import Path
from nemo.deploy import NemoDeploy


class TestNemoDeployment:

    nemo_checkpoint_link = "https://huggingface.co/nvidia/GPT-2B-001/resolve/main/GPT-2B-001_bf16_tp1.nemo"
    nemo_checkpoint_path = "/opt/checkpoints/GPT-2B.nemo"
    temp_nemo_dir = "/opt/checkpoints/GPT-2B/"

    @pytest.mark.unit
    def test_to_triton_pytriton(self):
        """Here we test the slow inference deployment to triton"""

        if not Path(self.nemo_checkpoint_path):
            print("File will be downloaded...")
            req.urlretrieve(self.nemo_checkpoint_link, self.nemo_checkpoint_path)
            print("File download completed.")
        else:
            print("Checkpoint has already been downloaded.")

        nm = NemoDeploy(checkpoint_path=self.nemo_checkpoint_path,
                        triton_model_name="GPT_2B",
                        temp_nemo_dir=self.temp_nemo_dir)

        nm.deploy()
        nm.run()
        nm.stop()