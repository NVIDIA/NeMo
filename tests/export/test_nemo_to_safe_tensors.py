# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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


def test_nemo2_convert_to_safe_tensors():
    """
    Test safe tensor exporter. This tests the whole nemo export until engine building.
    """
    from pathlib import Path

    from nemo.export.utils import convert_to_safe_tensors

    convert_to_safe_tensors(
        nemo_checkpoint_path="/home/TestData/llm/models/llama32_1b_nemo2",
        model_dir="/tmp/safe_tensor_test/",
        model_type="llama",
        delete_existing_files=True,
        tensor_parallelism_size=2,
        pipeline_parallelism_size= 1,
        gpus_per_node= 2,
        use_parallel_embedding= False,
        use_embedding_sharing= False,
        dtype= "bfloat16",
    )

    assert Path("/tmp/safe_tensor_test/").exists(), "Safe tensors were not generated."
    assert Path("/tmp/safe_tensor_test/rank0.safetensors").exists(), "Safe tensors for rank0 were not generated."
    assert Path("/tmp/safe_tensor_test/rank1.safetensors").exists(), "Safe tensors for rank1 were not generated."
    assert Path("/tmp/safe_tensor_test/config.json").exists(), "config.yaml was not generated."