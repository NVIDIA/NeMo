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

"""LLM deployment package with tensorrt_llm."""

from mpi4py import MPI

# Pre load MPI libs to avoid tensorrt_llm importing failures.
print(f"Loaded mpi lib {MPI.__file__} successfully")

# Pre import tensorrt_llm
try:
    import tensorrt_llm
except Exception as e:
    print(
        "tensorrt_llm package is not installed. Please build or install tensorrt_llm package"
        " properly before calling the llm deployment API."
    )
    raise (e)

from .model_config_trt import *  # noqa
from .nemo_utils import *  # noqa
from .quantization_utils import *  # noqa
from .tensorrt_llm_run import *  # noqa
