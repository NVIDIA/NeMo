# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

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

from .huggingface_utils import *  # noqa
from .model_config_trt import *  # noqa
from .model_config_utils import *  # noqa
from .nemo_utils import *  # noqa
from .quantization_utils import *  # noqa
from .tensorrt_llm_run import *  # noqa
