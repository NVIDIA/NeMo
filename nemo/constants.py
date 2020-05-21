# Copyright (C) NVIDIA CORPORATION. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.****

import numpy as np

# Supported Numpy DTypes: `np.sctypes`
ACCEPTED_INT_NUMBER_FORMATS = (
    int,
    np.uint8,
    np.uint16,
    np.uint32,
    np.uint64,
    np.int,
    np.int8,
    np.int16,
    np.int32,
    np.int64,
)

ACCEPTED_FLOAT_NUMBER_FORMATS = (
    float,
    np.float,
    np.float16,
    np.float32,
    np.float64,
    np.float128,
)

ACCEPTED_STR_NUMBER_FORMATS = (
    str,
    np.str,
)

ACCEPTED_NUMBER_FORMATS = ACCEPTED_INT_NUMBER_FORMATS + ACCEPTED_FLOAT_NUMBER_FORMATS + ACCEPTED_STR_NUMBER_FORMATS

# NEMO_ENV_VARNAME_DEBUG_VERBOSITY = "NEMO_DEBUG_VERBOSITY"
NEMO_ENV_VARNAME_ENABLE_COLORING = "NEMO_ENABLE_COLORING"
NEMO_ENV_VARNAME_REDIRECT_LOGS_TO_STDERR = "NEMO_REDIRECT_LOGS_TO_STDERR"
NEMO_ENV_VARNAME_TESTING = "NEMO_TESTING"
# NEMO_ENV_VARNAME_SAVE_LOGS_TO_DIR        = "NEMO_SAVE_LOGS_TO_DIR"
