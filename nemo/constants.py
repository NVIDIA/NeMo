#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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

ACCEPTED_NUMBER_FORMATS = \
    ACCEPTED_INT_NUMBER_FORMATS +  \
    ACCEPTED_FLOAT_NUMBER_FORMATS +  \
    ACCEPTED_STR_NUMBER_FORMATS

DLLOGGER_ENV_VARNAME_DEBUG_VERBOSITY         = "DLLOGGER_DEBUG_VERBOSITY"
DLLOGGER_ENV_VARNAME_ENABLE_GPU_WATCHER      = "DLLOGGER_ENABLE_GPU_WATCHER"
DLLOGGER_ENV_VARNAME_MULTIGPUS_INFERENCE     = "DLLOGGER_ENABLE_MULTIGPUS_INFERENCE"
DLLOGGER_ENV_VARNAME_ENABLE_COLORING         = "DLLOGGER_ENABLE_COLORING"
DLLOGGER_ENV_VARNAME_GPU_WATCH_FREQUENCY     = "DLLOGGER_GPU_WATCH_FREQUENCY"
DLLOGGER_ENV_VARNAME_MLPERF_COMPLIANT        = "DLLOGGER_MLPERF_COMPLIANT"
DLLOGGER_ENV_VARNAME_REDIRECT_LOGS_TO_STDERR = "DLLOGGER_REDIRECT_LOGS_TO_STDERR"
DLLOGGER_ENV_VARNAME_SAVE_LOGS_TO_DIR        = "DLLOGGER_SAVE_LOGS_TO_DIR"