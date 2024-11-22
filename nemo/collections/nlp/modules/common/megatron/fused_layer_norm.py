# coding=utf-8
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

import warnings
try:
    import transformer_engine as te

    HAVE_TE = True
except (ImportError, ModuleNotFoundError):
    HAVE_TE = False


def get_layer_norm(hidden_size, eps=1e-5, persist_layer_norm=False, sequence_parallel=False):

    if persist_layer_norm:
        warnings.warn("persist_layer_norm is deprecated. Disabling.")

    ## TODO: persist_layer_norm is deprecated
    return te.pytorch.LayerNorm(hidden_size, eps, sequence_parallel=sequence_parallel)
