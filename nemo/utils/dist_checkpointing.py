# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

from typing import Any, Dict


def preprocess_common_state_dict_before_consistency_check(state_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Filters common state dict entries before the consistency check is run.

    Distributed checkpointing saving automatically groups certain values into a common state dict,
    which is assumed to be equivalent across ranks.
    To ensure that the common state dict is actually equivalent across ranks, a consistency check is run.
    However, there are entries within NeMo that are known to be different across ranks, so here they are removed.
    """

    import copy

    # Deepcopy to ensure that all states in state dict are still saved
    state_dict_to_check = copy.deepcopy(state_dict)

    # Remove Timer callback states from consideration during consistency check
    state_dict_to_check.get("callbacks", {}).pop("Timer", None)

    return state_dict_to_check
