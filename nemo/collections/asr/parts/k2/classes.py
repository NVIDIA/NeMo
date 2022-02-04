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

from dataclasses import dataclass
from typing import Any, Optional, Union


@dataclass
class GraphIntersectDenseConfig:
    """Graph dense intersection config.
    """

    search_beam: float = 20.0
    output_beam: float = 10.0
    min_active_states: int = 30
    max_active_states: int = 10000


@dataclass
class GraphModuleConfig:
    """Config for graph modules.
    Typically used with graph losses and decoders.
    """

    topo_type: str = "default"
    topo_with_selfloops: bool = True
    graph_type: str = "topo"
    loss_type: str = "mmi"
    token_lm: Optional[Union[Any, str]] = None
    intersect_pruned: bool = False
    intersect_conf: GraphIntersectDenseConfig = GraphIntersectDenseConfig()
    boost_coeff: float = 0.0
