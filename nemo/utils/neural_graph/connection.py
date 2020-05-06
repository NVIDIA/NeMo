# -*- coding: utf-8 -*-

# =============================================================================
# Copyright (c) 2020 NVIDIA. All Rights Reserved.
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
# =============================================================================


from collections import namedtuple

# Tuple used for storing "step number", "module name" and "port name".
# (used in NmTensor's producer/consumer, port binding etc.).
# Module name is redundant, as it can be recovered from the step number.
StepModulePort = namedtuple('StepModulePort', ["step_number", "module_name", "port_name"])


# Tuple used for connection between a single producer and a single consummer consumer.
# (used in NmTensor's producer/consumer, port binding etc.).
Connection = namedtuple('Connection', ["producer", "consumer", "ntype"])
