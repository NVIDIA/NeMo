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

# Copyright 2018-2020 William Falcon
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

from collections import namedtuple

import torch

from .pl_utils import BATCH_SIZE, NUM_BATCHES, NUM_CLASSES

LossInput = namedtuple('Input', ["loss_sum_or_avg", "num_measurements"])


_no_zero_num_measurements = LossInput(
    loss_sum_or_avg=torch.rand(NUM_BATCHES) * 2.0 - 1.0, num_measurements=torch.randint(1, 100, (NUM_BATCHES,)),
)

_some_num_measurements_are_zero = LossInput(
    loss_sum_or_avg=torch.rand(NUM_BATCHES) * 2.0 - 1.0,
    num_measurements=torch.cat(
        (
            torch.randint(1, 100, (NUM_BATCHES // 2,), dtype=torch.int32),
            torch.zeros(NUM_BATCHES - NUM_BATCHES // 2, dtype=torch.int32),
        )
    ),
)

_all_num_measurements_are_zero = LossInput(
    loss_sum_or_avg=torch.rand(NUM_BATCHES) * 2.0 - 1.0, num_measurements=torch.zeros(NUM_BATCHES, dtype=torch.int32),
)
