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

import os

import torch

from nemo.utils import logging


def initialize_distributed(args, backend='nccl'):
    """Initialize torch.distributed."""
    # Get local rank in case it is provided.
    local_rank = args.local_rank

    # Get rank and world size.
    rank = int(os.getenv('RANK', '0'))
    world_size = int(os.getenv("WORLD_SIZE", '1'))

    logging.info(
        f'Initializing torch.distributed with local_rank: {local_rank}, rank: {rank}, world_size: {world_size}'
    )

    # Set the device id.
    device = rank % torch.cuda.device_count()
    if local_rank is not None:
        device = local_rank
    torch.cuda.set_device(device)

    # Call the init process.
    init_method = 'tcp://'
    master_ip = os.getenv('MASTER_ADDR', 'localhost')
    master_port = os.getenv('MASTER_PORT', '6000')
    init_method += master_ip + ':' + master_port
    torch.distributed.init_process_group(backend=backend, world_size=world_size, rank=rank, init_method=init_method)
    return local_rank, rank, world_size
