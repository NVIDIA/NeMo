# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

"""
This module provides a singleton instance of AsyncCallsQueue which manages
the async checkpoint save calls.
"""

import logging

from megatron.core.dist_checkpointing.strategies.async_utils import AsyncCallsQueue, AsyncRequest

from nemo.tron.config import CheckpointConfig
from nemo.tron.utils.common_utils import print_rank_0

logger = logging.getLogger(__name__)

# Singleton manager of async calls
# The default is `TemporalAsyncCaller`
_async_calls_queue = AsyncCallsQueue()


def init_persistent_async_worker():
    global _async_calls_queue
    # Recreate the async_calls_queue for persistent worker
    # This duplicate step is for backward compatiblity
    _async_calls_queue = AsyncCallsQueue(persistent=True)


def schedule_async_save(async_request: AsyncRequest):
    """Schedule the async save request.

    Args:
        async_request (AsyncRequest): the async save request.
    """
    _async_calls_queue.schedule_async_request(async_request)


def maybe_finalize_async_save(ckpt_cfg: CheckpointConfig, blocking: bool = False, terminate=False):
    """Finalizes active async save calls.

    Args:
        blocking (bool, optional): if True, will wait until all active requests
            are done. Otherwise, finalizes only the async request that already
            finished. Defaults to False.
        terminate (bool, optional): if True, the asynchronous queue will
                be closed as the last action of this function.
    """
    if not ckpt_cfg.async_save:
        return

    if blocking and not is_empty_async_queue():
        print_rank_0("Unfinalized async checkpoint saves. Finalizing them synchronously now.")

    _async_calls_queue.maybe_finalize_async_calls(blocking)

    if terminate:
        _async_calls_queue.close()


def is_empty_async_queue() -> bool:
    """Check if async calls queue is empty. This result is consistent across ranks.

    Returns:
        bool: True if there is any ongoing async call.
    """
    return _async_calls_queue.get_num_unfinalized_calls() == 0
