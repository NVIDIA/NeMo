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
from typing import List


def on_iteration_start(callbacks):
    if callbacks is not None and isinstance(callbacks, List) and len(callbacks) > 0:
        for callback in callbacks:
            callback.on_iteration_start()


def on_iteration_end(callbacks):
    if callbacks is not None and isinstance(callbacks, List) and len(callbacks) > 0:
        for callback in callbacks:
            callback.on_iteration_end()


def on_action_start(callbacks):
    if callbacks is not None and isinstance(callbacks, List) and len(callbacks) > 0:
        for callback in callbacks:
            callback.on_action_start()


def on_action_end(callbacks):
    if callbacks is not None and isinstance(callbacks, List) and len(callbacks) > 0:
        for callback in callbacks:
            callback.on_action_end()


def on_epoch_start(callbacks):
    if callbacks is not None and isinstance(callbacks, List) and len(callbacks) > 0:
        for callback in callbacks:
            callback.on_epoch_start()


def on_epoch_end(callbacks):
    if callbacks is not None and isinstance(callbacks, List) and len(callbacks) > 0:
        for callback in callbacks:
            callback.on_epoch_end()


def update_callbacks(
    callbacks=None, registered_tensors=None,
):
    if callbacks is not None and isinstance(callbacks, List) and len(callbacks) > 0:
        for callback in callbacks:
            callback._registered_tensors = registered_tensors


def init_callbacks(appstate, callbacks):
    if callbacks is not None and isinstance(callbacks, List) and len(callbacks) > 0:
        for callback in callbacks:
            # TODO: This is a hack to make current callbacks work. REDO
            callback.action = appstate
