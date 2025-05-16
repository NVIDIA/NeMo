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

import lightning.pytorch as pl

from nemo.collections.vlm.qwen25vl.model import Qwen25VLConfig3B, Qwen25VLConfig7B, Qwen25VLConfig72B, Qwen25VLModel


def qwen25vl_3b() -> pl.LightningModule:
    # pylint: disable=C0115,C0116
    return Qwen25VLModel(Qwen25VLConfig3B())


def qwen25vl_7b() -> pl.LightningModule:
    # pylint: disable=C0115,C0116
    return Qwen25VLModel(Qwen25VLConfig7B())


def qwen25vl_72b() -> pl.LightningModule:
    # pylint: disable=C0115,C0116
    return Qwen25VLModel(Qwen25VLConfig72B())


__all__ = ["qwen25vl_3b", "qwen25vl_7b", "qwen25vl_72b"]
