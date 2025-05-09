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
from contextlib import contextmanager
from typing import Any

import torch
from lightning.pytorch.plugins import HalfPrecision
from typing_extensions import override


class HalfPrecisionForAudio(HalfPrecision):
    """
    Adjusted Pytorch Lightning plugin for training with half precision.
    It avoids downcasting audio in bfloat16.
    """

    @override
    def convert_input(self, data: Any) -> Any:
        if not isinstance(data, dict):
            return super().convert_input(data)

        def _convert(v):
            if isinstance(v, dict):
                ans = {}
                for k, v in v.items():
                    if "audio" not in k or not torch.is_tensor(v):
                        v = _convert(v)
                    ans[k] = v
                return ans
            if isinstance(v, torch.Tensor) and torch.is_floating_point(v):
                return v.to(self._desired_input_dtype)
            return v  # any other type

        return _convert(data)


@contextmanager
def fp32_precision():
    """
    Workaround for precision related issues when training with bf16-true PyTorch Lightning precision setting.
    In bf16-true, PTL changes PyTorch's default dtype, which may break implicit assumptions for some models.
    This context manager restores default float32 precision and runs the computation in float32 autocast context.
    """
    default_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.float32)
    try:
        with torch.amp.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu", dtype=torch.float32):
            yield
    finally:
        torch.set_default_dtype(default_dtype)
