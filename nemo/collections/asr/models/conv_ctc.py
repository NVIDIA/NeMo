# ! /usr/bin/python
# -*- coding: utf-8 -*-

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

__all__ = ['ConvCTCModel', 'CTCModel', 'JasperNet', 'QuartzNet']

from abc import ABC

from nemo.core.classes import NeMoModelPT


class CTCModel(NeMoModelPT, ABC):
    """Abstract class which is a base for all CTC-models"""

    pass


class ConvCTCModel(CTCModel):
    """Implementation of convolution-based CTC model for ASR. Models like JasperNet and QuartzNet."""

    pass


class JasperNet(ConvCTCModel):
    pass


class QuartzNet(ConvCTCModel):
    pass
