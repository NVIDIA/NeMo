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

from nemo.collections.llm.modelopt.quantization.quantizer import QuantizationConfig


class TestQuantizationConfig:
    def test_is_weight_only_fp8(self):
        quant_cfg = QuantizationConfig(algorithm="fp8")
        assert not quant_cfg.is_weight_only()

    def test_is_weight_only_int8(self):
        quant_cfg = QuantizationConfig(algorithm="int8")
        assert not quant_cfg.is_weight_only()

    def test_is_weight_only_int8_sq(self):
        quant_cfg = QuantizationConfig(algorithm="int8_sq")
        assert not quant_cfg.is_weight_only()

    def test_is_weight_only_block_fp8(self):
        quant_cfg = QuantizationConfig(algorithm="block_fp8")
        assert quant_cfg.is_weight_only()

    def test_is_weight_only_int4_awq(self):
        quant_cfg = QuantizationConfig(algorithm="int4_awq")
        assert not quant_cfg.is_weight_only()

    def test_is_weight_only_w4a8_awq(self):
        quant_cfg = QuantizationConfig(algorithm="w4a8_awq")
        assert not quant_cfg.is_weight_only()

    def test_is_weight_only_int4(self):
        quant_cfg = QuantizationConfig(algorithm="int4")
        assert quant_cfg.is_weight_only()

    def test_is_weight_only_nvfp4(self):
        quant_cfg = QuantizationConfig(algorithm="nvfp4")
        assert not quant_cfg.is_weight_only()
