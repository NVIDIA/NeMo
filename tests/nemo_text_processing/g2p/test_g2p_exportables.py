# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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
import tempfile

import onnx
import pytest
from nemo_text_processing.g2p.models.heteronym_classification import HeteronymClassificationModel


class TestExportableG2P:
    @pytest.mark.run_only_on('GPU')
    @pytest.mark.unit
    def test_HeteronymClassification_export_to_onnx(self):
        chpt = "/home/ebakhturina/NeMo/examples/text_processing/g2p/nemo_experiments/HeteronymClassification/2022-10-12_05-59-29/checkpoints/HeteronymClassification.nemo"
        nemo_model = HeteronymClassificationModel.restore_from(chpt)

        with tempfile.TemporaryDirectory() as tmpdir:
            filename = os.path.join(tmpdir, "heteronym_classification.onnx")
            nemo_model.export(output=filename, check_trace=True)
            onnx_model = onnx.load(filename)
            onnx.checker.check_model(onnx_model, full_check=True)  # throws when failed
            assert onnx_model.graph.input[0].name == "input_ids"
            assert onnx_model.graph.input[1].name == "attention_mask"
            assert onnx_model.graph.input[2].name == "token_type_ids"
            assert onnx_model.graph.output[0].name == "logits"
