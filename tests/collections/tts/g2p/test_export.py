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
import os
import tempfile

import pytest
import torch
from omegaconf import OmegaConf

from nemo.collections.tts.g2p.models.t5 import T5G2PModel
from nemo.utils.app_state import AppState


@pytest.fixture()
def t5_model():
    this_test_dir = os.path.dirname(os.path.abspath(__file__))

    cfg = OmegaConf.load(os.path.join(this_test_dir, '../../../../examples/tts/g2p/conf/g2p_t5.yaml'))
    cfg.train_manifest = None
    cfg.validation_manifest = None
    app_state = AppState()
    app_state.is_model_being_restored = True
    model = T5G2PModel(cfg=cfg.model)
    app_state.is_model_being_restored = False
    model.eval()
    return model


def extra_cfg():
    cfg.model.init_from_ptl_ckpt = None
    cfg.model.train_ds.dataset.manifest_filepath = "dummy.json"
    cfg.model.train_ds.dataset.sup_data_path = "dummy.json"
    cfg.model.validation_ds.dataset.manifest_filepath = "dummy.json"
    cfg.model.validation_ds.dataset.sup_data_path = "dummy.json"
    cfg.pitch_mean = 212.35
    cfg.pitch_std = 68.52


class TestExportable:
    @pytest.mark.run_only_on('GPU')
    @pytest.mark.unit
    def test_T5Model_export_to_onnx(self, t5_model):
        model = t5_model.cuda()
        with tempfile.TemporaryDirectory() as tmpdir:
            filename = os.path.join(tmpdir, 'fp.onnx')
            model.export(output=filename, verbose=True, onnx_opset_version=18, check_trace=True)

    @pytest.mark.run_only_on('GPU')
    @pytest.mark.unit
    def test_T5Model_export_to_ts(self, t5_model):
        model = t5_model.cuda()
        with tempfile.TemporaryDirectory() as tmpdir:
            filename = os.path.join(tmpdir, 'fp.ts')
            model.export(output=filename, verbose=True, check_trace=True)
