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
import shutil
import tempfile

import pytest

from nemo.collections.nlp.models import MTEncDecModel
from nemo.collections.nlp.models.machine_translation.mt_enc_dec_config import AAYNBaseConfig


def export_test(model, suffix, try_script=False):
    with tempfile.TemporaryDirectory() as restore_folder:
        enc_filename = os.path.join(restore_folder, 'nmt_e' + suffix)
        dec_filename = os.path.join(restore_folder, 'nmt_d' + suffix)
        model.encoder.export(output=enc_filename, try_script=try_script)
        model.decoder.export(output=dec_filename, try_script=try_script)
        assert os.path.exists(enc_filename)
        assert os.path.exists(dec_filename)


def get_cfg():
    cfg = AAYNBaseConfig()
    cfg.encoder_tokenizer.tokenizer_name = 'yttm'
    cfg.encoder_tokenizer.tokenizer_model = 'tests/.data/yttm.4096.en-de.model'
    cfg.decoder_tokenizer.tokenizer_name = 'yttm'
    cfg.decoder_tokenizer.tokenizer_model = 'tests/.data/yttm.4096.en-de.model'
    cfg.train_ds = None
    cfg.validation_ds = None
    cfg.test_ds = None
    return cfg


class TestMTEncDecModel:
    @pytest.mark.unit
    def test_creation_saving_restoring(self):
        model = MTEncDecModel(cfg=get_cfg())
        assert isinstance(model, MTEncDecModel)
        # Create a new temporary directory
        with tempfile.TemporaryDirectory() as restore_folder:
            with tempfile.TemporaryDirectory() as save_folder:
                save_folder_path = save_folder
                # Where model will be saved
                model_save_path = os.path.join(save_folder, f"{model.__class__.__name__}.nemo")
                model.save_to(save_path=model_save_path)
                # Where model will be restored from
                model_restore_path = os.path.join(restore_folder, f"{model.__class__.__name__}.nemo")
                shutil.copy(model_save_path, model_restore_path)
            # at this point save_folder should not exist
            assert save_folder_path is not None and not os.path.exists(save_folder_path)
            assert not os.path.exists(model_save_path)
            assert os.path.exists(model_restore_path)
            # attempt to restore
            model_copy = model.__class__.restore_from(restore_path=model_restore_path)
            assert model.num_weights == model_copy.num_weights

    @pytest.mark.unit
    def test_cpu_export_onnx(self):
        model = MTEncDecModel(cfg=get_cfg())
        assert isinstance(model, MTEncDecModel)
        export_test(model, ".ts")

    @pytest.mark.unit
    def test_cpu_export_ts(self):
        model = MTEncDecModel(cfg=get_cfg())
        assert isinstance(model, MTEncDecModel)
        export_test(model, ".onnx")

    @pytest.mark.skipif(not os.path.exists('/home/TestData/nlp'), reason='Not a Jenkins machine')
    @pytest.mark.run_only_on('GPU')
    @pytest.mark.unit
    def test_gpu_export_ts(self):
        model = MTEncDecModel(cfg=get_cfg()).cuda()
        assert isinstance(model, MTEncDecModel)
        export_test(model, ".ts")

    @pytest.mark.run_only_on('GPU')
    @pytest.mark.unit
    def test_gpu_export_onnx(self):
        model = MTEncDecModel(cfg=get_cfg()).cuda()
        assert isinstance(model, MTEncDecModel)
        export_test(model, ".onnx")


if __name__ == "__main__":
    t = TestMTEncDecModel()
    t.test_gpu_export_ts()
