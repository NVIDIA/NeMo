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


class TestMTEncDecModel:
    @pytest.mark.unit
    def test_creation_saving_restoring(self):
        cfg = AAYNBaseConfig()
        cfg.encoder_tokenizer.tokenizer_name = 'yttm'
        cfg.encoder_tokenizer.tokenizer_model = 'tests/.data/yttm.4096.en-de.model'
        cfg.decoder_tokenizer.tokenizer_name = 'yttm'
        cfg.decoder_tokenizer.tokenizer_model = 'tests/.data/yttm.4096.en-de.model'
        cfg.train_ds = None
        cfg.validation_ds = None
        cfg.test_ds = None
        model = MTEncDecModel(cfg=cfg)
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
