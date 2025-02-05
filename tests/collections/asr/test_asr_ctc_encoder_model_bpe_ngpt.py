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

import copy
import os
import shutil
import tempfile

import pytest
import torch
from lhotse import CutSet, MonoCut
from lhotse.testing.dummies import DummyManifest
from omegaconf import DictConfig

from nemo.collections.asr.data import audio_to_text
from nemo.collections.asr.data.audio_to_text_lhotse import LhotseSpeechToTextBpeDataset
from nemo.collections.asr.models import configs
from nemo.collections.asr.models.ctc_bpe_models import EncDecCTCModelBPE
from nemo.collections.asr.parts.submodules import ctc_beam_decoding as beam_decode
from nemo.collections.asr.parts.submodules.ctc_decoding import CTCBPEDecoding, CTCBPEDecodingConfig
from nemo.collections.common import tokenizers
from nemo.utils.config_utils import assert_dataclass_signature_match


@pytest.fixture()
def asr_model(test_data_dir):
    preprocessor = {'_target_': 'nemo.collections.asr.modules.AudioToMelSpectrogramPreprocessor'}

    encoder = {
        'cls': 'nemo.collections.asr.modules.ngpt_encoder.NGPTEncoder',
        'params': {
            'feat_in': 64,
            'n_layers': 1,
            'd_model': 64,
            'subsampling': 'ngpt-frame-stack',
            'subsampling_factor': 2,
            'n_heads': 4,
            'use_nGPT': True,
        },
    }

    decoder = {
        '_target_': 'nemo.collections.asr.modules.ngpt_encoder.NGPTHead',
        'feat_in': 64,
        'num_classes': -1,
    }

    tokenizer = {'dir': os.path.join(test_data_dir, "asr", "tokenizers", "an4_wpe_128"), 'type': 'wpe'}

    modelConfig = DictConfig(
        {
            'preprocessor': DictConfig(preprocessor),
            'encoder': DictConfig(encoder),
            'decoder': DictConfig(decoder),
            'tokenizer': DictConfig(tokenizer),
        }
    )

    model_instance = EncDecCTCModelBPE(cfg=modelConfig)
    return model_instance


class TestEncDecCTCModel:
    @pytest.mark.unit
    def test_constructor(self, asr_model):
        asr_model.train()
        confdict = asr_model.to_config_dict()
        instance2 = EncDecCTCModelBPE.from_config_dict(confdict)
        assert isinstance(instance2, EncDecCTCModelBPE)

    @pytest.mark.unit
    def test_forward(self, asr_model):
        asr_model = asr_model.eval()

        asr_model.preprocessor.featurizer.dither = 0.0
        asr_model.preprocessor.featurizer.pad_to = 0

        input_signal = torch.randn(size=(4, 512))
        length = torch.randint(low=161, high=500, size=[4])

        with torch.no_grad():
            # batch size 1
            logprobs_instance = []
            for i in range(input_signal.size(0)):
                logprobs_ins, _, _ = asr_model.forward(
                    input_signal=input_signal[i : i + 1], input_signal_length=length[i : i + 1]
                )
                logprobs_instance.append(logprobs_ins)
                print(len(logprobs_ins))
            logprobs_instance = torch.cat(logprobs_instance, 0)

            # batch size 4
            logprobs_batch, _, _ = asr_model.forward(input_signal=input_signal, input_signal_length=length)

        assert logprobs_instance.shape == logprobs_batch.shape
        diff = torch.mean(torch.abs(logprobs_instance - logprobs_batch))
        assert diff <= 1e-6
        diff = torch.max(torch.abs(logprobs_instance - logprobs_batch))
        assert diff <= 1e-6

    @pytest.mark.unit
    def test_predict_step(self, asr_model):
        asr_model = asr_model.eval()
        cuts = DummyManifest(CutSet, begin_id=0, end_id=1, with_data=True)
        dataset = LhotseSpeechToTextBpeDataset(tokenizer=asr_model.tokenizer, return_cuts=True)
        batch = dataset[cuts]
        outputs = asr_model.predict_step(batch, 0)
        assert len(outputs) == 1
        assert len(outputs[0]) == 2
        assert isinstance(outputs[0][0], MonoCut)
        assert isinstance(outputs[0][1], str)

    @pytest.mark.unit
    def test_save_restore_artifact(self, asr_model):
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, 'ctc_bpe.nemo')
            asr_model.train()
            asr_model.save_to(save_path)

            new_model = EncDecCTCModelBPE.restore_from(save_path)
            assert isinstance(new_model, type(asr_model))
            assert new_model.vocab_path.endswith('_vocab.txt')

            assert len(new_model.tokenizer.tokenizer.get_vocab()) == 128
