# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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
import json

import pytest
import torch
from lhotse import CutSet, MonoCut
from lhotse.testing.dummies import DummyManifest
from omegaconf import DictConfig

from nemo.collections.asr.data.audio_to_text_lhotse_target_speaker import LhotseSpeechToTextTgtSpkBpeDataset
from nemo.collections.common.data.lhotse.nemo_adapters import LazyNeMoIterator
from nemo.collections.asr.models.hybrid_rnnt_ctc_bpe_tgt_spk_models import EncDecHybridRNNTCTCTgtSpkBPEModel
from nemo.collections.asr.parts.submodules import rnnt_beam_decoding as beam_decode
from nemo.collections.asr.parts.submodules import rnnt_greedy_decoding as greedy_decode
from nemo.collections.asr.parts.submodules.ctc_decoding import CTCBPEDecoding, CTCBPEDecodingConfig
from nemo.collections.asr.parts.utils.rnnt_utils import Hypothesis
from nemo.collections.common import tokenizers
from nemo.core.utils import numba_utils
from nemo.core.utils.numba_utils import __NUMBA_MINIMUM_VERSION__

NUMBA_RNNT_LOSS_AVAILABLE = numba_utils.numba_cpu_is_supported(
    __NUMBA_MINIMUM_VERSION__
) or numba_utils.numba_cuda_is_supported(__NUMBA_MINIMUM_VERSION__)


@pytest.fixture()
def hybrid_asr_tgt_spk_model(test_data_dir):
    preprocessor = {'cls': 'nemo.collections.asr.modules.AudioToMelSpectrogramPreprocessor', 'params': dict({})}

    model_defaults = {'enc_hidden': 1024, 'pred_hidden': 64}

    encoder = {
        'cls': 'nemo.collections.asr.modules.ConvASREncoder',
        'params': {
            'feat_in': 64,
            'activation': 'relu',
            'conv_mask': True,
            'jasper': [
                {
                    'filters': model_defaults['enc_hidden'],
                    'repeat': 1,
                    'kernel': [1],
                    'stride': [1],
                    'dilation': [1],
                    'dropout': 0.0,
                    'residual': False,
                    'separable': True,
                    'se': True,
                    'se_context_size': -1,
                }
            ],
        },
    }

    decoder = {
        '_target_': 'nemo.collections.asr.modules.RNNTDecoder',
        'prednet': {
            'pred_hidden': model_defaults['pred_hidden'],
            'pred_rnn_layers': 1,
        },
    }

    joint = {
        '_target_': 'nemo.collections.asr.modules.RNNTJoint',
        'jointnet': {
            'joint_hidden': 32,
            'activation': 'relu',
        },
    }

    decoding = {'strategy': 'greedy_batch', 'greedy': {'max_symbols': 30}}

    tokenizer = {'dir': os.path.join(test_data_dir, "asr", "tokenizers", "an4_wpe_128"), 'type': 'wpe'}

    loss = {'loss_name': 'default', 'warprnnt_numba_kwargs': {'fastemit_lambda': 0.001}}

    aux_ctc = {
        'ctc_loss_weight': 0.3,
        'use_cer': False,
        'ctc_reduction': 'mean_batch',
        'decoder': {
            '_target_': 'nemo.collections.asr.modules.ConvASRDecoder',
            'feat_in': 1024,
            'num_classes': -2,
            'vocabulary': None,
        },
        'decoding': DictConfig(CTCBPEDecodingConfig),
    }

    modelConfig = DictConfig(
        {
            'preprocessor': DictConfig(preprocessor),
            'model_defaults': DictConfig(model_defaults),
            'encoder': DictConfig(encoder),
            'decoder': DictConfig(decoder),
            'joint': DictConfig(joint),
            'tokenizer': DictConfig(tokenizer),
            'decoding': DictConfig(decoding),
            'loss': DictConfig(loss),
            'aux_ctc': DictConfig(aux_ctc),
        }
    )

    model_instance = EncDecHybridRNNTCTCTgtSpkBPEModel(cfg=modelConfig)
    return model_instance


class TestEncDecHybridRNNTCTgtSpkBPEModel:
    @pytest.mark.skipif(
        not NUMBA_RNNT_LOSS_AVAILABLE,
        reason='RNNTLoss has not been compiled with appropriate numba version.',
    )
    @pytest.mark.with_downloads()
    @pytest.mark.unit
    def test_constructor(self, hybrid_asr_tgt_spk_model):
        hybrid_asr_tgt_spk_model.train()
        # TODO: make proper config and assert correct number of weights
        # Check to/from config_dict:
        confdict = hybrid_asr_tgt_spk_model.to_config_dict()
        instance2 = EncDecHybridRNNTCTCTgtSpkBPEModel.from_config_dict(confdict)
        assert isinstance(instance2, EncDecHybridRNNTCTCTgtSpkBPEModel)

    @pytest.mark.with_downloads()
    @pytest.mark.skipif(
        not NUMBA_RNNT_LOSS_AVAILABLE,
        reason='RNNTLoss has not been compiled with appropriate numba version.',
    )
    @pytest.mark.unit
    def test_forward(self, hybrid_asr_tgt_spk_model):
        hybrid_asr_tgt_spk_model = hybrid_asr_tgt_spk_model.eval()

        hybrid_asr_tgt_spk_model.preprocessor.featurizer.dither = 0.0
        hybrid_asr_tgt_spk_model.preprocessor.featurizer.pad_to = 0

        hybrid_asr_tgt_spk_model.compute_eval_loss = False

        input_signal = torch.randn(size=(4, 512))
        length = torch.randint(low=161, high=500, size=[4])
        spk_targets = torch.zeros(size=[4])

        with torch.no_grad():
            # batch size 1
            logprobs_instance = []
            for i in range(input_signal.size(0)):
                logprobs_ins, _ = hybrid_asr_tgt_spk_model.forward(
                    input_signal=input_signal[i : i + 1], input_signal_length=length[i : i + 1], spk_targets = spk_targets[i : i + 1]
                )
                logprobs_instance.append(logprobs_ins)
            logits_instance = torch.cat(logprobs_instance, 0)

            # batch size 4
            logprobs_batch, _ = hybrid_asr_tgt_spk_model.forward(input_signal=input_signal, input_signal_length=length, spk_targets = spk_targets)

        assert logits_instance.shape == logprobs_batch.shape
        diff = torch.mean(torch.abs(logits_instance - logprobs_batch))
        assert diff <= 1e-6
        diff = torch.max(torch.abs(logits_instance - logprobs_batch))
        assert diff <= 1e-6

    @pytest.mark.unit
    def test_predict_step(self, hybrid_asr_tgt_spk_model):
        hybrid_asr_tgt_spk_model = hybrid_asr_tgt_spk_model.eval()

        cuts = DummyManifest(CutSet, begin_id=0, end_id=1, with_data=True)
        with tempfile.TemporaryDirectory() as tmp_dir:
            cuts[0].save_audio(os.path.join(tmp_dir, 'query.wav'))
            # add query related info
            cuts[0].speaker_id = 'spk0'
            cuts[0].query_audio_filepath = os.path.join(tmp_dir, 'query.wav')
            cuts[0].query_offset = 0
            cuts[0].query_duration = 1
            cuts[0].query_speaker_id = 'spk0'

            cfg = DictConfig({
                'sample_rate': 16000,
            })

            dataset = LhotseSpeechToTextTgtSpkBpeDataset(tokenizer=hybrid_asr_tgt_spk_model.tokenizer, cfg=cfg)
            batch = dataset[cuts]
            outputs = hybrid_asr_tgt_spk_model.predict_step(batch, 0)
        assert len(outputs) == 1
        assert len(outputs[0]) == 2
        assert isinstance(outputs[0][1], Hypothesis)

    @pytest.mark.with_downloads()
    @pytest.mark.skipif(
        not NUMBA_RNNT_LOSS_AVAILABLE,
        reason='RNNTLoss has not been compiled with appropriate numba version.',
    )
    @pytest.mark.unit
    def test_save_restore_artifact(self, hybrid_asr_tgt_spk_model):
        hybrid_asr_tgt_spk_model.train()

        with tempfile.TemporaryDirectory() as tmp_dir:
            path = os.path.join(tmp_dir, 'rnnt_bpe.nemo')
            hybrid_asr_tgt_spk_model.save_to(path)

            new_model = EncDecHybridRNNTCTCTgtSpkBPEModel.restore_from(path)
            assert isinstance(new_model, type(hybrid_asr_tgt_spk_model))
            assert new_model.vocab_path.endswith('_vocab.txt')

            assert len(new_model.tokenizer.tokenizer.get_vocab()) == 128

    @pytest.mark.with_downloads()
    @pytest.mark.skipif(
        not NUMBA_RNNT_LOSS_AVAILABLE,
        reason='RNNTLoss has not been compiled with appropriate numba version.',
    )
    @pytest.mark.unit
    def test_save_restore_artifact_spe(self, hybrid_asr_tgt_spk_model, test_data_dir):
        hybrid_asr_tgt_spk_model.train()

        with tempfile.TemporaryDirectory() as tmpdir:
            tokenizer_dir = os.path.join(test_data_dir, "asr", "tokenizers", "an4_spe_128")
            hybrid_asr_tgt_spk_model.change_vocabulary(new_tokenizer_dir=tokenizer_dir, new_tokenizer_type='bpe')

            save_path = os.path.join(tmpdir, 'ctc_bpe.nemo')
            hybrid_asr_tgt_spk_model.train()
            hybrid_asr_tgt_spk_model.save_to(save_path)

            new_model = EncDecHybridRNNTCTCTgtSpkBPEModel.restore_from(save_path)
            assert isinstance(new_model, type(hybrid_asr_tgt_spk_model))
            assert isinstance(new_model.tokenizer, tokenizers.SentencePieceTokenizer)
            assert new_model.model_path.endswith('_tokenizer.model')
            assert new_model.vocab_path.endswith('_vocab.txt')
            assert new_model.spe_vocab_path.endswith('_tokenizer.vocab')

    @pytest.mark.with_downloads()
    @pytest.mark.unit
    def test_save_restore_artifact_agg(self, hybrid_asr_tgt_spk_model, test_data_dir):
        tokenizer_dir = os.path.join(test_data_dir, "asr", "tokenizers", "an4_spe_128")
        tok_en = {"dir": tokenizer_dir, "type": "wpe"}
        # the below is really an english tokenizer but we pretend it is spanish
        tok_es = {"dir": tokenizer_dir, "type": "wpe"}
        tcfg = DictConfig({"type": "agg", "langs": {"en": tok_en, "es": tok_es}})
        with tempfile.TemporaryDirectory() as tmpdir:
            hybrid_asr_tgt_spk_model.change_vocabulary(new_tokenizer_dir=tcfg, new_tokenizer_type="agg")

            save_path = os.path.join(tmpdir, "ctc_agg.nemo")
            hybrid_asr_tgt_spk_model.train()
            hybrid_asr_tgt_spk_model.save_to(save_path)

            new_model = EncDecHybridRNNTCTCTgtSpkBPEModel.restore_from(save_path)
            assert isinstance(new_model, type(hybrid_asr_tgt_spk_model))
            assert isinstance(new_model.tokenizer, tokenizers.AggregateTokenizer)

            # should be double
            assert new_model.tokenizer.tokenizer.vocab_size == 254
            assert len(new_model.tokenizer.tokenizer.get_vocab()) == 254

