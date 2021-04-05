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

import pytest
import torch
from omegaconf import DictConfig, ListConfig

from nemo.collections.asr.metrics import rnnt_wer
from nemo.collections.asr.models import EncDecRNNTModel
from nemo.collections.asr.parts import rnnt_beam_decoding as beam_decode
from nemo.collections.asr.parts import rnnt_greedy_decoding as greedy_decode
from nemo.collections.asr.parts.numba import __NUMBA_MINIMUM_VERSION__, numba_utils
from nemo.utils.config_utils import assert_dataclass_signature_match

NUMBA_RNNT_LOSS_AVAILABLE = numba_utils.numba_cuda_is_supported(__NUMBA_MINIMUM_VERSION__)


@pytest.fixture()
def asr_model():
    preprocessor = {'cls': 'nemo.collections.asr.modules.AudioToMelSpectrogramPreprocessor', 'params': dict({})}

    # fmt: off
    labels = [' ', 'a', 'b', 'c', 'd', 'e', 'f',
              'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o',
              'p', 'q', 'r', 's', 't', 'u', 'v', 'w',
              'x', 'y', 'z', "'",
              ]
    # fmt: on

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
        'prednet': {'pred_hidden': model_defaults['pred_hidden'], 'pred_rnn_layers': 1},
    }

    joint = {
        '_target_': 'nemo.collections.asr.modules.RNNTJoint',
        'jointnet': {'joint_hidden': 32, 'activation': 'relu'},
    }

    decoding = {'strategy': 'greedy_batch', 'greedy': {'max_symbols': 30}}

    modelConfig = DictConfig(
        {
            'labels': ListConfig(labels),
            'preprocessor': DictConfig(preprocessor),
            'model_defaults': DictConfig(model_defaults),
            'encoder': DictConfig(encoder),
            'decoder': DictConfig(decoder),
            'joint': DictConfig(joint),
            'decoding': DictConfig(decoding),
        }
    )

    model_instance = EncDecRNNTModel(cfg=modelConfig)
    return model_instance


class TestEncDecRNNTModel:
    @pytest.mark.skipif(
        not NUMBA_RNNT_LOSS_AVAILABLE, reason='RNNTLoss has not been compiled with appropriate numba version.',
    )
    @pytest.mark.unit
    def test_constructor(self, asr_model):
        asr_model.train()
        # TODO: make proper config and assert correct number of weights
        # Check to/from config_dict:
        confdict = asr_model.to_config_dict()
        instance2 = EncDecRNNTModel.from_config_dict(confdict)
        assert isinstance(instance2, EncDecRNNTModel)

    @pytest.mark.skipif(
        not NUMBA_RNNT_LOSS_AVAILABLE, reason='RNNTLoss has not been compiled with appropriate numba version.',
    )
    @pytest.mark.unit
    def test_forward(self, asr_model):
        asr_model = asr_model.eval()

        asr_model.preprocessor.featurizer.dither = 0.0
        asr_model.preprocessor.featurizer.pad_to = 0

        asr_model.compute_eval_loss = False

        input_signal = torch.randn(size=(4, 512))
        length = torch.randint(low=161, high=500, size=[4])

        with torch.no_grad():
            # batch size 1
            logprobs_instance = []
            for i in range(input_signal.size(0)):
                logprobs_ins, _ = asr_model.forward(
                    input_signal=input_signal[i : i + 1], input_signal_length=length[i : i + 1]
                )
                logprobs_instance.append(logprobs_ins)
            logprobs_instance = torch.cat(logprobs_instance, 0)

            # batch size 4
            logprobs_batch, _ = asr_model.forward(input_signal=input_signal, input_signal_length=length)

        assert logprobs_instance.shape == logprobs_batch.shape
        diff = torch.mean(torch.abs(logprobs_instance - logprobs_batch))
        assert diff <= 1e-6
        diff = torch.max(torch.abs(logprobs_instance - logprobs_batch))
        assert diff <= 1e-6

    @pytest.mark.skipif(
        not NUMBA_RNNT_LOSS_AVAILABLE, reason='RNNTLoss has not been compiled with appropriate numba version.',
    )
    @pytest.mark.unit
    def test_vocab_change(self, asr_model):
        old_vocab = copy.deepcopy(asr_model.joint.vocabulary)
        nw1 = asr_model.num_weights
        asr_model.change_vocabulary(new_vocabulary=old_vocab)
        # No change
        assert nw1 == asr_model.num_weights
        new_vocab = copy.deepcopy(old_vocab)
        new_vocab.append('!')
        new_vocab.append('$')
        new_vocab.append('@')
        asr_model.change_vocabulary(new_vocabulary=new_vocab)
        # fully connected + bias
        # rnn embedding + joint + bias
        pred_embedding = 3 * (asr_model.decoder.pred_hidden)
        joint_joint = 3 * (asr_model.joint.joint_hidden + 1)
        assert asr_model.num_weights == (nw1 + (pred_embedding + joint_joint))

    @pytest.mark.skipif(
        not NUMBA_RNNT_LOSS_AVAILABLE, reason='RNNTLoss has not been compiled with appropriate numba version.',
    )
    @pytest.mark.unit
    def test_decoding_change(self, asr_model):
        assert isinstance(asr_model.decoding.decoding, greedy_decode.GreedyBatchedRNNTInfer)

        new_strategy = DictConfig({})
        new_strategy.strategy = 'greedy'
        new_strategy.greedy = DictConfig({'max_symbols': 10})
        asr_model.change_decoding_strategy(decoding_cfg=new_strategy)
        assert isinstance(asr_model.decoding.decoding, greedy_decode.GreedyRNNTInfer)

        new_strategy = DictConfig({})
        new_strategy.strategy = 'beam'
        new_strategy.beam = DictConfig({'beam_size': 1})
        asr_model.change_decoding_strategy(decoding_cfg=new_strategy)
        assert isinstance(asr_model.decoding.decoding, beam_decode.BeamRNNTInfer)
        assert asr_model.decoding.decoding.search_type == "default"

        new_strategy = DictConfig({})
        new_strategy.strategy = 'beam'
        new_strategy.beam = DictConfig({'beam_size': 2})
        asr_model.change_decoding_strategy(decoding_cfg=new_strategy)
        assert isinstance(asr_model.decoding.decoding, beam_decode.BeamRNNTInfer)
        assert asr_model.decoding.decoding.search_type == "default"

        new_strategy = DictConfig({})
        new_strategy.strategy = 'tsd'
        new_strategy.beam = DictConfig({'beam_size': 2})
        asr_model.change_decoding_strategy(decoding_cfg=new_strategy)
        assert isinstance(asr_model.decoding.decoding, beam_decode.BeamRNNTInfer)
        assert asr_model.decoding.decoding.search_type == "tsd"

        new_strategy = DictConfig({})
        new_strategy.strategy = 'alsd'
        new_strategy.beam = DictConfig({'beam_size': 2})
        asr_model.change_decoding_strategy(decoding_cfg=new_strategy)
        assert isinstance(asr_model.decoding.decoding, beam_decode.BeamRNNTInfer)
        assert asr_model.decoding.decoding.search_type == "alsd"

    @pytest.mark.unit
    def test_GreedyRNNTInferConfig(self):
        IGNORE_ARGS = ['decoder_model', 'joint_model', 'blank_index']

        result = assert_dataclass_signature_match(
            greedy_decode.GreedyRNNTInfer, greedy_decode.GreedyRNNTInferConfig, ignore_args=IGNORE_ARGS
        )

        signatures_match, cls_subset, dataclass_subset = result

        assert signatures_match
        assert cls_subset is None
        assert dataclass_subset is None

    @pytest.mark.unit
    def test_GreedyBatchedRNNTInferConfig(self):
        IGNORE_ARGS = ['decoder_model', 'joint_model', 'blank_index']

        result = assert_dataclass_signature_match(
            greedy_decode.GreedyBatchedRNNTInfer, greedy_decode.GreedyBatchedRNNTInferConfig, ignore_args=IGNORE_ARGS
        )

        signatures_match, cls_subset, dataclass_subset = result

        assert signatures_match
        assert cls_subset is None
        assert dataclass_subset is None

    @pytest.mark.unit
    def test_BeamRNNTInferConfig(self):
        IGNORE_ARGS = ['decoder_model', 'joint_model', 'blank_index']

        result = assert_dataclass_signature_match(
            beam_decode.BeamRNNTInfer, beam_decode.BeamRNNTInferConfig, ignore_args=IGNORE_ARGS
        )

        signatures_match, cls_subset, dataclass_subset = result

        assert signatures_match
        assert cls_subset is None
        assert dataclass_subset is None
