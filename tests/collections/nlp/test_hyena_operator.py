# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

import pytest
import torch.nn

from nemo.collections.nlp.modules.common.hyena.hyena import HyenaOperator, SingleHeadHyenaConv, MultiHeadHyenaConv
from nemo.collections.nlp.modules.common.hyena.hyena_spec import get_hyena_layer_with_transformer_engine_spec
from megatron.core.transformer.transformer_config import TransformerConfig
from nemo.collections.nlp.parts.utils_funcs import torch_dtype_from_precision

try:
    import fftconv
    HAVE_FFTCONV = True
except ImportError:
    HAVE_FFTCONV = False

try:
    import flashfftconv
    HAVE_FLASHFFTCONV = True
except ImportError:
    HAVE_FLASHFFTCONV = False

try:
    import causal_conv1d
    HAVE_CAUSAL_CONV1D = True
except ImportError:
    HAVE_CAUSAL_CONV1D = False


@pytest.fixture()
def transformer_config():
    cfg = TransformerConfig(
        num_layers=2, hidden_size=864, num_attention_heads=1
    )
    return cfg


@pytest.fixture()
def hyena_config():
    cfg = {
        # HyenaOperator parameters
        'l_max': 1024,
        'order': 2,
        'filter_order': 64,
        'num_heads': 1,
        'dropout': 0.0,
        'short_filter_order': 3,
        'activation': "identity",

        # HyenaFilter parameters
        'precision': 'bf16',
        'fftconv_type': None,
        'emb_dim': 33,
        'learn_pos_emb_z': True,
        'w': 1,
        'bias': True,
        'normalized': False,
        'num_inner_mlps': 2,

        # ExponentialModulation parameters
        'fast_decay_pct': 0.3,
        'slow_decay_pct': 1.5,
        'target': 1e-2,
        'learn_modulation': False,
        'modulate': True,
        'shift': 0.0
    }
    return cfg


@pytest.fixture()
def submodules(hyena_config):
    return get_hyena_layer_with_transformer_engine_spec(hyena_config).submodules


@pytest.mark.run_only_on('GPU')
@pytest.mark.skipif(not HAVE_CAUSAL_CONV1D, reason='causal-conv-1d not installed')
class TestHyenaOperator:
    @pytest.mark.skipif(not HAVE_FFTCONV, reason='Safari fftconv not installed')
    @pytest.mark.parametrize("optionals_enabled, num_heads, expected_num_weights",
                             [(False, 1, 3068256), (True, 1, 3102912), (True, 8, 3053016)])
    def test_parameters(self, optionals_enabled, num_heads, expected_num_weights,
                        transformer_config, hyena_config, submodules):
        # Expected num weights calculation:
        #
        # Denote: inner_width = d_model * (order + 1)
        #         head_dim = d_model / num_heads
        #
        # in_proj (layer_norm) --> d_model * 2
        # in_proj (linear) --> d_model * inner_width + inner_width
        # out_proj (linear) --> d_model * d_model + d_model
        # short_filter (depthwise-separable 1d conv) --> inner_width * short_filter_order + inner_width
        # long_conv bias --> head_dim
        # filter:
        #   pos_emb.z --> l_max * emb_dim
        #   sin activation freqs --> filter_order
        #   mlp:
        #     input layer -->  emb_dim * filter_order + filter_order
        #     inner layers --> num_inner_mlps * (filter_order ^ 2 + filter_order)
        #     output_layer (no bias) --> filter_order * head_dim
        #   modulation: head_dim

        hyena_config['fftconv_type'] = 'safari'

        hyena_config['learn_pos_emb_z'] = optionals_enabled
        hyena_config['learn_modulation'] = optionals_enabled
        hyena_config['num_heads'] = num_heads
        hyena_module = HyenaOperator(transformer_config, submodules=submodules, **hyena_config)

        assert hyena_module.d_model == transformer_config.hidden_size
        assert isinstance(hyena_module.long_conv.filter.pos_emb.z, torch.nn.Parameter) == optionals_enabled
        assert isinstance(hyena_module.long_conv.filter.modulation.deltas, torch.nn.Parameter) == optionals_enabled

        num_weights = sum([p.numel() for p in hyena_module.parameters()])
        assert num_weights == expected_num_weights

    @staticmethod
    def check_gpu_forward(hyena_module, transformer_config, hyena_config):
        dtype = torch_dtype_from_precision(hyena_config['precision'])
        hyena_module = hyena_module.to(device='cuda', dtype=dtype)

        bs = 4
        seq_len = hyena_config['l_max']
        d_model = transformer_config.hidden_size

        x = torch.randn(seq_len, bs, d_model)
        x = x.to(device='cuda', dtype=dtype)

        y, _ = hyena_module(x)
        assert y.shape[0] == seq_len
        assert y.shape[1] == bs
        assert y.shape[2] == d_model

    @pytest.mark.skipif(not HAVE_FFTCONV, reason='Safari fftconv not installed')
    def test_single_head_safari(self, transformer_config, hyena_config, submodules):
        hyena_config['fftconv_type'] = 'safari'
        hyena_config['num_heads'] = 1
        hyena_module = HyenaOperator(transformer_config, submodules=submodules, **hyena_config)

        assert isinstance(hyena_module.long_conv, SingleHeadHyenaConv)
        assert hyena_module.long_conv.fftconv_fn == hyena_module.long_conv._safari_fft

        self.check_gpu_forward(hyena_module, transformer_config, hyena_config)

    @pytest.mark.skipif(not HAVE_FLASHFFTCONV, reason='Safari fftconv not installed')
    def test_single_head_flash(self, transformer_config, hyena_config, submodules):
        hyena_config['fftconv_type'] = 'flash'
        hyena_config['num_heads'] = 1
        hyena_module = HyenaOperator(transformer_config, submodules=submodules, **hyena_config)

        assert isinstance(hyena_module.long_conv, SingleHeadHyenaConv)
        assert hyena_module.long_conv.fftconv_fn == hyena_module.long_conv._flash_fft

        self.check_gpu_forward(hyena_module, transformer_config, hyena_config)

    @pytest.mark.skipif(not HAVE_FFTCONV, reason='Safari fftconv not installed')
    def test_multi_head(self, transformer_config, hyena_config, submodules):
        hyena_config['fftconv_type'] = 'safari'
        hyena_config['num_heads'] = 8
        hyena_module = HyenaOperator(transformer_config, submodules=submodules, **hyena_config)

        assert isinstance(hyena_module.long_conv, MultiHeadHyenaConv)

        self.check_gpu_forward(hyena_module, transformer_config, hyena_config)
