# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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

from nemo.core.classes.module import NeuralModule
from nemo.collections.asr.parts.utils.activations import Swish
from torch import nn
import torch

class FusionModule(NeuralModule):

    """
    Audio-Visual Fusion module
    Concatenate audio and visual signals along the feature dimension and apply joint feed-forward network.
    Requires audio and visual signals to have the same length.

    args:
        a_dim_model: audio signal input dim
        v_dim_model: visual signal input dim
        f_dim_model: fused signal dim
        ff_ratio: feed forward ratio 
        fusion_type: concat or sum features before feed forward network, default to concat
        a_dropout: %chance of dropout audio signal during training
        v_dropout: %chance of dropout visual signal during training
        eval_a_dropout: %chance of dropout audio signal during evaluation
        eval_v_dropout: %chance of dropout visual signal during evaluation
        apply_ffn: whether to apply joint feed forward network after concat/sum fusion of audio-visual features, default to True
    
    """

    def __init__(self, a_dim_model, v_dim_model, f_dim_model, ff_ratio, fusion_type="concat", a_dropout=0.0, v_dropout=0.0, eval_a_dropout=0.0, eval_v_dropout=0.0, apply_ffn=True):
        super(FusionModule, self).__init__()

        assert fusion_type in ["concat", "sum"]
        self.fusion_type = fusion_type
        self.apply_ffn = apply_ffn

        if self.fusion_type == "concat":
            dim_in = a_dim_model + v_dim_model
        else:
            assert a_dim_model == v_dim_model
            dim_in = a_dim_model

        dim_ffn = ff_ratio * f_dim_model
        dim_out = f_dim_model

        # Layers
        if self.apply_ffn:
            self.layers = nn.Sequential(
                nn.Linear(dim_in, dim_ffn),
                Swish(),
                nn.Linear(dim_ffn, dim_out),
            )

        # Modality Dropout Train
        assert 0 <= a_dropout <= 1
        assert 0 <= v_dropout <= 1
        assert a_dropout + v_dropout <= 1
        self.a_dropout = a_dropout 
        self.v_dropout = v_dropout

        # Modality Dropout Eval
        assert 0 <= eval_a_dropout <= 1
        assert 0 <= eval_v_dropout <= 1
        assert eval_a_dropout + eval_v_dropout <= 1
        self.eval_a_dropout = eval_a_dropout 
        self.eval_v_dropout = eval_v_dropout

    def forward(self, audio_signal, video_signal):

        # Modality Dropout
        if self.training and (self.a_dropout > 0 or self.v_dropout > 0):

            # Sample for each batch element
            samples = torch.rand(audio_signal.shape[0])

            # Modalities dropped
            audio_dropped = samples < self.a_dropout
            video_dropped = samples > 1 - self.v_dropout

            # Mask Signals
            audio_signal[audio_dropped] = 0
            video_signal[video_dropped] = 0

        elif (not self.training) and (self.eval_a_dropout > 0 or self.eval_v_dropout > 0):

            # Sample for each batch element
            samples = torch.rand(audio_signal.shape[0])

            # Modalities dropped
            audio_dropped = samples < self.eval_a_dropout
            video_dropped = samples > 1 - self.eval_v_dropout

            # Mask Signals
            audio_signal[audio_dropped] = 0
            video_signal[video_dropped] = 0

        else:
            audio_dropped = None
            video_dropped = None
            
        # Fusion
        if self.fusion_type == "concat":
            x = torch.cat([audio_signal, video_signal], dim=1)
        elif self.fusion_type == "sum":
            x = audio_signal + video_signal

        # FFN
        if self.apply_ffn:
            x = self.layers(x.transpose(1, 2)).transpose(1, 2)

        return x, audio_dropped, video_dropped