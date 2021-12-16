# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
import random
from typing import List, Tuple
from nemo.core.neural_types.elements import AcousticEncodedRepresentation
from omegaconf.dictconfig import DictConfig

from nemo.collections.asr.parts.submodules.jasper import (
    init_weights, 
    jasper_activations,
)

import torch
from torch import nn
from torch.nn import functional as F

from nemo.core.neural_types import NeuralType, AudioSignal, LengthsType, SpectrogramType, LogprobsType
from nemo.core.classes.exportable import Exportable
from nemo.core.classes.module import NeuralModule
from nemo.core.classes.common import typecheck

from nemo.collections.common.parts import form_attention_mask, transformer_weights_init
from nemo.collections.nlp.modules.common.transformer import TransformerEncoder

from omegaconf import DictConfig
from collections import OrderedDict

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TransposeLast(torch.nn.Module):
    """
    Transposes last dimension. Useful for adding to a sequential block.
    """

    def forward(self, x):
        return x.transpose(-2, -1)


class SamePad(torch.nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.remove = kernel_size % 2 == 0

    def forward(self, x):
        if self.remove:
            x = x[:, :, :-1]
        return x


class ConvFeatureEncoder(NeuralModule):
    """
        Converts input raw audio into features for downstream transformer model.
        Uses 1D convolutional blocks with GeLU activation.
    """
    @property
    def input_types(self):
        """Returns definitions of module input ports.
        input_signal:
            0: AxisType(BatchTag)
            1: AxisType(TimeTag)
        input_signal_length:
            0: AxisType(BatchTag)
        Note: length is in number of samples, not seconds
        """
        return {
            "input_signal": NeuralType(('B', 'T'), AudioSignal(freq=self._sample_rate)),
            "length": NeuralType(
                tuple('B'), LengthsType()
            ),
        }

    @property
    def output_types(self):
        """Returns definitions of module output ports. 
        For compatibility, processed features are treated as Spectrogram types
        processed_signal:
            0: AxisType(BatchTag)
            1: AxisType(ChannelTag)
            2: AxisType(ProcessedTimeTag)
        processed_signal_length:
            0: AxisType(BatchTag)
        """
        return {
            "processed_signal": NeuralType(('B', 'C', 'T'), SpectrogramType()),
            "processed_signal_length": NeuralType(tuple('B'), LengthsType()),
        }

    def __init__(
        self,
        conv_layers: List[Tuple[int, int, int]],
        extractor_mode: str = "layer_norm",
        conv_bias: bool = False,
        feature_grad_mult = 1.0,
        normalize_audio = True,
        embedding_dim = 768
    ):
        super().__init__()

        self.grad_mult = feature_grad_mult
        self.normalize_input = normalize_audio

        def block(
            n_in, n_out, k, stride, is_layer_norm=False, is_group_norm=False, conv_bias=False,
        ):
            def make_conv():
                conv = nn.Conv1d(n_in, n_out, k, stride=stride, bias=conv_bias)
                nn.init.kaiming_normal_(conv.weight)
                return conv

            assert (is_layer_norm and is_group_norm) is False, "layer norm and group norm are exclusive"

            if is_layer_norm:
                return nn.Sequential(
                    make_conv(),
                    nn.Sequential(TransposeLast(), nn.LayerNorm(dim, elementwise_affine=True), TransposeLast()),
                    nn.GELU(),
                )
            elif is_group_norm:
                return nn.Sequential(make_conv(), nn.GroupNorm(dim, dim, affine=True), nn.GELU(),)
            else:
                return nn.Sequential(make_conv(), nn.GELU())

        in_d = 1
        self.layer_cfg = conv_layers
        self.conv_layers = nn.ModuleList()
        for i, cl in enumerate(conv_layers):
            assert len(cl) == 3, "invalid conv definition: " + str(cl)
            (dim, k, stride) = cl

            self.conv_layers.append(
                block(
                    in_d,
                    dim,
                    k,
                    stride,
                    is_layer_norm=extractor_mode == "layer_norm",
                    is_group_norm=extractor_mode == "group_norm" and i == 0, # Paper applies norm to first layer only
                    conv_bias=conv_bias,
                )
            )
            in_d = dim
    
        # Model Layers
        final_conv_dim = self.layer_cfg[-1][0]  # Select last conv output layer dimension
        self.post_extract_proj = ( # To project feature encodings to transformer
            nn.Linear(final_conv_dim, embedding_dim)
            if final_conv_dim != embedding_dim
            else None
        )
        self.layer_norm = nn.LayerNorm(embedding_dim)
    
    def apply_layers(self, x):
        for conv in self.conv_layers:
            x = conv(x)
        return x
    
    def normalize(self, source, lengths):
        with torch.no_grad(): # Normalizes audio source
            for i in range(lengths.size(0)):
                orig = source[i, :lengths[i]]
                norm = F.layer_norm(orig, orig.shape) # From FAIR
                source[i, :lengths[i]] = norm
        return source

    def forward(self, input_signal, length):
        if self.normalize_input:
            source = self.normalize(input_signal, length)

        # BxT -> BxCxT
        processed_signal = source.unsqueeze(1)

        # Applies grad mult scaling
        if self.grad_mult > 0:
            processed_signal = self.apply_layers(processed_signal)
            if self.grad_mult != 1.0: 
                processed_signal = GradMultiply.apply(processed_signal, self.grad_mult)
        else:
            with torch.no_grad(): # 0 indicates frozen feature encoder
                processed_signal = self.apply_layers(processed_signal)

        
        processed_signal = processed_signal.transpose(1, 2) # B,T,C
        # Project to embedding
        if self.post_extract_proj is not None:
            processed_signal = self.post_extract_proj(processed_signal)

        # Adding normalization for output
        processed_signal = self.layer_norm(processed_signal)
        processed_signal = processed_signal.transpose(1, 2) # B,C,T

        # Feature lengths will have been changed through convolutions
        processed_signal_length = self.get_lengths(audio_lengths=length)

        return processed_signal, processed_signal_length

    def get_lengths(self, audio_lengths): # from hugging face
        # converts audio lengths to timestep lengths
        for conv in self.layer_cfg:
            kernel = conv[1]
            stride = conv[2]
            audio_lengths = torch.div(audio_lengths - kernel, stride, rounding_mode='floor') + 1 # from pytorch doc
        return audio_lengths

class Wav2VecTransformerEncoder(TransformerEncoder):
    def __init__(self, pos_embed: DictConfig, transformer: DictConfig, layer_drop: float = 0.0):
        super().__init__(**transformer) # see nlp.collections

        # positional convolutional embeddings
        emb_dim = pos_embed.embedding_dim
        self.pos_conv = nn.Conv1d(
            emb_dim,
            emb_dim,
            kernel_size=pos_embed.conv_pos,
            padding=pos_embed.conv_pos // 2, # Padding size preserves time step length
            groups=pos_embed.conv_pos_groups,
        )

        self.layer_drop = layer_drop

        self.dropout = transformer.attn_layer_dropout # He initialization
        std = math.sqrt((4 * (1.0 - self.dropout)) / (pos_embed.conv_pos * pos_embed.embedding_dim))
        nn.init.normal_(self.pos_conv.weight, mean=0, std=std)
        nn.init.constant_(self.pos_conv.bias, 0)

        self.pos_conv = nn.utils.weight_norm(self.pos_conv, name="weight", dim=2)
        self.pos_conv = nn.Sequential(self.pos_conv, SamePad(pos_embed.conv_pos), nn.GELU())

        self.layer_norm = nn.LayerNorm(emb_dim)
        self.apply(lambda x: transformer_weights_init(x, xavier=False))

    @property
    def input_types(self):
        """Returns definitions of module output ports. 
        We treat features as SpectrogramType for Nemo compatibility
        audio_signal:
            0: AxisType(BatchTag)
            1: AxisType(ChannelTag)
            2: AxisType(ProcessedTimeTag)
        length:
            0: AxisType(BatchTag)
        """
        return {
            "audio_signal": NeuralType(('B', 'C', 'T'), SpectrogramType()),
            "length": NeuralType(tuple('B'), LengthsType()),
        }

    @property
    def output_types(self):
        """Returns definitions of module output ports. 
        We're using SpectrogramType for now to keep things Nemo safe
        processed_signal:
            0: AxisType(BatchTag)
            1: AxisType(ProcessedTimeTag)
            2: AxisType(ChannelTag)
        processed_length:
            0: AxisType(BatchTag)
        """
        return {
            "processed_signal": NeuralType(('B', 'T', 'C'), AcousticEncodedRepresentation()),
            "processed_length": NeuralType(tuple('B'), LengthsType()),
        }


    def forward(self, audio_signal, length):

        # Padding mask needed for transformer
        padding_mask = self.create_padding_mask(length)

        # Applying padding before convolution
        for idx, len in enumerate(length):
            audio_signal[idx, :, len:] = 0.0

        signal_conv = self.pos_conv(audio_signal) # B, C, T
        audio_signal += signal_conv

        audio_signal = audio_signal.transpose(1,2) #B, C, T -> B, T, C
        audio_signal = self.layer_norm(audio_signal)

        context_emb = self.apply_transformer(audio_signal, padding_mask=padding_mask)

        return context_emb, length # Returning length for NeMo compatibility
    
    def apply_transformer(self, x, padding_mask=None):
        encoder_attn_mask = form_attention_mask(padding_mask)
        if self.layer_drop and self.training: # Stochastic layer drop as in: Huang et al. https://arxiv.org/pdf/1603.09382.pdf
            for _, layer in enumerate(self.layers):
                    p = random.random()
                    if p > self.layer_drop:
                        x = layer(x, encoder_attn_mask, x)
        else:
            for _, layer in enumerate(self.layers):
                x = layer(x, encoder_attn_mask, x)
        return x

    def create_padding_mask(self, length): 
        # Broadcast to vectorize creating the padding mask
        max_len = max(length)
        padding_mask = torch.arange(max_len, device=DEVICE)

        # Switch to binary for transformer, 1 for valid tokens, 0 for padding
        padding_mask = (padding_mask.expand(len(length), max_len) < length.unsqueeze(1)).type(torch.uint8)

        return padding_mask

class GradMultiply(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale):
        ctx.scale = scale
        res = x.new(x)
        return res

    @staticmethod
    def backward(ctx, grad):
        return grad * ctx.scale, None

class Wav2VecLinearDecoder(NeuralModule, Exportable):
    """Simple ASR Decoder for linear projection of Wav2Vec embeddings
    """

    def save_to(self, save_path: str):
        pass

    @classmethod
    def restore_from(cls, restore_path: str):
        pass

    @property
    def input_types(self):
        return OrderedDict({"encoder_output": NeuralType(('B', 'T', 'D'), AcousticEncodedRepresentation())})

    @property
    def output_types(self):
        return OrderedDict({"logprobs": NeuralType(('B', 'T', 'D'), LogprobsType())})

    def __init__(self, feat_in, num_classes, init_mode="xavier_uniform", vocabulary=None):
        super().__init__()

        if vocabulary is not None:
            if num_classes != len(vocabulary):
                raise ValueError(
                    f"If vocabulary is specified, it's length should be equal to the num_classes. Instead got: num_classes={num_classes} and len(vocabulary)={len(vocabulary)}"
                )
            self.__vocabulary = vocabulary
        self._feat_in = feat_in
        # Add 1 for blank char
        self._num_classes = num_classes + 1

        self.projection = torch.nn.Linear(self._feat_in, self._num_classes, bias=False)
        self.apply(lambda x: init_weights(x, mode=init_mode))

    @typecheck()
    def forward(self, encoder_output):
        return torch.nn.functional.log_softmax(self.projection(encoder_output), dim=-1)

    def input_example(self):
        """
        Generates input examples for tracing etc.
        Returns:
            A tuple of input examples.
        """
        bs = 8
        seq = 64
        input_example = torch.randn(bs, self._feat_in, seq).to(next(self.parameters()).device)
        return tuple([input_example])

    def _prepare_for_export(self, **kwargs):
        pass

    @property
    def vocabulary(self):
        return self.__vocabulary

    @property
    def num_classes_with_blank(self):
        return self._num_classes

class Wav2VecLinearReconstruction(NeuralModule, Exportable):
    """ ASR Decoder for reconstructing masked regions of spectrogram
    """
    @property
    def input_types(self):
        return OrderedDict({"encoder_output": NeuralType(('B', 'T', 'D'), AcousticEncodedRepresentation())})

    @property
    def output_types(self):
        return OrderedDict({"audio_recon": NeuralType(('B', 'T', 'D'), SpectrogramType())})

    
    def __init__(self, feat_in, feat_out, init_mode="xavier_uniform", activation=None):
        super().__init__()

        self.feat_in = feat_in
        self.feat_out = feat_out

        self.projection = torch.nn.Linear(self.feat_in, self.feat_out, bias=False)

        if activation:
            self.projection = nn.Sequential(self.projection, jasper_activations[activation]())

        self.apply(lambda x: init_weights(x, mode=init_mode))
    
    @typecheck()
    def forward(self, encoder_output):
        return self.projection(encoder_output)