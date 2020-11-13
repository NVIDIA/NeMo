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
import numpy as np
import torch
import torch.nn as nn

from nemo.collections.tts.modules import fastspeech2_modules
from nemo.core.classes import NeuralModule, typecheck

""" From the paper:
Our FastSpeech 2 consists of 4 feed-forward Transformer (FFT) blocks [20]
in the encoder and the mel-spectrogram decoder. In each FFT block, the dimension of phoneme
embeddings and the hidden size of the self-attention are set to 256. The number of attention heads
is set to 2 and the kernel sizes of the 1D-convolution in the 2-layer convolutional network after
the self-attention layer are set to 9 and 1, with input/output size of 256/1024 for the first layer and
1024/256 in the second layer. The output linear layer converts the 256-dimensional hidden states into
80-dimensional mel-spectrograms and optimized with mean absolute error (MAE). The size of the
phoneme vocabulary is 76, including punctuations. In the variance predictor, the kernel sizes of the
1D-convolution are set to 3, with input/output sizes of 256/256 for both layers and the dropout rate
is set to 0.5. Our waveform decoder consists of 1-layer transposed 1D-convolution with filter size
64 and 30 layers of dilated residual convolution blocks, whose skip channel size and kernel size of
1D-convolution are set to 64 and 3. The configurations of the discriminator in FastSpeech 2s are the
same as Parallel WaveGAN [27]. We list hyperparameters and configurations of all models used in
our experiments in Appendix A."""

# Very WIP.
# Hyperparams hard-coded for now to the best of my understanding

@experimental
class Encoder(NeuralModule):
    def __init__(self):
        """
        FastSpeech 2 encoder. Converts phoneme sequence to the phoneme hidden sequence.
        Consists of a phoneme embedding lookup, positional encoding, and four feed-forward
        Transformer blocks.

        Args:
        """
        #TODO: documentation of params
        super().__init__()

        self.encoder = fastspeech2_modules.FFTransformer(
            n_layer=4,
            n_head=2,
            d_model=256,
            d_head=256,
            d_inner=1024,
            kernel_size=(9, 1),
            dropout=0.2,
            dropatt=0.1,    #??? Not sure if this is right, don't see it in paper
            embed_input=True    # For the encoder, need to do embedding lookup
        )

    @property
    def input_types(self):  # phonemes
        return {
            "text": NeuralType(('B', 'T'), TokenIndex()),
            "text_lengths": NeuralType(('B'), LengthsType())
        }

    @property
    def output_types(self):
        return {
            "encoder_embedding": NeuralType(('B', 'T', 'D'), EmbeddedTextType()),
        }

    @typecheck()
    def forward(self, *, text, text_lengths):
        self.encoder(text, seq_lens=text_lengths)


@experimental
class VarianceAdaptor(NeuralModuel):
    def __init__(self, max_duration, pitch_min, pitch_max, energy_min, energy_max):
        """
        FastSpeech 2 variance adaptor, which adds information like duration, pitch, etc. to the phoneme encoding.
        Sets of conv1D blocks with ReLU and dropout.

        Args:
        """
        #TODO: documentation of params
        super().__init__()
        #TODO: need to set all the other default min/max params - based on dataset?

        """In the variance predictor, the kernel sizes of the
        1D-convolution are set to 3, with input/output sizes of 256/256 for both layers and the dropout rate
        is set to 0.5."""
        # -- Duration Setup --
        #TODO: what should this max duration be? should this be set at all?
        self.max_duration = max_duration    
        self.duration_predictor = fastspeech2_modules.VariancePredictor(
            d_model=256,
            d_inner=256,
            kernel_size=3,
            dropout=0.5
        )
        self.length_regulator = fastspeech2_modules.LengthRegulator()

        # -- Pitch Setup --
        self.register_buffer(   # Log scale bins
            "pitch_bins",
            torch.exp(
                torch.linspace(start=np.log(pitch_min), end=np.log(pitch_max), steps=255)   # n_f0_bins - 1
            )
        )
        self.pitch_predictor = fastspeech2_modules.VariancePredictor(
            d_model=256,    # va_hidden_size
            d_inner=256,    # n_f0_bins
            kernel_size=3,
            dropout=0.5
        )
        # Predictor outputs values directly rather than one-hot vectors, therefore Embedding
        self.pitch_lookup = nn.Embedding(256, 256) # f0_bins, va_hidden_size

        # -- Energy Setup --
        self.register_buffer(    # Linear scale bins
            "energy_bins",
            torch.linspace(start=energy_min, end=energy_max, steps=255)     # n_energy_bins - 1
        )
        self.energy_predictor = fastspeech2_modules.VariancePredictor(
            d_model=256,    # va_hidden_size
            d_inner=256,    # n_energy_bins
            kernel_size=3,
            dropout=0.5
        )
        self.energy_lookup = nn.Embedding(256, 256) # n_energy_bins, va_hidden_size

    @property
    def input_types(self):
        return {
            "encoder_embedding": NeuralType(('B', 'T', 'D'), EmbeddedTextType()),
        }

    @property
    def output_types(self):
        return {
            # Might need a better name and type for this
            "variance_embedding": NeuralType(('B', 'T', 'D'), EncodedRepresentation()),
        }

    @typecheck()
    def forward(self, *, x, dur_target=None, pitch_target=None, energy_target=None):
        """
        Args:
            dur_target: Needs to be passed in during training. Duration targets for the duration predictor.
        """
        #TODO: no_grad instead of self.training?
        #TODO: or maybe condition on a new parameter like is_inference?

        # Duration predictions (or ground truth) fed into Length Regulator to
        # expand the hidden states of the encoder embedding
        log_dur_preds = self.duration_predictor(x)
        if self.training:
            dur_out = self.length_regulator(x, dur_target)
        else:
            dur_preds = torch.clamp(torch.round(torch.exp(log_durations) - 1), min=0, max=self.max_duration)
            dur_out = self.length_regulator(x, dur_preds)

        # Pitch
        pitch_preds = self.pitch_predictor(dur_out)
        if self.training:
            pitch_out = self.pitch_lookup(torch.bucketize(pitch_target, self.pitch_bins))
        else:
            pitch_out = self.pitch_lookup(torch.bucketize(pitch_preds, self.pitch_bins))

        # Energy
        energy_preds = self.energy_predictor(dur_out)
        if self.training:
            energy_out = self.energy_lookup(torch.bucketize(energy_target, self.energy_bins))
        else:
            energy_out = self.energy_lookup(torch.bucketize(energy_preds, self.energy_bins))

        out = dur_out + pitch_out + energy_out
        return out


@experimental
class MelSpecDecoder(NeuralModule):
    def __init__(self):
        """
        FastSpeech 2 mel-spectrogram decoder. Converts adapted hidden sequence to a mel-spectrogram sequence.
        Consists of four feed-forward Transformer blocks.

        Args:
        """
        super().__init__()

        self.decoder = fastspeech2_modules.FFTransformer(
            n_layer=4,
            n_head=2,
            d_model=384,    # Some paragraphs say 256, the table in the appendix says 384
            d_head=384,
            d_inner=1024,
            kernel_size=(9, 1),
            dropout=0.2,
            dropatt=0.1,    #??? Not mentioned? Or am I just blind?
            embed_input=False
        )
        self.linear = nn.Linear(384, 80)

    @property
    def input_types(self):
        #TODO
        pass

    @property
    def output_types(self):
        #TODO
        pass

    @typecheck()
    def forward(self, *, decoder_input):
        decoder_out = self.decoder(decoder_input)
        mel_out = self.linear(decoder_out)
        return mel_out


@experimental
class WaveformDecoder(NeuralModule):
    def __init__(
        self,
        in_channels=256,
        out_channels=256,   # See WaveNet Section 2.2 regarding waveform quantization
        gen_trans_kernel_size=64,
        gen_n_layers=30,
        gen_dilation_cycle=3,
        gen_dilated_kernel_size=3,
        gen_residual_channels=64,
        gen_skip_channels=64,
        dis_n_layers=10,
        dis_relu_alpha=0.2,
        dis_conv_channels=64,
        dis_kernel_size=3,
    ):
        """
        FastSpeech 2 waveform decoder. Converts adapted hidden sequence to a waveform sequence.
        Consists of one transposed conv1d layer, and 30 layers of dilated residual conv blocks.

        Args:
            in_channels (int): Number of input channels to the waveform decoder
            out_channels (int)
            gen_trans_kernel_size (int): Filter size of the upsampling transposed 1D convolution in the generator
            gen_n_layers (int): Number of layers of dilated residual convolutional blocks in the generator
            gen_dilation_cycle (int): The number of layers with a given dilation before moving up by a power of two.
                `n_layers` should be divisible by `dilation_cycle`.
            gen_dilated_kernel_size (int): Kernel size for the dilated conv1Ds in the generator
            gen_residual_channels (int)
            gen_skip_channels (int)
        """
        super().__init__()

        """ From the paper:
        Our waveform decoder consists of 1-layer transposed 1D-convolution with filter size
        64 and 30 layers of dilated residual convolution blocks, whose skip channel size and kernel size of
        1D-convolution are set to 64 and 3
        """
        # Waveform generator
        self.generator = fastspeech2_submodules.WaveformGenerator(
            in_channels=in_channels,
            out_channels=out_channels,
            trans_kernel_size=gen_trans_kernel_size,
            n_layers=gen_n_layers,
            dilation_cycle=gen_dilation_cycle,
            dilated_kernel_size=gen_dilated_kernel_size,
            residual_channels=gen_residual_channels,
            skip_channels=gen_skip_channels,
        )

    @property
    def input_types(self):
        #TODO
        pass

    @property
    def output_types(self):
        #TODO
        pass

    @typecheck()
    def forward(self, *, decoder_input):
        generator_output = self.generator(decoder_input)
        pass
