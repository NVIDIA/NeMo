from collections import OrderedDict

import torch

from nemo.collections.asr.modules.conv_asr import ConvASRDecoder
from nemo.collections.asr.parts.submodules.jasper import init_weights
from nemo.core.classes import NeuralModule, typecheck
from nemo.core.neural_types import (
    AcousticEncodedRepresentation,
    LengthsType,
    LogitsType,
    LogprobsType,
    NeuralType,
    SpectrogramType,
)


class MultiSoftmaxDecoder(NeuralModule):
    @property
    def input_types(self):
        return OrderedDict({"encoder_output": NeuralType(('B', 'D', 'T'), AcousticEncodedRepresentation())})

    @property
    def output_types(self):
        if self.squeeze_single and self.num_decoders == 1:
            return OrderedDict({"logprobs": NeuralType(('B', 'T', 'C'), LogprobsType())})
        return OrderedDict({"logprobs": NeuralType(('B', 'T', 'C', 'H'), LogprobsType())})

    def __init__(
        self,
        feat_in: int,
        num_classes: int,
        num_decoders: int = 1,
        init_mode: str = "xavier_uniform",
        use_bias: bool = True,
        squeeze_single: bool = False,
    ) -> None:
        super().__init__()
        self.feat_in = feat_in
        self.num_classes = num_classes
        self.num_decoders = num_decoders
        self.squeeze_single = squeeze_single

        self.decoder_layers = torch.nn.Sequential(
            torch.nn.Conv1d(self.feat_in, self.num_classes * self.num_decoders, kernel_size=1, bias=use_bias)
        )
        self.apply(lambda x: init_weights(x, mode=init_mode))

    @typecheck()
    def forward(self, encoder_output):
        logits = self.decoder_layers(encoder_output).transpose(1, 2)
        logits = logits.reshape(logits.shape[0], logits.shape[1], self.num_classes, self.num_decoders)
        if self.squeeze_single and self.num_decoders == 1:
            logits = logits.squeeze(-1)

        return torch.nn.functional.log_softmax(logits, dim=2)
