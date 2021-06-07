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

import torch

from nemo.core.classes import NeuralModule, typecheck
from nemo.core.neural_types import LengthsType, LogprobsType, NeuralType, PredictionsType


class ViterbiDecoderWithGraph(NeuralModule):

    @property
    def input_types(self):
        """Returns definitions of module input ports.
        """
        return {
            "log_probs": NeuralType(('B', 'T', 'D'), LogprobsType()),
            "log_probs_length": NeuralType(tuple('B'), LengthsType()),
        }

    @property
    def output_types(self):
        """Returns definitions of module output ports.
        """
        return {"predictions": NeuralType(('B', 'T'), PredictionsType())}

    def __init__(self, num_classes, backend='k2', dec_type='tokenlm', return_type='1best', output_aligned=False, **decode_kwargs):
        self._blank = num_classes
        self.output_aligned = output_aligned

        if return_type == '1best':
            self.return_lattices = False
        elif return_type == 'lattice':
            self.return_lattices = True
        elif return_type == 'nbest':
            raise NotImplementedError
        else:
            raise ValueError

        # we assume that self._blank + 1 == num_classes
        if backend == 'k2':
            if dec_type == 'tokenlm':
                from nemo.collections.asr.parts.k2.graph_decoders import TokenLMDecoder as Decoder
            elif dec_type == 'tlg':
                from nemo.collections.asr.parts.k2.graph_decoders import TLGDecoder as Decoder

            self._decoder = Decoder(num_classes=self._blank+1, blank=self._blank, **decode_kwargs)
        elif backend == 'gtn':
            raise NotImplementedError("gtn-backed decoding is not implemented")

        super().__init__()

    @torch.no_grad()
    def forward(self, log_probs, log_probs_length):
        return self._decoder.decode(log_probs, log_probs_length, return_lattices=self.return_lattices, return_ilabels=self.output_aligned)
