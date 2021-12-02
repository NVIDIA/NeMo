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

    def __init__(self, num_classes, backend='k2', dec_type='tokenlm', return_type='1best', output_aligned=False, split_batch_size=0, **decode_kwargs):
        self._blank = num_classes
        self.output_aligned = output_aligned
        self.split_batch_size = split_batch_size

        if return_type == '1best':
            self.return_lattices = False
        elif return_type == 'lattice':
            self.return_lattices = True
        elif return_type == 'nbest':
            raise NotImplementedError(f"return_type {return_type} is not supported at the moment")
        else:
            raise ValueError

        # we assume that self._blank + 1 == num_classes
        if backend == 'k2':
            if dec_type == 'tokenlm':
                from nemo.collections.asr.parts.k2.graph_decoders import TokenLMDecoder as Decoder
            elif dec_type == 'tlg':
                raise NotImplementedError(f"dec_type {dec_type} is not supported at the moment")
            else:
                raise ValueError(f"Unsupported dec_type: {dec_type}")

            self._decoder = Decoder(num_classes=self._blank+1, blank=self._blank, **decode_kwargs)
        elif backend == 'gtn':
            raise NotImplementedError("gtn-backed decoding is not implemented")

        super().__init__()

    def update_graph(self, graph):
        self._decoder.update_graph(graph)

    @torch.no_grad()
    def forward(self, log_probs, log_probs_length):
        # do not use self.return_lattices and self.output_aligned for now
        batch_size = log_probs.shape[0]
        if self.split_batch_size > 0 and self.split_batch_size < batch_size:
            predictions_list = []
            scores_list = []
            for batch_idx in range(0, batch_size, self.split_batch_size):
                begin = batch_idx
                end = min(begin + self.split_batch_size, batch_size)
                log_probs_part = log_probs[begin:end]
                log_probs_length_part = log_probs_length[begin:end]
                predictions_part, scores_part = self._decoder.decode(log_probs_part, log_probs_length_part, return_lattices=False, return_ilabels=True)
                predictions_list += predictions_part
                scores_list.append(scores_part)
                del log_probs_part, log_probs_length_part
            predictions = predictions_list
            scores = torch.cat(scores_list, 0)
        else:
            predictions, scores = self._decoder.decode(log_probs, log_probs_length, return_lattices=False, return_ilabels=True)
        lengths = torch.tensor([len(pred) for pred in predictions], device=predictions[0].device)
        predictions_tensor = torch.full((len(predictions), lengths.max()), self._blank).to(device=predictions[0].device)
        for i, pred in enumerate(predictions):
            predictions_tensor[i,:lengths[i]] = pred
        return predictions_tensor, lengths, scores
