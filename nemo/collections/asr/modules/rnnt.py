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

from typing import Any, Dict, Optional

import torch

from nemo.collections.asr.parts import rnn
from nemo.core.classes import NeuralModule, typecheck
from nemo.core.neural_types import (
    AcousticEncodedRepresentation,
    EmbeddedTextType,
    LabelsType,
    LengthsType,
    LogitsType,
    NeuralType,
)
from nemo.utils import logging


class RNNTDecoder(NeuralModule):
    """A Recurrent Neural Network Transducer (RNN-T).
    Args:
        in_features: Number of input features per step per batch.
        vocab_size: Number of output symbols (inc blank).
        forget_gate_bias: Total initialized value of the bias used in the
            forget gate. Set to None to use PyTorch's default initialisation.
            (See: http://proceedings.mlr.press/v37/jozefowicz15.pdf)
        batch_norm: Use batch normalization in encoder and prediction network
            if true.
        pred_n_hidden:  Internal hidden unit size of the prediction network.
        pred_rnn_layers: Prediction network number of layers.
        rnn_type: string. Type of rnn in SUPPORTED_RNNS.
    """

    @property
    def input_types(self):
        """Returns definitions of module input ports.
        """
        return {
            "targets": NeuralType(('B', 'T'), LabelsType()),
            "target_length": NeuralType(tuple('B'), LengthsType()),
        }

    @property
    def output_types(self):
        """Returns definitions of module output ports.
        """
        return {
            "outputs": NeuralType(('B', 'D', 'T'), EmbeddedTextType()),
            "encoded_lengths": NeuralType(tuple('B'), LengthsType()),
        }

    def __init__(
        self,
        prednet: Dict[str, Any],
        num_classes: int,
        normalization_mode: Optional[str] = None,
        random_state_sampling: bool = False,
    ):
        super().__init__()

        self.random_state_sampling = random_state_sampling

        # Required arguments
        self.pred_hidden = prednet['pred_hidden']
        self.pred_rnn_layers = prednet["pred_rnn_layers"]

        # Optional arguments
        forget_gate_bias = prednet.get('forget_gate_bias', 1.0)
        t_max = prednet.get('t_max', None)
        dropout = prednet.get('dropout', 0.0)

        self.prediction = self._predict(
            vocab_size=num_classes + 1,  # add 1 for blank symbol
            pred_n_hidden=self.pred_hidden,
            pred_rnn_layers=self.pred_rnn_layers,
            forget_gate_bias=forget_gate_bias,
            t_max=t_max,
            norm=normalization_mode,
            dropout=dropout,
        )

    @typecheck()
    def forward(self, targets, target_length):
        # y: (B, U)
        y = rnn.label_collate(targets)

        g, _ = self.predict(y, state=None, device=y.device)  # (B, U + 1, D)
        g = g.transpose(1, 2)  # (B, D, U + 1)

        return g, target_length

    def predict(
        self,
        y: Optional[torch.Tensor],
        state: Optional[torch.Tensor] = None,
        add_sos: bool = True,
        batch_size: int = None,
    ):
        """
        B - batch size
        U - label length
        H - Hidden dimension size
        L - Number of decoder layers = 2
        Args:
            y: (B, U)
        Returns:
            Tuple (g, hid) where:
                g: (B, U + 1, H)
                hid: (h, c) where h is the final sequence hidden state and c is
                    the final cell state:
                        h (tensor), shape (L, B, H)
                        c (tensor), shape (L, B, H)
        """
        if y is not None:
            # (B, U) -> (B, U, H)
            y = self.prediction["embed"](y)
        else:
            if batch_size is None:
                B = 1 if state is None else state[0].size(1)
            else:
                B = batch_size

            device = self.prediction['dec_rnn'].device
            dtype = self.prediction['dec_rnn'].dtype
            y = torch.zeros((B, 1, self.pred_hidden), device=device, dtype=dtype)

        # prepend blank "start of sequence" symbol
        if add_sos:
            B, U, H = y.shape
            start = torch.zeros((B, 1, H), device=y.device, dtype=y.dtype)
            y = torch.cat([start, y], dim=1).contiguous()  # (B, U + 1, H)
        else:
            start = None  # makes del call later easier

        if state is None:
            if self.random_state_sampling and self.training:
                batch = y.size(0)
                state = (
                    torch.randn(self.pred_rnn_layers, batch, self.pred_hidden, dtype=y.dtype, device=y.device),
                    torch.randn(self.pred_rnn_layers, batch, self.pred_hidden, dtype=y.dtype, device=y.device),
                )

        y = y.transpose(0, 1)  # (U + 1, B, H)
        g, hid = self.prediction["dec_rnn"](y, state)
        g = g.transpose(0, 1)  # (B, U + 1, H)

        del y, start, state

        return g, hid

    def _predict(self, vocab_size, pred_n_hidden, pred_rnn_layers, forget_gate_bias, t_max, norm, dropout):
        layers = torch.nn.ModuleDict(
            {
                "embed": torch.nn.Embedding(vocab_size - 1, pred_n_hidden),
                "dec_rnn": rnn.rnn(
                    input_size=pred_n_hidden,
                    hidden_size=pred_n_hidden,
                    num_layers=pred_rnn_layers,
                    norm=norm,
                    forget_gate_bias=forget_gate_bias,
                    t_max=t_max,
                    dropout=dropout,
                ),
            }
        )
        return layers


class RNNTJoint(NeuralModule):
    """A Recurrent Neural Network Transducer (RNN-T).
    Args:
        in_features: Number of input features per step per batch.
        vocab_size: Number of output symbols (inc blank).
        forget_gate_bias: Total initialized value of the bias used in the
            forget gate. Set to None to use PyTorch's default initialisation.
            (See: http://proceedings.mlr.press/v37/jozefowicz15.pdf)
        batch_norm: Use batch normalization in encoder and prediction network
            if true.
        encoder_n_hidden: Internal hidden unit size of the encoder.
        encoder_rnn_layers: Encoder number of layers.
        pred_n_hidden:  Internal hidden unit size of the prediction network.
        pred_rnn_layers: Prediction network number of layers.
        joint_n_hidden: Internal hidden unit size of the joint network.
        rnn_type: string. Type of rnn in SUPPORTED_RNNS.
    """

    @property
    def input_types(self):
        """Returns definitions of module input ports.
        """
        return {
            "encoder_outputs": NeuralType(('B', 'D', 'T'), AcousticEncodedRepresentation()),
            "decoder_outputs": NeuralType(('B', 'D', 'T'), EmbeddedTextType()),
        }

    @property
    def output_types(self):
        """Returns definitions of module output ports.
        """
        return {
            "outputs": NeuralType(('B', 'T', 'D', 'D'), LogitsType()),
        }

    def __init__(
        self,
        joint: Dict[str, Any],
        num_classes: int,
        log_softmax: Optional[bool] = None,
        preserve_memory: bool = False,
    ):
        super().__init__()

        # Log softmax should be applied explicitly only for CPU
        self.log_softmax = log_softmax
        self.preserve_memory = preserve_memory

        # Required arguments
        encoder_hidden = joint['encoder_hidden']
        pred_hidden = joint['pred_hidden']
        joint_hidden = joint['joint_hidden']
        activation = joint['activation']

        # Optional arguments
        dropout = joint.get('dropout', 0.0)

        self.pred, self.enc, self.joint_net = self._joint_net(
            vocab_size=num_classes + 1,  # add 1 for blank symbol
            pred_n_hidden=pred_hidden,
            enc_n_hidden=encoder_hidden,
            joint_n_hidden=joint_hidden,
            activation=activation,
            dropout=dropout,
        )

    @typecheck()
    def forward(self, encoder_outputs: torch.Tensor, decoder_outputs: torch.Tensor) -> torch.Tensor:
        # encoder = (B, D, T)
        # decoder = (B, D, U + 1)
        encoder_outputs = encoder_outputs.transpose(1, 2)  # (B, T, D)
        decoder_outputs = decoder_outputs.transpose(1, 2)  # (B, U + 1, D)

        out = self.joint(encoder_outputs, decoder_outputs)  # [B, T, U, K + 1]

        # encoder_outputs.transpose_(1, 2)  # (B, D, T)
        # decoder_outputs.transpose_(1, 2)  # (B, D, U + 1)
        return out

    def joint(self, f: torch.Tensor, g: torch.Tensor):
        """
        f should be shape (B, T, H)
        g should be shape (B, U + 1, H)
        returns:
            logits of shape (B, T, U, K + 1)
        """
        f = self.enc(f)
        f.unsqueeze_(dim=2)  # (B, T, 1, D)

        g = self.pred(g)
        g.unsqueeze_(dim=1)  # (B, 1, U + 1, H)

        inp = f + g
        del f, g

        res = self.joint_net(inp)
        del inp

        if self.preserve_memory:
            torch.cuda.empty_cache()

        # If log_softmax is automatic
        if self.log_softmax is None:
            if not res.is_cuda:  # Use log softmax only if on CPU
                res = res.log_softmax(dim=-1)
        else:
            if self.log_softmax:
                res = res.log_softmax(dim=-1)

        return res

    def _joint_net(self, vocab_size, pred_n_hidden, enc_n_hidden, joint_n_hidden, activation, dropout):
        pred = torch.nn.Linear(pred_n_hidden, joint_n_hidden)
        enc = torch.nn.Linear(enc_n_hidden, joint_n_hidden)

        if activation not in ['relu', 'sigmoid', 'tanh']:
            raise ValueError("Unsupported activation for joint step - please pass one of " "[relu, sigmoid, tanh]")

        activation = activation.lower()

        if activation == 'relu':
            activation = torch.nn.ReLU(inplace=True)
        elif activation == 'sigmoid':
            activation = torch.nn.Sigmoid()
        elif activation == 'tanh':
            activation = torch.nn.Tanh()

        layers = (
            [activation]
            + ([torch.nn.Dropout(p=dropout)] if dropout else [])
            + [torch.nn.Linear(joint_n_hidden, vocab_size)]
        )
        return pred, enc, torch.nn.Sequential(*layers)
