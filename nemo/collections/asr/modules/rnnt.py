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

from typing import Any, Dict, List, Optional, Tuple

import torch

from nemo.collections.asr.parts import rnn, rnnt_utils
from nemo.core.classes import typecheck
from nemo.core.neural_types import (
    AcousticEncodedRepresentation,
    ElementType,
    EmbeddedTextType,
    LabelsType,
    LengthsType,
    LogprobsType,
    NeuralType,
)


class RNNTDecoder(rnnt_utils.AbstractRNNTDecoder):
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
            "states": NeuralType(('D', 'B', 'D'), ElementType(), optional=True),
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
        vocab_size: int,
        normalization_mode: Optional[str] = None,
        random_state_sampling: bool = False,
        blank_as_pad: bool = True,
    ):
        # Required arguments
        self.pred_hidden = prednet['pred_hidden']
        self.pred_rnn_layers = prednet["pred_rnn_layers"]
        self.blank_idx = vocab_size

        # Initialize the model (blank token increases vocab size by 1)
        super().__init__(vocab_size=vocab_size, blank_idx=self.blank_idx, blank_as_pad=blank_as_pad)

        # Optional arguments
        forget_gate_bias = prednet.get('forget_gate_bias', 1.0)
        t_max = prednet.get('t_max', None)
        dropout = prednet.get('dropout', 0.0)
        self.random_state_sampling = random_state_sampling

        self.prediction = self._predict(
            vocab_size=vocab_size,  # add 1 for blank symbol
            pred_n_hidden=self.pred_hidden,
            pred_rnn_layers=self.pred_rnn_layers,
            forget_gate_bias=forget_gate_bias,
            t_max=t_max,
            norm=normalization_mode,
            dropout=dropout,
        )

    @typecheck()
    def forward(self, targets, target_length, states=None):
        # y: (B, U)
        y = rnn.label_collate(targets)

        g, _ = self.predict(y, state=states)  # (B, U + 1, D)
        g = g.transpose(1, 2)  # (B, D, U + 1)

        return g, target_length

    def predict(
        self,
        y: Optional[torch.Tensor] = None,
        state: Optional[List[torch.Tensor]] = None,
        add_sos: bool = True,
        batch_size: Optional[int] = None,
    ) -> (torch.Tensor, List[torch.Tensor]):
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
        # Get device and dtype of current module
        _p = next(self.parameters())
        device = _p.device
        dtype = _p.dtype

        if y is not None:
            # (B, U) -> (B, U, H)
            if y.device != device:
                y = y.to(device)

            y = self.prediction["embed"](y)
        else:
            if batch_size is None:
                B = 1 if state is None else state[0].size(1)
            else:
                B = batch_size
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
                state = self.initialize_state(y)

        y = y.transpose(0, 1)  # (U + 1, B, H)
        g, hid = self.prediction["dec_rnn"](y, state)
        g = g.transpose(0, 1)  # (B, U + 1, H)

        del y, start, state

        return g, hid

    def _predict(self, vocab_size, pred_n_hidden, pred_rnn_layers, forget_gate_bias, t_max, norm, dropout):
        if self.blank_as_pad:
            embed = torch.nn.Embedding(vocab_size + 1, pred_n_hidden, padding_idx=self.blank_idx)
        else:
            embed = torch.nn.Embedding(vocab_size, pred_n_hidden)

        layers = torch.nn.ModuleDict(
            {
                "embed": embed,
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

    def initialize_state(self, y: torch.Tensor) -> List[torch.Tensor]:
        batch = y.size(0)
        if self.random_state_sampling and self.training:
            state = (
                torch.randn(self.pred_rnn_layers, batch, self.pred_hidden, dtype=y.dtype, device=y.device),
                torch.randn(self.pred_rnn_layers, batch, self.pred_hidden, dtype=y.dtype, device=y.device),
            )

        else:
            state = (
                torch.zeros(self.pred_rnn_layers, batch, self.pred_hidden, dtype=y.dtype, device=y.device),
                torch.zeros(self.pred_rnn_layers, batch, self.pred_hidden, dtype=y.dtype, device=y.device),
            )
        return state

    def score_hypothesis(
        self, hypothesis: rnnt_utils.Hypothesis, cache: Dict[Tuple[int], Any]
    ) -> (torch.Tensor, List[torch.Tensor], torch.Tensor):
        """

        Args:
            hypothesis:
            cache:

        Returns:

        """
        if hypothesis.dec_state is not None:
            device = hypothesis.dec_state[0].device
            dtype = hypothesis.dec_state[0].dtype
        else:
            _p = next(self.parameters())
            device = _p.device
            dtype = _p.dtype

        # parse "blank" tokens in hypothesis
        if len(hypothesis.y_sequence) > 0 and hypothesis.y_sequence[-1] == self.blank_idx:
            blank_state = True
        else:
            blank_state = False

        target = torch.full([1, 1], fill_value=hypothesis.y_sequence[-1], device=device, dtype=torch.long)
        lm_token = target[:, -1]  # [1]

        # if blank_state:
        #     hypothesis.y_sequence.pop()

        sequence = tuple(hypothesis.y_sequence)

        if sequence in cache:
            y, new_state = cache[sequence]
        else:
            if blank_state:
                y, new_state = self.predict(None, state=None, add_sos=False, batch_size=1)  # [1, U, H]

            else:
                y, new_state = self.predict(
                    target, state=hypothesis.dec_state, add_sos=False, batch_size=1
                )  # [1, U, H]

            y = y[:, -1:, :]  # U[1, 1, H]
            cache[sequence] = (y, new_state)

        return y, new_state, lm_token

    def batch_score_hypothesis(
        self, hypotheses: List[rnnt_utils.Hypothesis], cache: Dict[Tuple[int], Any], batch_states: List[torch.Tensor]
    ) -> (torch.Tensor, List[torch.Tensor], torch.Tensor):
        """

        Args:
            hypotheses:
            cache:
            batch_states:

        Returns:

        """
        final_batch = len(hypotheses)

        if final_batch == 0:
            raise ValueError("No hypotheses was provided for the batch!")

        _p = next(self.parameters())
        device = _p.device
        dtype = _p.dtype

        tokens = []
        process = []
        done = [None for _ in range(final_batch)]

        for i, hyp in enumerate(hypotheses):
            sequence = tuple(hyp.y_sequence)

            if sequence in cache:
                done[i] = cache[sequence]
            else:
                tokens.append(hyp.y_sequence[-1])
                process.append((sequence, hyp.dec_state))

        if process:
            batch = len(process)

            # pad tokens
            tokens = torch.tensor(tokens, device=device, dtype=torch.long).view(batch, -1)
            dec_states = self.initialize_state(tokens.to(dtype=dtype))  # [L, B, D]
            dec_states = self.batch_initialize_states(dec_states, [d_state for seq, d_state in process])

            y, dec_states = self.predict(
                tokens, state=dec_states, add_sos=False, batch_size=batch
            )  # [B, U + 1, H], List([L, B, H])

        j = 0
        for i in range(final_batch):
            if done[i] is None:
                new_state = self.batch_select_state(dec_states, j)

                done[i] = (y[j], new_state)
                cache[process[j][0]] = (y[j], new_state)

                j += 1

        batch_states = self.batch_initialize_states(batch_states, [d_state for y_j, d_state in done])
        batch_y = torch.stack([y_j for y_j, d_state in done])

        lm_tokens = torch.tensor([h.y_sequence[-1] for h in hypotheses], device=device, dtype=torch.long).view(
            final_batch
        )

        return batch_y, batch_states, lm_tokens

    def batch_initialize_states(self, batch_states: List[torch.Tensor], decoder_states: List[List[torch.Tensor]]):
        """Create batch of decoder states.
           Args:
               batch_states (tuple): batch of decoder states
                  ([L x (B, dec_dim)], [L x (B, dec_dim)])
               decoder_states (list): list of decoder states
                   [B x ([L x (1, dec_dim)], [L x (1, dec_dim)])]
           Returns:
               batch_states (tuple): batch of decoder states
                   ([L x (B, dec_dim)], [L x (B, dec_dim)])
       """
        # LSTM has 2 states
        for layer in range(self.pred_rnn_layers):
            for state_id in range(len(batch_states)):
                batch_states[state_id][layer] = torch.stack([s[state_id][layer] for s in decoder_states])

        return batch_states

    def batch_select_state(self, batch_states: List[torch.Tensor], idx: int) -> List[List[torch.Tensor]]:
        """Get decoder state from batch of states, for given id.
        Args:
            batch_states (tuple): batch of decoder states
                ([L x (B, dec_dim)], [L x (B, dec_dim)])
            idx (int): index to extract state from batch of states
        Returns:
            (tuple): decoder states for given id
                ([L x (1, dec_dim)], [L x (1, dec_dim)])
        """
        state_list = []
        for state_id in range(len(batch_states)):
            states = [batch_states[state_id][layer][idx] for layer in range(self.pred_rnn_layers)]
            state_list.append(states)

        return state_list


class RNNTJoint(rnnt_utils.AbstractRNNTJoint):
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
            "outputs": NeuralType(('B', 'T', 'D', 'D'), LogprobsType()),
        }

    def __init__(
        self,
        jointnet: Dict[str, Any],
        num_classes: int,
        vocabulary: Optional[List] = None,
        log_softmax: Optional[bool] = None,
        preserve_memory: bool = False,
    ):
        super().__init__()

        self.vocabulary = vocabulary

        self._vocab_size = num_classes
        self._num_classes = num_classes + 1  # add 1 for blank symbol

        # Log softmax should be applied explicitly only for CPU
        self.log_softmax = log_softmax
        self.preserve_memory = preserve_memory

        # Required arguments
        self.encoder_hidden = jointnet['encoder_hidden']
        self.pred_hidden = jointnet['pred_hidden']
        self.joint_hidden = jointnet['joint_hidden']
        self.activation = jointnet['activation']

        # Optional arguments
        dropout = jointnet.get('dropout', 0.0)

        self.pred, self.enc, self.joint_net = self._joint_net(
            num_classes=self._num_classes,  # add 1 for blank symbol
            pred_n_hidden=self.pred_hidden,
            enc_n_hidden=self.encoder_hidden,
            joint_n_hidden=self.joint_hidden,
            activation=self.activation,
            dropout=dropout,
        )

    @typecheck()
    def forward(self, encoder_outputs: torch.Tensor, decoder_outputs: torch.Tensor) -> torch.Tensor:
        # encoder = (B, D, T)
        # decoder = (B, D, U + 1)
        encoder_outputs = encoder_outputs.transpose(1, 2)  # (B, T, D)
        decoder_outputs = decoder_outputs.transpose(1, 2)  # (B, U + 1, D)

        out = self.joint(encoder_outputs, decoder_outputs)  # [B, T, U, K + 1]
        return out

    def joint(self, f: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
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

    def _joint_net(self, num_classes, pred_n_hidden, enc_n_hidden, joint_n_hidden, activation, dropout):
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
            + [torch.nn.Linear(joint_n_hidden, num_classes)]
        )
        return pred, enc, torch.nn.Sequential(*layers)

    @property
    def num_classes_with_blank(self):
        return self._num_classes
