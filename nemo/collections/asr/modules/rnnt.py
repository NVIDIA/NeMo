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

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
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
from nemo.utils import logging


class RNNTDecoder(rnnt_utils.AbstractRNNTDecoder):
    """A Recurrent Neural Network Transducer Decoder / Prediction Network (RNN-T Prediction Network).
    An RNN-T Decoder/Prediction network, comprised of a stateful LSTM model.

    Args:
        prednet: A dict-like object which contains the following key-value pairs.
            pred_hidden: int specifying the hidden dimension of the prediction net.
            pred_rnn_layers: int specifying the number of rnn layers.

            Optionally, it may also contain the following:
            forget_gate_bias: float, set by default to 1.0, which constructs a forget gate
                initialized to 1.0.
                Reference:
                [An Empirical Exploration of Recurrent Network Architectures](http://proceedings.mlr.press/v37/jozefowicz15.pdf)
            t_max: int value, set to None by default. If an int is specified, performs Chrono Initialization
                of the LSTM network, based on the maximum number of timesteps `t_max` expected during the course
                of training.
                Reference:
                [Can recurrent neural networks warp time?](https://openreview.net/forum?id=SJcKhk-Ab)
            dropout: float, set to 0.0 by default. Optional dropout applied at the end of the final LSTM RNN layer.

        vocab_size: int, specifying the vocabulary size of the embedding layer of the Prediction network,
            excluding the RNNT blank token.

        normalization_mode: Can be either None, 'batch' or 'layer'. By default, is set to None.
            Defines the type of normalization applied to the RNN layer.

        random_state_sampling: bool, set to False by default. When set, provides normal-distribution
            sampled state tensors instead of zero tensors during training.
            Reference:
            [Recognizing long-form speech using streaming end-to-end models](https://arxiv.org/abs/1910.11455)

        blank_as_pad: bool, set to True by default. When set, will add a token to the Embedding layer of this
            prediction network, and will treat this token as a pad token. In essence, the RNNT pad token will
            be treated as a pad token, and the embedding layer will return a zero tensor for this token.

            It is set by default as it enables various batch optimizations required for batched beam search.
            Therefore, it is not recommended to disable this flag.
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

        # state maintenance is unnecessary during training forward call
        # to get state, use .predict() method.
        g, _ = self.predict(y, state=states)  # (B, U, D)
        g = g.transpose(1, 2)  # (B, D, U)

        return g, target_length

    def predict(
        self,
        y: Optional[torch.Tensor] = None,
        state: Optional[List[torch.Tensor]] = None,
        add_sos: bool = True,
        batch_size: Optional[int] = None,
    ) -> (torch.Tensor, List[torch.Tensor]):
        """
        Stateful prediction of scores and state for a (possibly null) tokenset.
        This method takes various cases into consideration :
        - No token, no state - used for priming the RNN
        - No token, state provided - used for blank token scoring
        - Given token, states - used for scores + new states

        Here:
        B - batch size
        U - label length
        H - Hidden dimension size of RNN
        L - Number of RNN layers

        Args:
            y: Optional torch tensor of shape [B, U] of dtype long which will be passed to the Embedding.
                If None, creates a zero tensor of shape [B, 1, H] which mimics output of pad-token on Embedding.

            state: An optional list of states for the RNN. Eg: For LSTM, it is the state list length is 2.
                Each state must be a tensor of shape [L, B, H].
                If None, and during training mode and `random_state_sampling` is set, will sample a
                normal distribution tensor of the above shape. Otherwise, None will be passed to the RNN.

            add_sos: bool flag, whether a zero vector describing a "start of signal" token should be
                prepended to the above "y" tensor. When set, output size is (B, U + 1, H).

            batch_size: An optional int, specifying the batch size of the `y` tensor.
                Can be infered if `y` and `state` is None. But if both are None, then batch_size cannot be None.

        Returns:
            A tuple  (g, hid) such that -

            If add_sos is False:
                g: (B, U, H)
                hid: (h, c) where h is the final sequence hidden state and c is the final cell state:
                    h (tensor), shape (L, B, H)
                    c (tensor), shape (L, B, H)

            If add_sos is True:
                g: (B, U + 1, H)
                hid: (h, c) where h is the final sequence hidden state and c is the final cell state:
                    h (tensor), shape (L, B, H)
                    c (tensor), shape (L, B, H)

        """
        # Get device and dtype of current module
        _p = next(self.parameters())
        device = _p.device
        dtype = _p.dtype

        # If y is not None, it is of shape [B, U] with dtype long.
        if y is not None:
            if y.device != device:
                y = y.to(device)

            # (B, U) -> (B, U, H)
            y = self.prediction["embed"](y)
        else:
            # Y is not provided, assume zero tensor with shape [B, 1, H] is required
            # Emulates output of embedding of pad token.
            if batch_size is None:
                B = 1 if state is None else state[0].size(1)
            else:
                B = batch_size

            y = torch.zeros((B, 1, self.pred_hidden), device=device, dtype=dtype)

        # Prepend blank "start of sequence" symbol (zero tensor)
        if add_sos:
            B, U, H = y.shape
            start = torch.zeros((B, 1, H), device=y.device, dtype=y.dtype)
            y = torch.cat([start, y], dim=1).contiguous()  # (B, U + 1, H)
        else:
            start = None  # makes del call later easier

        # If in training mode, and random_state_sampling is set,
        # initialize state to random normal distribution tensor.
        if state is None:
            if self.random_state_sampling and self.training:
                state = self.initialize_state(y)

        # Forward step through RNN
        y = y.transpose(0, 1)  # (U + 1, B, H)
        g, hid = self.prediction["dec_rnn"](y, state)
        g = g.transpose(0, 1)  # (B, U + 1, H)

        del y, start, state
        return g, hid

    def _predict(self, vocab_size, pred_n_hidden, pred_rnn_layers, forget_gate_bias, t_max, norm, dropout):
        """
        Prepare the trainable parameters of the Prediction Network.

        Args:
            vocab_size: Vocab size (excluding the blank token).
            pred_n_hidden: Hidden size of the RNNs.
            pred_rnn_layers: Number of RNN layers.
            forget_gate_bias: Whether to perform unit forget gate bias.
            t_max: Whether to perform Chrono LSTM init.
            norm: Type of normalization to perform in RNN.
            dropout: Whether to apply dropout to RNN.
        """
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
        """
        Initialize the state of the RNN layers, with same dtype and device as input `y`.

        Args:
            y: A torch.Tensor whose device the generated states will be placed on.

        Returns:
            List of torch.Tensor, each of shape [L, B, H], where
                L = Number of RNN layers
                B = Batch size
                H = Hidden size of RNN.
        """
        batch = y.size(0)
        if self.random_state_sampling and self.training:
            state = [
                torch.randn(self.pred_rnn_layers, batch, self.pred_hidden, dtype=y.dtype, device=y.device),
                torch.randn(self.pred_rnn_layers, batch, self.pred_hidden, dtype=y.dtype, device=y.device),
            ]

        else:
            state = [
                torch.zeros(self.pred_rnn_layers, batch, self.pred_hidden, dtype=y.dtype, device=y.device),
                torch.zeros(self.pred_rnn_layers, batch, self.pred_hidden, dtype=y.dtype, device=y.device),
            ]
        return state

    def score_hypothesis(
        self, hypothesis: rnnt_utils.Hypothesis, cache: Dict[Tuple[int], Any]
    ) -> (torch.Tensor, List[torch.Tensor], torch.Tensor):
        """
        Similar to the predict() method, instead this method scores a Hypothesis during beam search.
        Hypothesis is a dataclass representing one hypothesis in a Beam Search.

        Args:
            hypothesis: Refer to rnnt_utils.Hypothesis.
            cache: Dict which contains a cache to avoid duplicate computations.

        Returns:
            Returns a tuple (y, states, lm_token) such that:
            y is a torch.Tensor of shape [1, 1, H] representing the score of the last token in the Hypothesis.
            state is a list of RNN states, each of shape [L, 1, H].
            lm_token is the final integer token of the hypothesis.
        """
        if hypothesis.dec_state is not None:
            device = hypothesis.dec_state[0].device
        else:
            _p = next(self.parameters())
            device = _p.device

        # parse "blank" tokens in hypothesis
        if len(hypothesis.y_sequence) > 0 and hypothesis.y_sequence[-1] == self.blank_idx:
            blank_state = True
        else:
            blank_state = False

        # Convert last token of hypothesis to torch.Tensor
        target = torch.full([1, 1], fill_value=hypothesis.y_sequence[-1], device=device, dtype=torch.long)
        lm_token = target[:, -1]  # [1]

        # Convert current hypothesis into a tuple to preserve in cache
        sequence = tuple(hypothesis.y_sequence)

        if sequence in cache:
            y, new_state = cache[sequence]
        else:
            # Obtain score for target token and new states
            if blank_state:
                y, new_state = self.predict(None, state=None, add_sos=False, batch_size=1)  # [1, 1, H]

            else:
                y, new_state = self.predict(
                    target, state=hypothesis.dec_state, add_sos=False, batch_size=1
                )  # [1, 1, H]

            y = y[:, -1:, :]  # Extract just last state : [1, 1, H]
            cache[sequence] = (y, new_state)

        return y, new_state, lm_token

    def batch_score_hypothesis(
        self, hypotheses: List[rnnt_utils.Hypothesis], cache: Dict[Tuple[int], Any], batch_states: List[torch.Tensor]
    ) -> (torch.Tensor, List[torch.Tensor], torch.Tensor):
        """
        Used for batched beam search algorithms. Similar to score_hypothesis method.

        Args:
            hypothesis: List of Hypotheses. Refer to rnnt_utils.Hypothesis.
            cache: Dict which contains a cache to avoid duplicate computations.
            batch_states: List of torch.Tensor which represent the states of the RNN for this batch.
                Each state is of shape [L, B, H]

        Returns:
            Returns a tuple (b_y, b_states, lm_tokens) such that:
            b_y is a torch.Tensor of shape [B, 1, H] representing the scores of the last tokens in the Hypotheses.
            b_state is a list of list of RNN states, each of shape [L, B, H].
                Represented as B x List[states].
            lm_token is a list of the final integer tokens of the hypotheses in the batch.
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

        # For each hypothesis, cache the last token of the sequence and the current states
        for i, hyp in enumerate(hypotheses):
            sequence = tuple(hyp.y_sequence)

            if sequence in cache:
                done[i] = cache[sequence]
            else:
                tokens.append(hyp.y_sequence[-1])
                process.append((sequence, hyp.dec_state))

        if process:
            batch = len(process)

            # convert list of tokens to torch.Tensor, then reshape.
            tokens = torch.tensor(tokens, device=device, dtype=torch.long).view(batch, -1)
            dec_states = self.initialize_state(tokens.to(dtype=dtype))  # [L, B, H]
            dec_states = self.batch_initialize_states(dec_states, [d_state for seq, d_state in process])

            y, dec_states = self.predict(
                tokens, state=dec_states, add_sos=False, batch_size=batch
            )  # [B, 1, H], List([L, 1, H])

        # Update done states and cache shared by entire batch.
        j = 0
        for i in range(final_batch):
            if done[i] is None:
                # Select sample's state from the batch state list
                new_state = self.batch_select_state(dec_states, j)

                # Cache [1, H] scores of the current y_j, and its corresponding state
                done[i] = (y[j], new_state)
                cache[process[j][0]] = (y[j], new_state)

                j += 1

        # Set the incoming batch states with the new states obtained from `done`.
        batch_states = self.batch_initialize_states(batch_states, [d_state for y_j, d_state in done])

        # Create batch of all output scores
        # List[1, 1, H] -> [B, 1, H]
        batch_y = torch.stack([y_j for y_j, d_state in done])

        # Extract the last tokens from all hypotheses and convert to a tensor
        lm_tokens = torch.tensor([h.y_sequence[-1] for h in hypotheses], device=device, dtype=torch.long).view(
            final_batch
        )

        return batch_y, batch_states, lm_tokens

    def batch_initialize_states(self, batch_states: List[torch.Tensor], decoder_states: List[List[torch.Tensor]]):
        """
        Create batch of decoder states.

       Args:
           batch_states (list): batch of decoder states
              ([L x (B, H)], [L x (B, H)])

           decoder_states (list of list): list of decoder states
               [B x ([L x (1, H)], [L x (1, H)])]

       Returns:
           batch_states (tuple): batch of decoder states
               ([L x (B, H)], [L x (B, H)])
       """
        # LSTM has 2 states
        for layer in range(self.pred_rnn_layers):
            for state_id in range(len(batch_states)):
                batch_states[state_id][layer] = torch.stack([s[state_id][layer] for s in decoder_states])

        return batch_states

    def batch_select_state(self, batch_states: List[torch.Tensor], idx: int) -> List[List[torch.Tensor]]:
        """Get decoder state from batch of states, for given id.

        Args:
            batch_states (list): batch of decoder states
                ([L x (B, H)], [L x (B, H)])

            idx (int): index to extract state from batch of states

        Returns:
            (tuple): decoder states for given id
                ([L x (1, H)], [L x (1, H)])
        """
        state_list = []
        for state_id in range(len(batch_states)):
            states = [batch_states[state_id][layer][idx] for layer in range(self.pred_rnn_layers)]
            state_list.append(states)

        return state_list


class RNNTJoint(rnnt_utils.AbstractRNNTJoint):
    """A Recurrent Neural Network Transducer Joint Network (RNN-T Joint Network).
    An RNN-T Joint network, comprised of a feedforward model.

    Args:
        jointnet: A dict-like object which contains the following key-value pairs.
            encoder_hidden: int specifying the hidden dimension of the encoder net.
            pred_hidden: int specifying the hidden dimension of the prediction net.
            joint_hidden: int specifying the hidden dimension of the joint net
            activation: Activation function used in the joint step. Can be one of
                ['relu', 'tanh', 'sigmoid'].

            Optionally, it may also contain the following:
            dropout: float, set to 0.0 by default. Optional dropout applied at the end of the joint net.

        num_classes: int, specifying the vocabulary size that the joint network must predict,
            excluding the RNNT blank token.

        vocabulary: Optional list of strings/tokens that comprise the vocabulary of the joint network.
            Unused and kept only for easy access for character based encoding RNNT models.

        log_softmax: Optional bool, set to None by default. If set as None, will compute the log_softmax()
            based on the value provided.

        preserve_memory: Optional bool, set to False by default. If the model crashes due to the memory
            intensive joint step, one might try this flag to empty the tensor cache in pytorch.

            Warning: This will make the forward-backward pass much slower than normal.
            It also might not fix the OOM if the GPU simply does not have enough memory to compute the joint.
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

        if preserve_memory:
            logging.warning(
                "`preserve_memory` was set for the Joint Model. Please be aware this will severely impact "
                "the forward-backward step time. It also might not solve OOM issues if the GPU simply "
                "does not have enough memory to compute the joint."
            )

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
        # decoder = (B, D, U)
        encoder_outputs = encoder_outputs.transpose(1, 2)  # (B, T, D)
        decoder_outputs = decoder_outputs.transpose(1, 2)  # (B, U, D)

        out = self.joint(encoder_outputs, decoder_outputs)  # [B, T, U, V + 1]
        return out

    def joint(self, f: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        """
        Compute the joint step of the network.

        Here,
        B = Batch size
        T = Acoustic model timesteps
        U = Target sequence length
        H1, H2 = Hidden dimensions of the Encoder / Decoder respectively
        H = Hidden dimension of the Joint hidden step.
        V = Vocabulary size of the Decoder (excluding the RNNT blank token).

        NOTE:
            The implementation of this model is slightly modified from the original paper.
            The original paper proposes the following steps :
            (enc, dec) -> Expand + Concat + Sum [B, T, U, H1+H2] -> Forward through joint hidden [B, T, U, H] -- *1
            *1 -> Forward through joint final [B, T, U, V + 1].

            We instead split the joint hidden into joint_hidden_enc and joint_hidden_dec and act as follows:
            enc -> Forward through joint_hidden_enc -> Expand [B, T, 1, H] -- *1
            dec -> Forward through joint_hidden_dec -> Expand [B, 1, U, H] -- *2
            (*1, *2) -> Sum [B, T, U, H] -> Forward through joint final [B, T, U, V + 1].

        Args:
            f: Output of the Encoder model. A torch.Tensor of shape [B, T, H1]
            g: Output of the Decoder model. A torch.Tensor of shape [B, U, H2]

        Returns:
            Logits / log softmaxed tensor of shape (B, T, U, V + 1).
        """
        # f = [B, T, H1]
        f = self.enc(f)
        f.unsqueeze_(dim=2)  # (B, T, 1, H)

        # g = [B, U, H2]
        g = self.pred(g)
        g.unsqueeze_(dim=1)  # (B, 1, U, H)

        inp = f + g  # [B, T, U, H]
        del f, g

        res = self.joint_net(inp)  # [B, T, U, V + 1]
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
        """
        Prepare the trainable modules of the Joint Network

        Args:
            num_classes: Number of output classes (vocab size) excluding the RNNT blank token.
            pred_n_hidden: Hidden size of the prediction network.
            enc_n_hidden: Hidden size of the encoder network.
            joint_n_hidden: Hidden size of the joint network.
            activation: Activation of the joint. Can be one of [relu, tanh, sigmoid]
            dropout: Dropout value to apply to joint.
        """
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
