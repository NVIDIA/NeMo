###############################################################################
#
#  Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
###############################################################################
import torch
from torch import nn

from nemo.collections.tts.helpers.common import DenseLayer, SplineTransformationLayerAR


class AR_Back_Step(torch.nn.Module):
    def __init__(
        self, n_attr_channels, n_speaker_dim, n_text_dim, n_hidden, n_lstm_layers, scaling_fn, spline_flow_params=None
    ):
        super(AR_Back_Step, self).__init__()
        self.ar_step = AR_Step(
            n_attr_channels, n_speaker_dim, n_text_dim, n_hidden, n_lstm_layers, scaling_fn, spline_flow_params
        )

    def forward(self, mel, context, lens):
        mel = torch.flip(mel, (0,))
        context = torch.flip(context, (0,))
        # backwards flow, send padded zeros back to end
        for k in range(mel.size(1)):
            mel[:, k] = mel[:, k].roll(lens[k].item(), dims=0)
            context[:, k] = context[:, k].roll(lens[k].item(), dims=0)

        mel, log_s = self.ar_step(mel, context, lens)

        # move padded zeros back to beginning
        for k in range(mel.size(1)):
            mel[:, k] = mel[:, k].roll(-lens[k].item(), dims=0)

        return torch.flip(mel, (0,)), log_s

    def infer(self, residual, context):
        residual = self.ar_step.infer(torch.flip(residual, (0,)), torch.flip(context, (0,)))
        residual = torch.flip(residual, (0,))
        return residual


class AR_Step(torch.nn.Module):
    def __init__(
        self,
        n_attr_channels,
        n_speaker_dim,
        n_text_channels,
        n_hidden,
        n_lstm_layers,
        scaling_fn,
        spline_flow_params=None,
    ):
        super(AR_Step, self).__init__()
        if spline_flow_params is not None:
            self.spline_flow = SplineTransformationLayerAR(**spline_flow_params)
        else:
            self.n_out_dims = n_attr_channels
            self.conv = torch.nn.Conv1d(n_hidden, 2 * n_attr_channels, 1)
            self.conv.weight.data = 0.0 * self.conv.weight.data
            self.conv.bias.data = 0.0 * self.conv.bias.data

        self.attr_lstm = torch.nn.LSTM(n_attr_channels, n_hidden)
        self.lstm = torch.nn.LSTM(n_hidden + n_text_channels + n_speaker_dim, n_hidden, n_lstm_layers)

        if spline_flow_params is None:
            self.dense_layer = DenseLayer(in_dim=n_hidden, sizes=[n_hidden, n_hidden])
            self.scaling_fn = scaling_fn

    def run_padded_sequence(self, sorted_idx, unsort_idx, lens, padded_data, recurrent_model):
        """Sorts input data by previded ordering (and un-ordering) and runs the
        packed data through the recurrent model

        Args:
            sorted_idx (torch.tensor): 1D sorting index
            unsort_idx (torch.tensor): 1D unsorting index (inverse sorted_idx)
            lens: lengths of input data (sorted in descending order)
            padded_data (torch.tensor): input sequences (padded)
            recurrent_model (nn.Module): recurrent model to run data through
        Returns:
            hidden_vectors (torch.tensor): outputs of the RNN, in the original,
            unsorted, ordering
        """

        # sort the data by decreasing length using provided index
        # we assume batch index is in dim=1
        padded_data = padded_data[:, sorted_idx]
        padded_data = nn.utils.rnn.pack_padded_sequence(padded_data, lens.cpu())
        hidden_vectors = recurrent_model(padded_data)[0]
        hidden_vectors, _ = nn.utils.rnn.pad_packed_sequence(hidden_vectors)
        # unsort the results at dim=1 and return
        hidden_vectors = hidden_vectors[:, unsort_idx]
        return hidden_vectors

    def get_scaling_and_logs(self, scale_unconstrained):
        if self.scaling_fn == 'translate':
            s = torch.exp(scale_unconstrained * 0)
            log_s = scale_unconstrained * 0
        elif self.scaling_fn == 'exp':
            s = torch.exp(scale_unconstrained)
            log_s = scale_unconstrained  # log(exp
        elif self.scaling_fn == 'tanh':
            s = torch.tanh(scale_unconstrained) + 1 + 1e-6
            log_s = torch.log(s)
        elif self.scaling_fn == 'sigmoid':
            s = torch.sigmoid(scale_unconstrained + 10) + 1e-6
            log_s = torch.log(s)
        else:
            raise Exception("Scaling fn {} not supp.".format(self.scaling_fn))

        return s, log_s

    def forward(self, mel, context, lens):
        dummy = torch.FloatTensor(1, mel.size(1), mel.size(2)).zero_()
        dummy = dummy.type(mel.type())
        # seq_len x batch x dim
        mel0 = torch.cat([dummy, mel[:-1]], 0)

        self.lstm.flatten_parameters()
        self.attr_lstm.flatten_parameters()
        if lens is not None:
            # collect decreasing length indices
            lens, ids = torch.sort(lens, descending=True)
            original_ids = [0] * lens.size(0)
            for i, ids_i in enumerate(ids):
                original_ids[ids_i] = i
            # mel_seq_len x batch x hidden_dim
            mel_hidden = self.run_padded_sequence(ids, original_ids, lens, mel0, self.attr_lstm)
        else:
            mel_hidden = self.attr_lstm(mel0)[0]

        decoder_input = torch.cat((mel_hidden, context), -1)

        if lens is not None:
            # reorder, run padded sequence and undo reordering
            lstm_hidden = self.run_padded_sequence(ids, original_ids, lens, decoder_input, self.lstm)
        else:
            lstm_hidden = self.lstm(decoder_input)[0]

        if hasattr(self, 'spline_flow'):
            # spline flow fn expects inputs to be batch, channel, time
            lstm_hidden = lstm_hidden.permute(1, 2, 0)
            mel = mel.permute(1, 2, 0)
            mel, log_s = self.spline_flow(mel, lstm_hidden, inverse=False)
            mel = mel.permute(2, 0, 1)
            log_s = log_s.permute(2, 0, 1)
        else:
            lstm_hidden = self.dense_layer(lstm_hidden).permute(1, 2, 0)
            decoder_output = self.conv(lstm_hidden).permute(2, 0, 1)

            scale, log_s = self.get_scaling_and_logs(decoder_output[:, :, : self.n_out_dims])
            bias = decoder_output[:, :, self.n_out_dims :]

            mel = scale * mel + bias

        return mel, log_s

    def infer(self, residual, context):
        total_output = []  # seems 10FPS faster than pre-allocation

        output = None
        dummy = torch.cuda.FloatTensor(1, residual.size(1), residual.size(2)).zero_()
        self.attr_lstm.flatten_parameters()

        for i in range(0, residual.size(0)):
            if i == 0:
                output = dummy
                mel_hidden, (h, c) = self.attr_lstm(output)
            else:
                mel_hidden, (h, c) = self.attr_lstm(output, (h, c))

            decoder_input = torch.cat((mel_hidden, context[i][None]), -1)

            if i == 0:
                lstm_hidden, (h1, c1) = self.lstm(decoder_input)
            else:
                lstm_hidden, (h1, c1) = self.lstm(decoder_input, (h1, c1))

            if hasattr(self, 'spline_flow'):
                # expects inputs to be batch, channel, time
                lstm_hidden = lstm_hidden.permute(1, 2, 0)
                output = residual[i : i + 1].permute(1, 2, 0)
                output = self.spline_flow(output, lstm_hidden, inverse=True)
                output = output.permute(2, 0, 1)
            else:
                lstm_hidden = self.dense_layer(lstm_hidden).permute(1, 2, 0)
                decoder_output = self.conv(lstm_hidden).permute(2, 0, 1)

                s, log_s = self.get_scaling_and_logs(decoder_output[:, :, : decoder_output.size(2) // 2])
                b = decoder_output[:, :, decoder_output.size(2) // 2 :]
                output = (residual[i : i + 1] - b) / s
            total_output.append(output)

        total_output = torch.cat(total_output, 0)
        return total_output
