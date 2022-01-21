###############################################################################
#
#  Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from nemo.collections.tts.helpers.common import get_mask_from_lengths, LinearNorm, ConvNorm, PositionalEmbedding, LengthRegulator, Encoder, DenseLayer


class GaussianMixture(torch.nn.Module):
    def __init__(self, n_hidden, n_components, n_mel_channels, fixed_gaussian,
                 mean_scale):
        super(GaussianMixture, self).__init__()
        self.n_mel_channels = n_mel_channels
        self.n_components = n_components
        self.fixed_gaussian = fixed_gaussian
        self.mean_scale = mean_scale

        # TODO: fuse into one dense n_components * 3
        self.prob_layer = LinearNorm(n_hidden, n_components)

        if not fixed_gaussian:
            self.mean_layer = LinearNorm(n_hidden, n_mel_channels * n_components)
            self.log_var_layer = LinearNorm(n_hidden, n_mel_channels * n_components)
        else:
            mean = self.generate_mean(n_mel_channels, n_components, mean_scale)
            log_var = self.generate_log_var(n_mel_channels, n_components)
            self.register_buffer('mean', mean.float())
            self.register_buffer('log_var', log_var.float())

    def generate_mean(self, n_dimensions, n_components, scale=3):
        means = torch.eye(n_dimensions).float()
        ids = np.random.choice(range(n_dimensions), n_components, replace=False)
        means = means[ids] * scale
        means = means.transpose(0, 1)
        means = means[None]
        return means

    def generate_log_var(self, n_dimensions, n_components):
        log_var = torch.zeros(1, n_dimensions, n_components).float()
        return log_var

    def generate_prob(self):
        return torch.ones(1, 1).float()

    def forward(self, outputs, bs):
        prob = torch.softmax(self.prob_layer(outputs), dim=1)

        if not self.fixed_gaussian:
            mean = self.mean_layer(outputs).view(
                bs, self.n_mel_channels, self.n_components)
            log_var = self.log_var_layer(outputs).view(
                bs, self.n_mel_channels, self.n_components)
        else:
            mean = self.mean
            log_var = self.log_var

        return mean, log_var, prob


class MelEncoder(nn.Module):
    """Encoder module:
        - Three 1-d convolution banks
        - Bidirectional LSTM
    """
    def __init__(self, encoder_embedding_dim=512, encoder_kernel_size=3,
                 encoder_n_convolutions=2, norm_fn=nn.InstanceNorm1d):
        super(MelEncoder, self).__init__()

        convolutions = []
        for i in range(encoder_n_convolutions):
            conv_layer = nn.Sequential(
                ConvNorm(80 if i == 0 else encoder_embedding_dim,
                         encoder_embedding_dim,
                         kernel_size=encoder_kernel_size, stride=1,
                         padding=int((encoder_kernel_size - 1) / 2),
                         dilation=1, w_init_gain='relu'),
                norm_fn(encoder_embedding_dim, affine=True))
            convolutions.append(conv_layer)
        self.convolutions = nn.ModuleList(convolutions)

        self.lstm = nn.LSTM(encoder_embedding_dim,
                            int(encoder_embedding_dim / 2), 1,
                            bidirectional=True)

    def run_padded_sequence(self, sorted_idx, unsort_idx, lens, padded_data,
                            recurrent_model):
        """Sorts input data by previded ordering (and un-ordering) and runs the
        packed data through the recurrent model

        Args:
            sorted_idx (torch.tensor): 1D sorting ids
            unsort_idx (torch.tensor): 1D unsorting ids (sorted_idx inverse)
            lens: lengths of input data (sorted in descending order)
            padded_data (torch.tensor): input sequences (padded)
            recurrent_model (nn.Module): recurrent model(data)
        Returns:
            hidden_vectors (torch.tensor): RNN outputs, original ordering
        """

        # sort the data by decreasing length using provided index
        # we assume batch index is in dim=1
        padded_data = padded_data[:, sorted_idx]
        padded_data = nn.utils.rnn.pack_padded_sequence(padded_data, lens)
        hidden_vectors = recurrent_model(padded_data)[0]
        hidden_vectors, _ = nn.utils.rnn.pad_packed_sequence(hidden_vectors)
        # unsort the results at dim=1 and return
        hidden_vectors = hidden_vectors[:, unsort_idx]
        return hidden_vectors

    def forward(self, x, lens):
        if x.size()[0] > 1:
            x_embedded = []
            for b_ind in range(x.size()[0]):  # TODO: Speed with correctness
                curr_x = x[b_ind:b_ind+1, :, :lens[b_ind]].clone()
                for conv in self.convolutions:
                    curr_x = F.dropout(
                        F.relu(conv(curr_x)), 0.5, self.training)
                x_embedded.append(curr_x[0].transpose(0, 1))
            x = torch.nn.utils.rnn.pad_sequence(x_embedded, batch_first=True)
        else:
            for conv in self.convolutions:
                x = F.dropout(F.relu(conv(x)), 0.5, self.training)
            x = x.transpose(1, 2)

        x = x.transpose(0, 1)

        self.lstm.flatten_parameters()
        if lens is not None:
            # collect decreasing length indices
            lens, ids = torch.sort(lens, descending=True)
            original_ids = [0] * lens.size(0)
            for i in range(len(ids)):
                original_ids[ids[i]] = i
            x = self.run_padded_sequence(ids, original_ids, lens, x, self.lstm)
        else:
            x, _ = self.lstm(x)

        # average pooling over time dimension
        x = torch.mean(x, dim=0)
        return x

    def infer(self, x):
        for conv in self.convolutions:
            x = F.dropout(F.relu(conv(x)), 0.5, self.training)

        x = x.transpose(1, 2)

        self.lstm.flatten_parameters()
        outputs, _ = self.lstm(x)

        return outputs


class AR_Back_Step(torch.nn.Module):
    def __init__(self, n_mel_channels, n_speaker_dim, n_text_dim,
                 n_hidden, n_lstm_layers, scaling_fn, p_dropout,
                 n_out_dims=None, with_conv_in="", conv_in_nonlinearity="",
                 kernel_size=1, n_convs=1):
        super(AR_Back_Step, self).__init__()
        self.ar_step = AR_Step(n_mel_channels, n_speaker_dim, n_text_dim,
                               n_hidden, n_lstm_layers, scaling_fn, p_dropout,
                               n_out_dims, with_conv_in, conv_in_nonlinearity,
                               kernel_size)

    def forward(self, mel, context, mask, in_lens, out_lens, affine_dims=None):
        mel = torch.flip(mel, (0, ))
        context = torch.flip(context, (0, ))
        # backwards flow, send padded zeros back to end
        for k in range(mel.size(1)):
            mel[:, k] = mel[:, k].roll(out_lens[k].item(), dims=0)
            context[:, k] = context[:, k].roll(in_lens[k].item(), dims=0)

        mel, log_s = self.ar_step(mel, context, mask, in_lens, out_lens,
                                  affine_dims)

        # move padded zeros back to beginning
        for k in range(mel.size(1)):
            mel[:, k] = mel[:, k].roll(-out_lens[k].item(), dims=0)

        return torch.flip(mel, (0, )), log_s

    def infer(self, residual, context, affine_dims):
        residual = self.ar_step.infer(
            torch.flip(residual, (0, )), torch.flip(context, (0, )),
            affine_dims=affine_dims)
        residual = torch.flip(residual, (0, ))
        return residual


class AR_Step(torch.nn.Module):
    def __init__(self, n_mel_channels, n_speaker_dim, n_text_channels,
                 n_hidden, n_lstm_layers, scaling_fn, p_dropout=0.0,
                 n_out_dims=None, with_conv_in="", conv_in_nonlinearity="",
                 kernel_size=1, n_convs=1):
        super(AR_Step, self).__init__()
        if n_out_dims is not None:
            self.n_out_dims = n_out_dims
            self.conv = torch.nn.Conv1d(n_hidden, 2*n_out_dims, 1)
        elif scaling_fn == 'translate':
            self.n_out_dims = n_mel_channels
            self.conv = torch.nn.Conv1d(n_hidden, n_mel_channels, 1)
        else:
            self.n_out_dims = n_mel_channels
            self.conv = torch.nn.Conv1d(n_hidden, 2*n_mel_channels, 1)

        # for scaling each dimension independently before lstm
        self.with_conv_in = with_conv_in
        self.conv_in_nonlinearity = conv_in_nonlinearity
        self.n_convs = n_convs
        if with_conv_in == 'conv':
            if n_convs > 1:
                convs = []
                for i in range(n_convs):
                    dilation = 2 ** i
                    conv = ConvNorm(
                        n_mel_channels, n_mel_channels, dilation=dilation,
                        kernel_size=kernel_size, w_init_gain='tanh',
                        is_causal=True)
                    convs.append(conv)
                    convs.append(torch.nn.Tanh())
                self.conv_in = torch.nn.Sequential(*convs)
            else:
                self.conv_in = ConvNorm(
                    n_mel_channels, n_mel_channels, kernel_size=kernel_size,
                    is_causal=True)
        elif with_conv_in == 'linear':
            self.conv_in = LinearNorm(
                n_mel_channels, n_mel_channels)

        self.conv.weight.data = 0.0 * self.conv.weight.data
        self.conv.bias.data = 0.0 * self.conv.bias.data
        self.mel_lstm = torch.nn.LSTM(n_mel_channels, n_hidden)
        self.lstm = torch.nn.LSTM(n_hidden+n_text_channels+n_speaker_dim,
                                  n_hidden, n_lstm_layers)
        self.dense_layer = DenseLayer(in_dim=n_hidden,
                                      sizes=[n_hidden, n_hidden])
        self.scaling_fn = scaling_fn
        self.dropout = nn.Dropout(p=p_dropout)

    def run_padded_sequence(self, sorted_idx, unsort_idx, lens, padded_data,
                            recurrent_model):
        """Sorts input data by previded ordering (and un-ordering) and runs the
        packed data through the recurrent model

        Args:
            sorted_idx (torch.tensor): 1D sorting index
            unsort_idx (torch.tensor): 1D unsorting index (inverse of sorted_idx)
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
        padded_data = nn.utils.rnn.pack_padded_sequence(
            padded_data, lens.cpu())
        hidden_vectors = recurrent_model(padded_data)[0]
        hidden_vectors, _ = nn.utils.rnn.pad_packed_sequence(hidden_vectors)
        # unsort the results at dim=1 and return
        hidden_vectors = hidden_vectors[:, unsort_idx]
        return hidden_vectors

    def get_scaling_and_logs(self, scale_unconstrained):
        if self.scaling_fn == 'translate':
            s = torch.exp(scale_unconstrained*0)
            log_s = scale_unconstrained*0
        elif self.scaling_fn == 'exp':
            s = torch.exp(scale_unconstrained)
            log_s = scale_unconstrained  # log(exp
        elif self.scaling_fn == 'tanh':
            s = torch.tanh(scale_unconstrained) + 1 + 1e-6
            log_s = torch.log(s)
        elif self.scaling_fn == 'sigmoid':
            s = torch.sigmoid(scale_unconstrained + 10) + 1e-6
            log_s = torch.log(s)
        elif isinstance(self.scaling_fn, list):
            s_list, log_s_list = [], []
            for i in range(scale_unconstrained.shape[2]):
                scaling_i = self.scaling_fn[i]
                if scaling_i == 'translate':
                    s_i = torch.exp(scale_unconstrained[..., i]*0)
                    log_s_i = scale_unconstrained[..., i]*0
                elif scaling_i == 'exp':
                    s_i = torch.exp(scale_unconstrained[..., i])
                    log_s_i = scale_unconstrained[..., i]
                elif scaling_i == 'tanh':
                    s_i = torch.tanh(scale_unconstrained[..., i]) + 1 + 1e-6
                    log_s_i = torch.log(s_i)
                elif scaling_i == 'sigmoid':
                    s_i = torch.sigmoid(scale_unconstrained[..., i]) + 1e-6
                    log_s_i = torch.log(s_i)
                s_list.append(s_i[..., None])
                log_s_list.append(log_s_i[..., None])
            s = torch.cat(s_list, dim=2)
            log_s = torch.cat(log_s_list, dim=2)
        else:
            raise Exception("Scaling fn {} not supp.".format(self.scaling_fn))

        return s, log_s

    def forward(self, mel, context, mask, in_lens, out_lens, affine_dims=None):
        dummy = torch.FloatTensor(1, mel.size(1), mel.size(2)).zero_()
        dummy = dummy.type(mel.type())
        # seq_len x batch x dim
        mel0 = torch.cat([dummy, mel[:-1]], 0)

        if self.with_conv_in == 'conv':
            mel0 = self.conv_in(mel0.permute(1, 2, 0)).permute(2, 0, 1)
        elif self.with_conv_in == 'linear':
            mel0 = self.conv_in(mel0)

        if self.conv_in_nonlinearity == 'tanh':
            mel0 = F.tanh(mel0)
        elif self.conv_in_nonlinearity == 'relu':
            mel0 = F.relu(mel0)

        self.lstm.flatten_parameters()
        self.mel_lstm.flatten_parameters()
        if out_lens is not None:
            # collect decreasing length indices
            lens, ids = torch.sort(out_lens, descending=True)
            original_ids = [0] * lens.size(0)
            for i in range(len(ids)):
                original_ids[ids[i]] = i
            # mel_seq_len x batch x hidden_dim
            mel_hidden = self.run_padded_sequence(
                ids, original_ids, lens, mel0, self.mel_lstm)
        else:
            mel_hidden = self.mel_lstm(mel0)[0]

        decoder_input = torch.cat((mel_hidden, context), -1)

        if out_lens is not None:
            # reorder, run padded sequence and undo reordering
            lstm_hidden = self.run_padded_sequence(
                ids, original_ids, lens, decoder_input, self.lstm)
        else:
            lstm_hidden = self.lstm(decoder_input)[0]

        lstm_hidden = self.dropout(lstm_hidden)
        lstm_hidden = self.dense_layer(lstm_hidden).permute(1, 2, 0)
        decoder_output = self.conv(lstm_hidden).permute(2, 0, 1)

        if self.scaling_fn == 'translate':
            scale = 1
            log_s = decoder_output * 0
            bias = decoder_output
        else:
            scale, log_s = self.get_scaling_and_logs(
                decoder_output[:, :, :self.n_out_dims])
            bias = decoder_output[:, :, self.n_out_dims:]

        if affine_dims is None:
            mel = scale * mel + bias
        else:
            mel[..., affine_dims] = scale * mel[..., affine_dims] + bias

        return mel, log_s

    def infer(self, residual, context, affine_dims=None):
        total_output = []  # seems 10FPS faster than pre-allocation

        output = None
        dummy = torch.cuda.FloatTensor(1, residual.size(1), residual.size(2)).zero_()
        self.mel_lstm.flatten_parameters()
        for i in range(0, residual.size(0)):
            if i == 0:
                if self.with_conv_in == 'conv':
                    output = self.conv_in(dummy.permute(1, 2, 0)).permute(2, 0, 1)
                elif self.with_conv_in == 'linear':
                    output = self.conv_in(dummy)

                if self.conv_in_nonlinearity == 'tanh':
                    output = F.tanh(output)
                elif self.conv_in_nonlinearity == 'relu':
                    output = F.relu(output)
                else:
                    output = dummy

                mel_hidden, (h, c) = self.mel_lstm(output)
            else:
                if self.with_conv_in == 'conv':
                    output = self.conv_in(output.permute(1, 2, 0)).permute(2, 0, 1)
                elif self.with_conv_in == 'linear':
                    output = self.conv_in(output)

                if self.conv_in_nonlinearity == 'tanh':
                    output = F.tanh(output)
                elif self.conv_in_nonlinearity == 'relu':
                    output = F.relu(output)

                mel_hidden, (h, c) = self.mel_lstm(output, (h, c))

            decoder_input = torch.cat((mel_hidden, context[i][None]), -1)
            if i == 0:
                lstm_hidden, (h1, c1) = self.lstm(decoder_input)
            else:
                lstm_hidden, (h1, c1) = self.lstm(decoder_input, (h1, c1))
            lstm_hidden = self.dense_layer(lstm_hidden).permute(1, 2, 0)
            decoder_output = self.conv(lstm_hidden).permute(2, 0, 1)

            s, log_s = self.get_scaling_and_logs(
                decoder_output[:, :, :decoder_output.size(2)//2])
            b = decoder_output[:, :, decoder_output.size(2)//2:]
            if affine_dims is not None:
                output = residual[i:i+1]
                output[..., affine_dims] = (output[..., affine_dims] - b)/s
            else:
                output = (residual[i:i+1] - b)/s
            total_output.append(output)

        total_output = torch.cat(total_output, 0)
        return total_output


class Flowtron(torch.nn.Module):
    def __init__(self, n_speakers, n_speaker_dim, n_text, n_text_dim, n_flows,
                 n_mel_channels, n_hidden, n_lstm_layers,
                 mel_encoder_n_hidden, n_components,
                 fixed_gaussian, mean_scale, dummy_speaker_embedding,
                 use_positional_embedding, with_conv_in, conv_in_nonlinearity):

        super(Flowtron, self).__init__()
        norm_fn = nn.InstanceNorm1d
        self.speaker_embedding = torch.nn.Embedding(n_speakers, n_speaker_dim)
        self.embedding = torch.nn.Embedding(n_text, n_text_dim)
        self.flows = torch.nn.ModuleList()
        self.encoder = Encoder(norm_fn=norm_fn)
        self.dummy_speaker_embedding = dummy_speaker_embedding
        self.length_regulator = LengthRegulator()
        if use_positional_embedding:
            self.pos_emb_text = PositionalEmbedding(n_text_dim, max_len=500)
            self.pos_emb_context = PositionalEmbedding(n_text_dim, max_len=5000)

        if n_components > 1:
            self.mel_encoder = MelEncoder(mel_encoder_n_hidden, norm_fn=norm_fn)
            self.gaussian_mixture = GaussianMixture(mel_encoder_n_hidden,
                                                    n_components,
                                                    n_mel_channels,
                                                    fixed_gaussian, mean_scale)

        for i in range(n_flows):
            if i % 2 == 0:
                self.flows.append(AR_Step(n_mel_channels, n_speaker_dim,
                                          n_text_dim,
                                          n_hidden, n_lstm_layers,
                                          with_conv_in))
            else:
                self.flows.append(AR_Back_Step(n_mel_channels, n_speaker_dim,
                                               n_text_dim,
                                               n_hidden, n_lstm_layers,
                                               with_conv_in))

    def forward(self, mel, speaker_ids, text, token_durations, in_lens,
                out_lens):
        speaker_ids = speaker_ids*0 if self.dummy_speaker_embedding else speaker_ids
        speaker_vecs = self.speaker_embedding(speaker_ids)
        text = self.embedding(text).transpose(1, 2)
        if hasattr(self, 'pos_emb_text'):
            text = self.pos_emb_text(text)
        text = self.encoder(text, in_lens).transpose(1, 2)

        dur_pred = self.dur_pred_layer(text, in_lens)

        context = self.length_regulator(
            text.transpose(1, 2), token_durations).transpose(1, 2)
        if hasattr(self, 'pos_emb_context'):
            context = self.pos_emb_context(context)
        mean, log_var, prob = None, None, None
        if hasattr(self, 'gaussian_mixture'):
            mel_embedding = self.mel_encoder(mel)
            mean, log_var, prob = self.gaussian_mixture(
                mel_embedding, mel_embedding.size(0))

        mel = mel.permute(2, 0, 1)
        context = context.permute(2, 0, 1)

        encoder_outputs = torch.cat(
            [context, speaker_vecs.expand(context.size(0), -1, -1)], 2)
        log_s_list = []
        mask = ~get_mask_from_lengths(in_lens)[..., None]
        for i, flow in enumerate(self.flows):
            mel, log_s = flow(mel, encoder_outputs, mask, in_lens, out_lens)
            log_s_list.append(log_s)
        return mel, log_s_list, dur_pred, mean, log_var, prob

    def infer(self, residual, speaker_ids, text, token_durations=None):
        """Inference function. Inverse of the forward pass

        Args:
            residual: 1 x 80 x N_residual tensor of sampled z values
            speaker_ids: 1 x 1 tensor of int speaker ids (single value)
            text (torch.int64): 1 x N_text tensor holding text-token ids

        Returns:
            residual: input residual after flow. mel spectrogram values
        """

        speaker_ids = speaker_ids*0 if self.dummy_speaker_embedding else speaker_ids
        speaker_vecs = self.speaker_embedding(speaker_ids)
        text = self.embedding(text).transpose(1, 2)
        if hasattr(self, 'pos_emb_text'):
            text = text + self.pos_emb_text(text)
        text = self.encoder.infer(text).transpose(1, 2)

        if token_durations is None:
            token_durations = self.dur_pred_layer(text)

        context = self.length_regulator(
            text.transpose(1, 2), token_durations).transpose(1, 2)
        if hasattr(self, 'pos_emb_context'):
            context = context + self.pos_emb_context(context)

        residual = residual.permute(2, 0, 1)
        context = context.permute(2, 0, 1)

        encoder_outputs = torch.cat(
            [context, speaker_vecs.expand(context.size(0), -1, -1)], 2)

        for i, flow in enumerate(reversed(self.flows)):
            residual  = flow.infer(residual, encoder_outputs)
        return residual.permute(1, 2, 0)

    def test_invertibility(self, residual, speaker_ids, text, token_durations):
        """Model invertibility check. Call this the same way you would call self.infer()

        Args:
            residual: 1 x 80 x N_residual tensor of sampled z values
            speaker_ids: 1 x 1 tensor of integral speaker ids (should be a single value)
            text (torch.int64): 1 x N_text tensor holding text-token ids

        Returns:
            error: should be in the order of 1e-5 or less, or there may be an invertibility bug
        """
        mel = self.infer(residual, speaker_ids, text, token_durations)
        in_lens = torch.LongTensor([text.shape[1]]).cuda()
        residual_recon, log_s_list, _, _, _, _ = self.forward(
            mel, speaker_ids, text, token_durations, in_lens, None)
        residual_permuted = residual.permute(2, 0, 1)
        if len(self.flows) % 2 == 0:
            residual_permuted = torch.flip(residual_permuted, (0,))
            residual_recon = torch.flip(residual_recon, (0,))
        error = (residual_recon - residual_permuted[0:residual_recon.shape[0]]).abs().mean()
        return error
