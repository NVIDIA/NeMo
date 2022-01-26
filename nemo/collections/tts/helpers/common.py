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
import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.cuda import amp
import numpy as np
import ast
import numbers
from nemo.utils import logging


def shift_and_scale_data(data, mean=0.0, std=0.0, dtype=None):
    dtype = data.dtype
    if mean > 0.0 or std > 0.0:
        data_mean, data_std = data.mean(), data.std()
        if mean > 0.0 and std > 0.0:
            data = (data - data_mean) / data_std
            data = (data * std) + mean
        elif mean > 0.0:
            data = data - data_mean + mean
        elif std > 0.0:
            data = (data - data_mean) / data_std
            data = (data * std) + data_mean
    return data


class GaussianSmoothing(nn.Module):
    """
    Apply gaussian smoothing on a 1d tensor.
    kernel_size (int): Size of the gaussian kernel.
    sigma (float): Standard deviation of the gaussian kernel.
    """
    def __init__(self, kernel_size=7, sigma=1):
        super(GaussianSmoothing, self).__init__()
        self.update_kernel(kernel_size, sigma)
        self.conv = F.conv1d

    def forward(self, data, kernel_size=7, sigma=2):
        """
        Apply gaussian filter to input.
        Arguments:
            input (torch.Tensor): Input to apply gaussian filter on.
        Returns:
            filtered (torch.Tensor): Filtered output.
        """
        self.update_kernel(kernel_size, sigma)
        x = data.clone()
        if len(data.shape) == 2:
            x = x[:, None]

        padding = (kernel_size - 1) // 2
        x = F.pad(x, (padding, padding), mode='reflect')
        x = self.conv(x, weight=self.weight, groups=1)
        if len(data.shape) == 2:
            x = x[:, 0]

        return x

    def update_kernel(self, kernel_size, sigma):
        if isinstance(kernel_size, numbers.Number):
            kernel_size = [kernel_size]
        if isinstance(sigma, numbers.Number):
            sigma = [sigma]

        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        kernel = 1
        meshgrids = torch.meshgrid(
            [
                torch.arange(size, dtype=torch.float32)
                for size in kernel_size
            ]
        )
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= 1 / (std * np.sqrt(2 * np.pi)) * \
                      torch.exp(-((mgrid - mean) / std) ** 2 / 2)

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)

        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(1, *[1] * (kernel.dim() - 1))

        if hasattr(self, 'weight'):
            kernel = kernel.to(self.weight.device)
            self.weight = kernel
        else:
            self.register_buffer('weight', kernel)


def masked_instance_norm(x, mask, eps=1e-8):
    """
    from https://discuss.pytorch.org/t/masked-instance-norm/83502
    x of shape: [batch_size (N), num_objects (L), features(C)]
    mask of shape: [batch_size (N), num_objects (L)]
    """
    mask = mask.float().unsqueeze(-1)  # (N,L,1)
    mean = (torch.sum(x * mask, 1) / torch.sum(mask, 1))   # (N,C)
    mean = mean.detach()
    var_term = ((x - mean.unsqueeze(1).expand_as(x)) * mask)**2  # (N,L,C)
    var = (torch.sum(var_term, 1) / torch.sum(mask, 1))  # (N,C)
    var = var.detach()
    mean_reshaped = mean.unsqueeze(1).expand_as(x)  # (N, L, C)
    var_reshaped = var.unsqueeze(1).expand_as(x)    # (N, L, C)
    ins_norm = (x - mean_reshaped) / torch.sqrt(var_reshaped + eps)   # (N, L, C)
    return ins_norm


def update_params(config, params):
    for param in params:
        logging.info(param)
        k, v = param.split("=")
        try:
            v = ast.literal_eval(v)
        except:
            pass

        k_split = k.split('.')
        if len(k_split) > 1:
            parent_k = k_split[0]
            cur_param = ['.'.join(k_split[1:])+"="+str(v)]
            update_params(config[parent_k], cur_param)
        elif k in config and len(k_split) == 1:
            logging.info(f"overridding {k} with {v}")
            config[k] = v
        else:
            logging.info("{}, {} params not updated".format(k, v))


def truncated_normal(shape, mean=0, std=1, trunc_std=1.1):
    output = torch.zeros(shape).normal_()
    valid = (output < trunc_std) & (output > -trunc_std)
    num_invalid = (~valid).sum()

    while num_invalid > 0:
        output[~valid] = output[~valid].normal_().mul_(std).add_(mean)
        valid = (output < trunc_std) & (output > -trunc_std)
        num_invalid = (~valid).sum()

    return output.mul_(std).add_(mean)



def compute_cosine_autocorr(text_encoded, length_mask):
    """Computes the cosine loss of all pairs and returns the autocorr distance map

    Args:
        text_encoded (torch.tensor): 1D tensor representing text embeds generally
    Returns:
        cost (float): scalar
        map (tensor): map representing autocorr [B, seq_len, seq_len]
    """
    # normalization
    text_enc = text_encoded.transpose(1, 2)
    text_enc_norm = torch.linalg.norm(text_enc, dim=2, keepdim=True)

    norm_text_enc = text_enc.div(text_enc_norm.expand_as(text_enc))

    # mask out the vectors that are outside the seq length to remove NANs from the div
    norm_text_enc.data.masked_fill_(length_mask.bool(), 0.0)

    cosine_sim = torch.bmm(norm_text_enc, norm_text_enc.transpose(1, 2)) # [B, seq_len, seq_len]

    # normalize the similarity to be between 0, 1
    cosine_sim = (cosine_sim + 1.0) / 2.0

    cosine_dist = 1.0 - cosine_sim # diagnol elements will become 0.0

    # forming the cosine distance mask using length_mask by defining as [B, seq_len, seq_len mask]
    # mask out the distances between masked and unmasked ones - we dont wanna max these.
    repeated_length_mask = length_mask.float().repeat(1, 1, length_mask.shape[1])
    cosine_dist_mask = (repeated_length_mask.transpose(1, 2) + repeated_length_mask).bool()
    cosine_dist.data.masked_fill_(cosine_dist_mask.bool(), 0.0)

    # the optimizer will minimze the cost...and we have to maximize the distance between text reps
    cost = -1.0*cosine_dist.mean()

    # avg the cost
    cost = cost / text_encoded.shape[0]

    return cost, cosine_dist, norm_text_enc.transpose(1, 2)


def get_mask_from_lengths(lengths):
    """Constructs binary mask from a 1D torch tensor of input lengths

    Args:
        lengths (torch.tensor): 1D tensor
    Returns:
        mask (torch.tensor): num_sequences x max_length x 1 binary tensor
    """
    max_len = torch.max(lengths).item()
    ids = torch.arange(0, max_len, out=torch.cuda.LongTensor(max_len))
    mask = (ids < lengths.unsqueeze(1)).bool()
    return mask


@torch.jit.script
def fused_add_tanh_sigmoid_multiply(input_a, input_b):
    t_act = torch.tanh(input_a)
    s_act = torch.sigmoid(input_b)
    acts = t_act * s_act
    return acts


class LinearNorm(torch.nn.Module):
    def __init__(self, in_dim, out_dim, bias=True, w_init_gain='linear'):
        super(LinearNorm, self).__init__()
        self.linear_layer = torch.nn.Linear(in_dim, out_dim, bias=bias)

        torch.nn.init.xavier_uniform_(
            self.linear_layer.weight,
            gain=torch.nn.init.calculate_gain(w_init_gain))

    def forward(self, x):
        return self.linear_layer(x)


class ConvNorm(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
                 padding=None, dilation=1, bias=True, w_init_gain='linear',
                 is_causal=False):
        super(ConvNorm, self).__init__()
        if padding is None:
            assert(kernel_size % 2 == 1)
            padding = int(dilation * (kernel_size - 1) / 2)
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.is_causal = is_causal
        self.conv = torch.nn.Conv1d(in_channels, out_channels,
                                    kernel_size=kernel_size, stride=stride,
                                    padding=padding, dilation=dilation,
                                    bias=bias)

        torch.nn.init.xavier_uniform_(
            self.conv.weight, gain=torch.nn.init.calculate_gain(w_init_gain))

    def forward(self, signal):
        if self.is_causal:
            padding = (int((self.kernel_size - 1) * (self.dilation)), 0)
            signal = torch.nn.functional.pad(signal, padding)
        conv_signal = self.conv(signal)
        return conv_signal


class DenseLayer(nn.Module):
    def __init__(self, in_dim=1024, sizes=[1024, 1024]):
        super(DenseLayer, self).__init__()
        in_sizes = [in_dim] + sizes[:-1]
        self.layers = nn.ModuleList(
            [LinearNorm(in_size, out_size, bias=True)
             for (in_size, out_size) in zip(in_sizes, sizes)])

    def forward(self, x):
        for linear in self.layers:
            x = torch.tanh(linear(x))
        return x


class PositionalEmbedding(nn.Module):
    def __init__(self, dim, p_dropout=0., max_len=5000):
        super(PositionalEmbedding, self).__init__()

        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(1, 2)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[..., :x.size(2)]
        return x


class LengthRegulator(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, dur):
        output = []
        for x_i, dur_i in zip(x, dur):
            expanded = self.expand(x_i, dur_i)
            output.append(expanded)
        output = self.pad(output)
        return output

    def expand(self, x, dur):
        output = []
        for i, frame in enumerate(x):
            expanded_len = int(dur[i] + 0.5)
            expanded = frame.expand(expanded_len, -1)
            output.append(expanded)
        output = torch.cat(output, 0)
        return output

    def pad(self, x):
        output = []
        max_len = max([x[i].size(0) for i in range(len(x))])
        for i, seq in enumerate(x):
            padded = F.pad(seq, [0, 0, 0, max_len - seq.size(0)], 'constant', 0.0)
            output.append(padded)
        output = torch.stack(output)
        return output


class ConvLSTMLinear(nn.Module):
    def __init__(self, in_dim, out_dim, n_layers=2, n_channels=256,
                 kernel_size=3, p_dropout=0.1, decoupled=False):
        super(ConvLSTMLinear, self).__init__()
        self.dropout = nn.Dropout(p=p_dropout)
        self.decoupled = decoupled

        convolutions = []
        for i in range(n_layers):
            conv_layer = ConvNorm(
                in_dim if i == 0 else n_channels, n_channels,
                kernel_size=kernel_size, stride=1,
                padding=int((kernel_size - 1) / 2), dilation=1,
                w_init_gain='relu')
            conv_layer = torch.nn.utils.weight_norm(
                conv_layer.conv, name='weight')
            convolutions.append(conv_layer)

        self.convolutions = nn.ModuleList(convolutions)

        self.bilstm = nn.LSTM(n_channels, int(n_channels // 2), 1,
                              batch_first=True, bidirectional=True)
        lstm_norm_fn_pntr = nn.utils.spectral_norm
        self.bilstm = lstm_norm_fn_pntr(self.bilstm, 'weight_hh_l0')
        self.bilstm = lstm_norm_fn_pntr(self.bilstm, 'weight_hh_l0_reverse')
        if self.decoupled:
            conv_f = ConvNorm(
                n_channels, n_channels, kernel_size=3, stride=1, padding=1,
                dilation=1, w_init_gain='relu')
            dense_f = LinearNorm(n_channels, 1, w_init_gain='linear')
            self.conv_f = torch.nn.utils.weight_norm(conv_f.conv)
            self.dense_f = torch.nn.utils.weight_norm(dense_f.linear_layer)

            conv_e = ConvNorm(
                n_channels, n_channels, kernel_size=3, stride=1, padding=1,
                dilation=1, w_init_gain='relu')
            dense_e = LinearNorm(n_channels, 1, w_init_gain='linear')
            self.conv_e = torch.nn.utils.weight_norm(conv_e.conv)
            self.dense_e = torch.nn.utils.weight_norm(dense_e.linear_layer)

            conv_v = ConvNorm(
                n_channels, n_channels, kernel_size=3, stride=1, padding=1,
                dilation=1, w_init_gain='relu')
            dense_v = LinearNorm(n_channels, 1, w_init_gain='sigmoid')
            self.conv_v = torch.nn.utils.weight_norm(conv_v.conv)
            self.dense_v = torch.nn.utils.weight_norm(dense_v.linear_layer)
        else:
            self.dense_fev = nn.Linear(n_channels, out_dim)

    def run_padded_sequence(self, context, lens):
        context_embedded = []
        for b_ind in range(context.size()[0]):  # TODO: speed up
            curr_context = context[b_ind:b_ind+1, :, :lens[b_ind]].clone()
            for conv in self.convolutions:
                curr_context = self.dropout(F.relu(conv(curr_context)))
            context_embedded.append(curr_context[0].transpose(0, 1))
        context = torch.nn.utils.rnn.pad_sequence(
            context_embedded, batch_first=True)
        return context

    def run_unsorted_inputs(self, fn, context, lens):
        lens_sorted, ids_sorted = torch.sort(lens, descending=True)
        unsort_ids = [0] * lens.size(0)
        for i in range(len(ids_sorted)):
            unsort_ids[ids_sorted[i]] = i
        lens_sorted = lens_sorted.long().cpu()

        context = context[ids_sorted]
        context = nn.utils.rnn.pack_padded_sequence(
            context, lens_sorted, batch_first=True)
        context = fn(context)[0]
        context = nn.utils.rnn.pad_packed_sequence(
            context, batch_first=True)[0]

        # map back to original indices
        context = context[unsort_ids]
        return context

    def forward(self, context, lens):
        if context.size()[0] > 1:
            context = self.run_padded_sequence(context, lens)
        else:
            for conv in self.convolutions:
                context = self.dropout(F.relu(conv(context)))
            context = context.transpose(1, 2)

        self.bilstm.flatten_parameters()
        if lens is not None:
            context = self.run_unsorted_inputs(self.bilstm, context, lens)
        else:
            context = self.bilstm(context)[0]

        if self.decoupled:
            context = context.permute(0, 2, 1)
            f = torch.relu(self.conv_f(context))
            e = torch.relu(self.conv_e(context))
            v = torch.relu(self.conv_v(context))

            f = self.dense_f(f.permute(0, 2, 1))
            e = self.dense_e(e.permute(0, 2, 1))
            v = self.dense_v(v.permute(0, 2, 1))

            x_hat = torch.cat((f, e, v), dim=2).permute(0, 2, 1)
        else:
            x_hat = self.dense_fev(context).permute(0, 2, 1)

        return x_hat

    def infer(self, z, txt_enc, spk_emb):
        x_hat = self.forward(txt_enc, spk_emb, None, None)['x_hat']
        x_hat = self.feature_processing.denormalize(x_hat)
        return x_hat


class Encoder(nn.Module):
    """Encoder module:
        - Three 1-d convolution banks
        - Bidirectional LSTM
    """
    def __init__(self, encoder_n_convolutions=3, encoder_embedding_dim=512,
                 encoder_kernel_size=5, norm_fn=nn.BatchNorm1d,
                 lstm_norm_fn=None):
        super(Encoder, self).__init__()

        convolutions = []
        for _ in range(encoder_n_convolutions):
            conv_layer = nn.Sequential(
                ConvNorm(encoder_embedding_dim,
                         encoder_embedding_dim,
                         kernel_size=encoder_kernel_size, stride=1,
                         padding=int((encoder_kernel_size - 1) / 2),
                         dilation=1, w_init_gain='relu'),
                norm_fn(encoder_embedding_dim, affine=True))
            convolutions.append(conv_layer)
        self.convolutions = nn.ModuleList(convolutions)

        self.lstm = nn.LSTM(encoder_embedding_dim,
                            int(encoder_embedding_dim / 2), 1,
                            batch_first=True, bidirectional=True)
        if lstm_norm_fn is not None:
            if 'spectral' in lstm_norm_fn:
                logging.info("Applying spectral norm to text encoder LSTM")
                lstm_norm_fn_pntr = torch.nn.utils.spectral_norm
            elif 'weight' in lstm_norm_fn:
                logging.info("Applying weight norm to text encoder LSTM")
                lstm_norm_fn_pntr = torch.nn.utils.weight_norm
            self.lstm = lstm_norm_fn_pntr(self.lstm, 'weight_hh_l0')
            self.lstm = lstm_norm_fn_pntr(self.lstm, 'weight_hh_l0_reverse')

    @amp.autocast(False)
    def forward(self, x, in_lens):
        """
        Args:
            x (torch.tensor): N x C x L padded input of text embeddings
            in_lens (torch.tensor): 1D tensor of sequence lengths
        """
        if x.size()[0] > 1:
            x_embedded = []
            for b_ind in range(x.size()[0]):  # TODO: improve speed
                curr_x = x[b_ind:b_ind+1, :, :in_lens[b_ind]].clone()
                for conv in self.convolutions:
                    curr_x = F.dropout(F.relu(conv(curr_x)), 0.5, self.training)
                x_embedded.append(curr_x[0].transpose(0, 1))
            x = torch.nn.utils.rnn.pad_sequence(x_embedded, batch_first=True)
        else:
            for conv in self.convolutions:
                x = F.dropout(F.relu(conv(x)), 0.5, self.training)
            x = x.transpose(1, 2)

        # recent amp change -- change in_lens to int
        in_lens = in_lens.int().cpu()
        in_lens = sorted(in_lens, reverse=True)
        x = nn.utils.rnn.pack_padded_sequence(x, in_lens, batch_first=True)

        self.lstm.flatten_parameters()
        outputs, _ = self.lstm(x)

        outputs, _ = nn.utils.rnn.pad_packed_sequence(
            outputs, batch_first=True)

        return outputs

    @amp.autocast(False)
    def infer(self, x):
        for conv in self.convolutions:
            x = F.dropout(F.relu(conv(x)), 0.5, self.training)

        x = x.transpose(1, 2)
        self.lstm.flatten_parameters()
        outputs, _ = self.lstm(x)

        return outputs


# 1x1 conv variants

class Invertible1x1ConvL(torch.nn.Module):
    def __init__(self, c):
        super(Invertible1x1ConvL, self).__init__()
        # Sample a random orthonormal matrix to initialize weights
        p = torch.eye(c, c)[:, torch.randperm(c)]
        self.register_buffer('p', p)
        lower = torch.eye(c, c)
        self.lower = nn.Parameter(lower)

    def forward(self, z, reverse=False):
        L = torch.tril(self.lower)
        W = torch.mm(self.p, L)
        if reverse:
            if not hasattr(self, 'W_inverse'):
                # Reverse computation
                W_inverse = W.float().inverse()
                if z.type() == 'torch.cuda.HalfTensor':
                    W_inverse = W_inverse.half()

                self.W_inverse = W_inverse[..., None]
            z = F.conv1d(z, self.W_inverse, bias=None, stride=1, padding=0)
            return z
        else:
            log_det_W = torch.sum(torch.log(torch.abs(torch.diag(L))))
            W = W[..., None]
            z = F.conv1d(z, W, bias=None, stride=1, padding=0)

            return z, log_det_W


class Invertible1x1ConvLU(torch.nn.Module):
    def __init__(self, c):
        super(Invertible1x1ConvLU, self).__init__()
        # Sample a random orthonormal matrix to initialize weights
        W = torch.qr(torch.FloatTensor(c, c).normal_())[0]
        # Ensure determinant is 1.0 not -1.0
        if torch.det(W) < 0:
            W[:, 0] = -1*W[:, 0]
        p, lower, upper = torch.lu_unpack(*torch.lu(W))

        self.register_buffer('p', p)
        self.lower = nn.Parameter(lower)
        self.upper = nn.Parameter(upper)

    def forward(self, z, reverse=False):
        U = torch.triu(self.upper)
        L = torch.tril(self.lower)
        W = torch.mm(self.p, torch.mm(L, U))

        if reverse:
            if not hasattr(self, 'W_inverse'):
                # Reverse computation
                W_inverse = W.float().inverse()
                if z.type() == 'torch.cuda.HalfTensor':
                    W_inverse = W_inverse.half()

                self.W_inverse = W_inverse[..., None]
            z = F.conv1d(z, self.W_inverse, bias=None, stride=1, padding=0)
            return z
        else:
            W = W[..., None]
            z = F.conv1d(z, W, bias=None, stride=1, padding=0)
            log_det_W = torch.sum(torch.log(torch.abs(torch.diag(U))) +
                                  torch.log(torch.abs(torch.diag(L))))
            return z, log_det_W


class Invertible1x1ConvLUS(torch.nn.Module):
    def __init__(self, c):
        super(Invertible1x1ConvLUS, self).__init__()
        # Sample a random orthonormal matrix to initialize weights
        W = torch.qr(torch.FloatTensor(c, c).normal_())[0]
        # Ensure determinant is 1.0 not -1.0
        if torch.det(W) < 0:
            W[:, 0] = -1*W[:, 0]
        p, lower, upper = torch.lu_unpack(*torch.lu(W))

        self.register_buffer('p', p)
        # diagonals of lower will always be 1s anyway
        lower = torch.tril(lower, -1)
        lower_diag = torch.diag(torch.eye(c, c))
        self.register_buffer('lower_diag', lower_diag)
        self.lower = nn.Parameter(lower)
        self.upper_diag = nn.Parameter(torch.diag(upper))
        self.upper = nn.Parameter(torch.triu(upper, 1))

    @amp.autocast(False)
    def forward(self, z, reverse=False):
        U = torch.triu(self.upper, 1) + torch.diag(self.upper_diag)
        L = torch.tril(self.lower, -1) + torch.diag(self.lower_diag)
        W = torch.mm(self.p, torch.mm(L, U))
        if reverse:
            if not hasattr(self, 'W_inverse'):
                # Reverse computation
                W_inverse = W.float().inverse()
                if z.type() == 'torch.cuda.HalfTensor':
                    W_inverse = W_inverse.half()

                self.W_inverse = W_inverse[..., None]
            z = F.conv1d(z, self.W_inverse, bias=None, stride=1, padding=0)
            return z
        else:
            W = W[..., None]
            z = F.conv1d(z, W, bias=None, stride=1, padding=0)
            log_det_W = torch.sum(torch.log(torch.abs(self.upper_diag)))
            return z, log_det_W


class Invertible1x1Conv(torch.nn.Module):
    """
    The layer outputs both the convolution, and the log determinant
    of its weight matrix.  If reverse=True it does convolution with
    inverse
    """
    def __init__(self, c):
        super(Invertible1x1Conv, self).__init__()
        self.conv = torch.nn.Conv1d(c, c, kernel_size=1, stride=1, padding=0,
                                    bias=False)

        # Sample a random orthonormal matrix to initialize weights
        W = torch.qr(torch.FloatTensor(c, c).normal_())[0]

        # Ensure determinant is 1.0 not -1.0
        if torch.det(W) < 0:
            W[:, 0] = -1*W[:, 0]
        W = W.view(c, c, 1)
        self.conv.weight.data = W

    def forward(self, z, reverse=False):
        # DO NOT apply n_of_groups, as it doesn't account for padded sequences
        W = self.conv.weight.squeeze()

        if reverse:
            if not hasattr(self, 'W_inverse'):
                # Reverse computation
                W_inverse = W.float().inverse()
                if z.type() == 'torch.cuda.HalfTensor':
                    W_inverse = W_inverse.half()

                self.W_inverse = W_inverse[..., None]
            z = F.conv1d(z, self.W_inverse, bias=None, stride=1, padding=0)
            return z
        else:
            # Forward computation
            log_det_W = torch.logdet(W).clone()
            z = self.conv(z)
            return z, log_det_W


class SimpleConvNet(torch.nn.Module):
    def __init__(self, n_mel_channels, n_context_dim, n_layers=2,
                 kernel_size=5, with_dilation=True, max_channels=1024,
                 p_dropout=0.0):
        super(SimpleConvNet, self).__init__()
        self.layers = torch.nn.ModuleList()
        self.n_layers = n_layers
        in_channels = n_mel_channels + n_context_dim
        out_channels = -1
        for i in range(n_layers):
            dilation = 2 ** i if with_dilation else 1
            padding = int((kernel_size*dilation - dilation)/2)
            out_channels = min(max_channels, in_channels * 2)
            self.layers.append(ConvNorm(in_channels, out_channels,
                                        kernel_size=kernel_size, stride=1,
                                        padding=padding, dilation=dilation,
                                        bias=True, w_init_gain='relu'))
            in_channels = out_channels
        self.last_layer = torch.nn.Conv1d(
            out_channels, 2*n_mel_channels, kernel_size=1)
        self.last_layer.weight.data *= 0
        self.last_layer.bias.data *= 0
        if p_dropout > 0.0:
            self.dropout = torch.nn.Dropout(p=p_dropout)

    def forward(self, z_w_context, seq_lens=None):
        # output should be b x n_mel_channels x z_w_context.shape(2)
        for i in range(self.n_layers):
            z_w_context = self.layers[i](z_w_context)
            if hasattr(self, 'dropout'):
                z_w_context = self.dropout(z_w_context)
            z_w_context = torch.relu(z_w_context)

        z_w_context = self.last_layer(z_w_context)
        return z_w_context


class WNT(torch.nn.Module):
    """
    Adapted from WN() module in WaveGlow with modififcations to variable names
    """
    def __init__(self, n_in_channels, n_context_dim, n_layers, n_channels=256,
                 kernel_size=5, use_feature_gating=False,
                 affine_activation='softplus', enable_lstm=False,
                 p_dropout=0.0, use_pconv=False, use_transformer=True):
        super(WNT, self).__init__()
        assert(kernel_size % 2 == 1)
        assert(n_channels % 2 == 0)
        self.n_layers = n_layers
        self.n_channels = n_channels
        self.in_layers = torch.nn.ModuleList()
        self.res_skip_layers = torch.nn.ModuleList()
        self.use_feature_gating = use_feature_gating
        start = nn.Sequential(torch.nn.Conv1d(n_in_channels, n_channels, 1),
                              torch.nn.Softplus(),
                              torch.nn.Dropout(p=0.0),
                              torch.nn.Conv1d(n_channels, n_channels, 1))

        self.start = start
        self.softplus = torch.nn.Softplus()
        self.affine_activation = affine_activation
        self.p_dropout = p_dropout
        self.use_pconv = use_pconv
        self.use_transformer = use_transformer
        if self.use_transformer:
            from transformer import FFTransformer
            self.tformer = FFTransformer(
                n_channels, n_layers=n_layers, n_head=2, d_head=64,
                d_inner=512)
        # only matters for convs with kernel size > 1
        conv_fn = torch.nn.Conv1d

        if self.p_dropout:
            self.dropout = torch.nn.Dropout(p=p_dropout)

        # Initializing last layer to 0 makes the affine coupling layers
        # do nothing at first.  This helps with training stability
        end = torch.nn.Conv1d(n_channels, 2*n_in_channels, 1)
        end.weight.data.zero_()
        end.bias.data.zero_()

        self.end = end
        context_enc = nn.Sequential(
            torch.nn.Conv1d(n_context_dim, n_context_dim, 1), nn.Softplus(),
            nn.Dropout(p=0.0), torch.nn.Conv1d(n_context_dim, n_channels, 1))
        self.context_encoder = context_enc
        for i in range(n_layers):
            dilation = 2 ** i
            padding = int((kernel_size*dilation - dilation)/2)
            in_layer = conv_fn(n_channels, n_channels, kernel_size,
                               dilation=dilation, padding=padding)
            in_layer = torch.nn.utils.weight_norm(in_layer, name='weight')
            self.in_layers.append(in_layer)
            res_skip_layer = torch.nn.Conv1d(n_channels, n_channels, 1)
            res_skip_layer = torch.nn.utils.weight_norm(
                res_skip_layer, name='weight')
            self.res_skip_layers.append(res_skip_layer)

    def forward(self, forward_input, seq_lens=None):
        z, context = forward_input
        # append context to z as well

        z = self.start(z)
        context = self.context_encoder(context)
        z = z + context
        # z = fused_add_tanh_sigmoid_multiply(z, context)

        output = torch.zeros_like(z)

        for i in range(self.n_layers):
            if self.affine_activation == 'softplus':
                z = self.softplus(self.in_layers[i](z))
                if self.p_dropout:
                    z = self.dropout(z)
                res_skip_acts = self.softplus(self.res_skip_layers[i](z))
                if self.p_dropout:
                    res_skip_acts = self.dropout(res_skip_acts)
            else:  # ReLU network
                z = torch.relu(self.in_layers[i](z))
                if self.p_dropout:
                    z = self.dropout(z)
                res_skip_acts = torch.relu(self.res_skip_layers[i](z))
                if self.p_dropout:
                    res_skip_acts = self.dropout(res_skip_acts)
            output = output + res_skip_acts

        if self.use_transformer:
            output = output + self.tformer(output, seq_lens).permute(0, 2, 1)

        output = self.end(output)  # [B, dim, seq_len]
        return output


class WN(torch.nn.Module):
    """
    Adapted from WN() module in WaveGlow with modififcations to variable names
    """
    def __init__(self, n_in_channels, n_context_dim, n_layers, n_channels,
                 kernel_size=5, use_feature_gating=False,
                 affine_activation='softplus', enable_lstm=False,
                 p_dropout=0.0):
        super(WN, self).__init__()
        assert(kernel_size % 2 == 1)
        assert(n_channels % 2 == 0)
        self.n_layers = n_layers
        self.n_channels = n_channels
        self.in_layers = torch.nn.ModuleList()
        self.res_skip_layers = torch.nn.ModuleList()
        self.use_feature_gating = use_feature_gating
        start = torch.nn.Conv1d(n_in_channels+n_context_dim, n_channels, 1)
        start = torch.nn.utils.weight_norm(start, name='weight')
        self.start = start
        self.softplus = torch.nn.Softplus()
        self.affine_activation = affine_activation

        self.enable_lstm = enable_lstm

        if self.enable_lstm:
            ### adding in new BiLSTM layer at the end
            end_seq = torch.nn.LSTM(input_size=2*n_in_channels,
                                    hidden_size=n_in_channels, num_layers=1,
                                    batch_first=True, bidirectional=True)
            self.end_seq = end_seq

        if p_dropout > 0.0:
            self.dropout = torch.nn.Dropout(p=p_dropout)

        # Initializing last layer to 0 makes the affine coupling layers
        # do nothing at first.  This helps with training stability
        end = torch.nn.Conv1d(n_channels, 2*n_in_channels, 1)
        end.weight.data.zero_()
        end.bias.data.zero_()
        self.end = end
        if self.use_feature_gating:
            cond_layer = nn.Conv1d(n_context_dim, n_channels*n_layers, 1)
            self.cond_layer = nn.utils.weight_norm(cond_layer)

        for i in range(n_layers):
            dilation = 2 ** i
            padding = int((kernel_size*dilation - dilation)/2)
            in_layer = nn.Conv1d(n_channels, n_channels, kernel_size,
                                 dilation=dilation, padding=padding)
            in_layer = nn.utils.weight_norm(in_layer)
            self.in_layers.append(in_layer)
            res_skip_layer = nn.Conv1d(n_channels, n_channels, 1)
            res_skip_layer = nn.utils.weight_norm(res_skip_layer)
            self.res_skip_layers.append(res_skip_layer)

    def forward(self, forward_input, seq_lens=None):
        z, context = forward_input
        # append context to z as well
        z = torch.cat((z, context), 1)
        z = self.start(z)
        output = torch.zeros_like(z)

        if self.use_feature_gating:
            context = self.cond_layer(context)
            for i in range(self.n_layers):  # gated features
                context_offset = i*self.n_channels
                acts = fused_add_tanh_sigmoid_multiply(
                    self.in_layers[i](z),
                    context[:, context_offset:context_offset+self.n_channels, :])
                res_skip_acts = self.res_skip_layers[i](acts)
                output = output + res_skip_acts
        else:
            for i in range(self.n_layers):
                if self.affine_activation == 'softplus':
                    z = self.softplus(self.in_layers[i](z))
                    if hasattr(self, 'dropout'):
                        z = self.dropout(z)
                    res_skip_acts = self.softplus(self.res_skip_layers[i](z))
                    if hasattr(self, 'dropout'):
                        res_skip_acts = self.dropout(res_skip_acts)
                else: # ReLU network
                    z = torch.relu(self.in_layers[i](z))
                    if hasattr(self, 'dropout'):
                        z = self.dropout(z)
                    res_skip_acts = torch.relu(self.res_skip_layers[i](z))
                    if hasattr(self, 'dropout'):
                        res_skip_acts = self.dropout(res_skip_acts)
                output = output + res_skip_acts

        output = self.end(output) # [B, dim, seq_len]
        if self.enable_lstm:
            # [B, seq_len, dim]
            output, _ = self.end_seq(output.transpose(1, 2))
            # no explicit need to save hidden state: every batch is different
            # [B, dim, seq_len]
            output = output.transpose(1, 2)
        return output


class WNFEV(torch.nn.Module):
    def __init__(self, in_dim, out_dim, n_layers, n_channels, kernel_size=5,
                 p_dropout=0.0):
        super(WNFEV, self).__init__()
        assert(kernel_size % 2 == 1)
        assert(n_channels % 2 == 0)
        self.n_layers = n_layers
        self.n_channels = n_channels
        self.in_layers = torch.nn.ModuleList()
        self.res_skip_layers = torch.nn.ModuleList()

        start = torch.nn.Conv1d(in_dim, n_channels, 1)
        start = torch.nn.utils.weight_norm(start)
        end = torch.nn.Conv1d(n_channels, out_dim, 1)

        self.start = start
        self.end = end

        self.softplus = torch.nn.Softplus()
        self.dropout = torch.nn.Dropout(p=p_dropout)

        for i in range(n_layers):
            dilation = 2 ** i
            padding = int((kernel_size*dilation - dilation)/2)
            in_layer = nn.Conv1d(n_channels, n_channels, kernel_size,
                                 dilation=dilation, padding=padding)
            in_layer = nn.utils.weight_norm(in_layer)
            self.in_layers.append(in_layer)
            res_skip_layer = nn.Conv1d(n_channels, n_channels, 1)
            res_skip_layer = nn.utils.weight_norm(res_skip_layer)
            self.res_skip_layers.append(res_skip_layer)

    def forward(self, context, lens=None):
        z = self.start(context)
        output = torch.zeros_like(z)

        for i in range(self.n_layers):
            z = self.softplus(self.in_layers[i](z))
            z = self.dropout(z)
            res_skip_acts = self.softplus(self.res_skip_layers[i](z))
            res_skip_acts = self.dropout(res_skip_acts)
            output = output + res_skip_acts

        output = self.end(output) # [B, dim, seq_len]
        return output


class WNTFEV(torch.nn.Module):
    def __init__(self, in_dim, out_dim, n_layers, n_channels=256,
                 kernel_size=5, p_dropout=0.0):
        super(WNTFEV, self).__init__()
        assert(kernel_size % 2 == 1)
        assert(n_channels % 2 == 0)
        from transformer import FFTransformer

        self.n_layers = n_layers
        self.n_channels = n_channels
        self.in_layers = torch.nn.ModuleList()
        self.res_skip_layers = torch.nn.ModuleList()
        start = nn.Sequential(torch.nn.Conv1d(in_dim, n_channels, 1),
                              torch.nn.Softplus(),
                              torch.nn.Dropout(p=0.0),
                              torch.nn.Conv1d(n_channels, n_channels, 1))
        end = torch.nn.Conv1d(n_channels, out_dim, 1)
        self.start = start
        self.end = end

        self.tformer = FFTransformer(
            n_channels, n_layers=n_layers, n_head=2, d_head=64, d_inner=512)

        self.softplus = torch.nn.Softplus()
        self.dropout = torch.nn.Dropout(p=p_dropout)

        for i in range(n_layers):
            dilation = 2 ** i
            padding = int((kernel_size*dilation - dilation)/2)
            in_layer = nn.Conv1d(n_channels, n_channels, kernel_size,
                                 dilation=dilation, padding=padding)
            in_layer = torch.nn.utils.weight_norm(in_layer)
            self.in_layers.append(in_layer)

            res_skip_layer = torch.nn.Conv1d(n_channels, n_channels, 1)
            res_skip_layer = torch.nn.utils.weight_norm(res_skip_layer)
            self.res_skip_layers.append(res_skip_layer)

    def forward(self, context, seq_lens=None):
        z = self.start(context)
        output = torch.zeros_like(z)

        for i in range(self.n_layers):
            z = self.softplus(self.in_layers[i](z))
            z = self.dropout(z)
            res_skip_acts = self.softplus(self.res_skip_layers[i](z))
            res_skip_acts = self.dropout(res_skip_acts)
            output = output + res_skip_acts

        output = output + self.tformer(output, seq_lens).permute(0, 2, 1)

        # [B, dim, seq_len]
        output = self.end(output)
        return output


class AffineTransformationLayer(torch.nn.Module):
    def __init__(self, n_mel_channels, n_context_dim, n_layers,
                 affine_model='simple_conv', with_dilation=True, kernel_size=5,
                 scaling_fn='exp', use_feature_gating=False,
                 affine_activation='softplus', enable_lstm=False,
                 p_dropout=0.0, n_channels=1024):
        super(AffineTransformationLayer, self).__init__()
        if affine_model not in ("wavenet", "simple_conv", "wnt"):
            raise Exception("{} affine model not supported".format(affine_model))
        if isinstance(scaling_fn, list):
            if not all([x in ("translate", "exp", "tanh", "sigmoid") for x in scaling_fn]):
                raise Exception("{} scaling fn not supported".format(scaling_fn))
        else:
            if scaling_fn not in ("translate", "exp", "tanh", "sigmoid"):
                raise Exception("{} scaling fn not supported".format(scaling_fn))

        self.affine_model = affine_model
        self.scaling_fn = scaling_fn
        if affine_model == 'wavenet':
            self.affine_param_predictor = WN(
                int(n_mel_channels/2), n_context_dim, n_layers=n_layers,
                n_channels=n_channels, use_feature_gating=use_feature_gating,
                affine_activation=affine_activation, enable_lstm=enable_lstm,
                p_dropout=p_dropout)
        elif affine_model == 'simple_conv':
            self.affine_param_predictor = SimpleConvNet(
                int(n_mel_channels/2), n_context_dim, n_layers,
                with_dilation=with_dilation, kernel_size=kernel_size,
                p_dropout=p_dropout)
        elif affine_model == 'wnt':
            self.affine_param_predictor = WNT(
                int(n_mel_channels/2), n_context_dim, n_layers=n_layers,
                n_channels=n_channels, use_feature_gating=use_feature_gating,
                affine_activation=affine_activation, enable_lstm=enable_lstm,
                p_dropout=p_dropout)
        self.n_mel_channels = n_mel_channels

    def get_scaling_and_logs(self, scale_unconstrained):
        # (rvalle) re-write this
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
            for i in range(scale_unconstrained.shape[1]):
                scaling_i = self.scaling_fn[i]
                if scaling_i == 'translate':
                    s_i = torch.exp(scale_unconstrained[:i]*0)
                    log_s_i = scale_unconstrained[:, i]*0
                elif scaling_i == 'exp':
                    s_i = torch.exp(scale_unconstrained[:, i])
                    log_s_i = scale_unconstrained[:, i]
                elif scaling_i == 'tanh':
                    s_i = torch.tanh(scale_unconstrained[:, i]) + 1 + 1e-6
                    log_s_i = torch.log(s_i)
                elif scaling_i == 'sigmoid':
                    s_i = torch.sigmoid(scale_unconstrained[:, i]) + 1e-6
                    log_s_i = torch.log(s_i)
                s_list.append(s_i[:, None])
                log_s_list.append(log_s_i[:, None])
            s = torch.cat(s_list, dim=1)
            log_s = torch.cat(log_s_list, dim=1)
        return s, log_s

    def forward(self, z, context, reverse=False, lens=None):
        n_half = int(self.n_mel_channels / 2)
        z_0, z_1 = z[:, :n_half], z[:, n_half:]
        if self.affine_model == 'wavenet':
            affine_params = self.affine_param_predictor((z_0, context))
        elif self.affine_model == 'simple_conv':
            z_w_context = torch.cat((z_0, context), 1)
            affine_params = self.affine_param_predictor(z_w_context)
        elif self.affine_model == 'wnt':
            affine_params = self.affine_param_predictor((z_0, context), lens)

        scale_unconstrained = affine_params[:, :n_half, :]
        b = affine_params[:, n_half:, :]
        s, log_s = self.get_scaling_and_logs(scale_unconstrained)

        if reverse:
            z_1 = (z_1 - b) / s
            z = torch.cat((z_0, z_1), dim=1)
            return z
        else:
            z_1 = s * z_1 + b
            z = torch.cat((z_0, z_1), dim=1)
            return z, log_s


class ConvAttention(torch.nn.Module):
    def __init__(self, n_mel_channels=80, n_speaker_dim=128,
                 n_text_channels=512, n_att_channels=80, temperature=1.0,
                 n_mel_convs=2, align_query_enc_type='3xconv',
                 use_query_proj=True, passthru=False):
        super(ConvAttention, self).__init__()
        self.temperature = temperature
        self.att_scaling_factor = np.sqrt(n_att_channels)
        self.softmax = torch.nn.Softmax(dim=3)
        self.log_softmax = torch.nn.LogSoftmax(dim=3)
        self.query_proj = Invertible1x1ConvLUS(n_mel_channels)
        self.attn_proj = torch.nn.Conv2d(n_att_channels, 1, kernel_size=1)
        self.align_query_enc_type = align_query_enc_type
        self.use_query_proj = bool(use_query_proj)

        self.passthru = bool(passthru)

        if not self.passthru:
            self.key_proj = nn.Sequential(
                ConvNorm(n_text_channels, n_text_channels*2, kernel_size=3,
                         bias=True, w_init_gain='relu'),
                torch.nn.ReLU(),
                ConvNorm(n_text_channels*2, n_att_channels, kernel_size=1,
                         bias=True))
            self.align_query_enc_type = align_query_enc_type

            if align_query_enc_type == "inv_conv":
                self.query_proj = Invertible1x1ConvLUS(n_mel_channels)
            elif align_query_enc_type == "3xconv":
                self.query_proj = nn.Sequential(ConvNorm(n_mel_channels, n_mel_channels*2, kernel_size=3, bias=True, w_init_gain='relu'),
                torch.nn.ReLU(),
                ConvNorm(n_mel_channels*2, n_mel_channels, kernel_size=1, bias=True),
                torch.nn.ReLU(),
                ConvNorm(n_mel_channels, n_att_channels, kernel_size=1, bias=True))
            else:
                raise ValueError("Unknown query encoder type specified")

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
        padded_data = nn.utils.rnn.pack_padded_sequence(padded_data, lens)
        hidden_vectors = recurrent_model(padded_data)[0]
        hidden_vectors, _ = nn.utils.rnn.pad_packed_sequence(hidden_vectors)
        # unsort the results at dim=1 and return
        hidden_vectors = hidden_vectors[:, unsort_idx]
        return hidden_vectors

    def encode_query(self, query, query_lens):
        query = query.permute(2, 0, 1)  # seq_len, batch, feature dim
        lens, ids = torch.sort(query_lens, descending=True)
        original_ids = [0] * lens.size(0)
        for i in range(len(ids)):
            original_ids[ids[i]] = i

        query_encoded = self.run_padded_sequence(ids, original_ids, lens, query, self.query_lstm)
        query_encoded = query_encoded.permute(1, 2, 0)
        return query_encoded

    def forward(self, queries, keys, query_lens, mask=None, key_lens=None, keys_encoded=None, attn_prior=None):
        """Attention mechanism for flowtron parallel
        Unlike in Flowtron, we have no restrictions such as causality etc,
        since we only need this during training.

        Args:
            queries (torch.tensor): B x C x T1 tensor (probably going to be mel data)
            keys (torch.tensor): B x C2 x T2 tensor (text data)
            query_lens: lengths for sorting the queries in descending order
            mask (torch.tensor): uint8 binary mask for variable length entries (should be in the T2 domain)
        Output:
            attn (torch.tensor): B x 1 x T1 x T2 attention mask. Final dim T2 should sum to 1
        """
        cost = None
        temp = 0.0005
        if self.passthru:
            temp = 0.5
            keys_enc = keys.clone()
            queries_enc = queries.clone()
        else:
            keys_enc = self.key_proj(keys)  # B x n_attn_dims x T2
            # Beware can only do this since query_dim = attn_dim = n_mel_channels
            if self.use_query_proj:
                if self.align_query_enc_type == "inv_conv":
                    queries_enc, log_det_W = self.query_proj(queries)
                elif self.align_query_enc_type == "3xconv":
                    queries_enc = self.query_proj(queries)
                    log_det_W = 0.0
                else:
                    queries_enc, log_det_W = self.query_proj(queries)
            else:
                queries_enc, log_det_W = queries, 0.0

        # different ways of computing attn, one is isotopic gaussians (per phoneme)
        # Simplistic Gaussian Isotopic Attention
        attn = (queries_enc[:, :, :, None] - keys_enc[:, :, None])**2  # B x n_attn_dims x T1 x T2

        attn = -temp*attn.sum(1, keepdim=True) # compute log likelihood from a gaussian
        if attn_prior is not None:
            attn = self.log_softmax(attn) + torch.log(attn_prior[:, None]+1e-8)

        attn_logprob = attn.clone()

        if mask is not None:
            attn.data.masked_fill_(mask.permute(0, 2, 1).unsqueeze(2), -float("inf"))

        attn = self.softmax(attn) # softmax along T2
        return attn, attn_logprob


class AdditiveAugmentationLayer(torch.nn.Module):
    def __init__(self, n_mel_channels, n_augs=2, aug_probs=[0.6, .2, .2]):
        """ Invertible augmentation based on channel scaling
        Args:
            n_augs (int): number of randomly initialized augmentations
            aug_probs (list[float]): list of augmentation probabilities. First one should correspond to pass-thru
        """
        super(AdditiveAugmentationLayer, self).__init__()
        assert(len(aug_probs) == n_augs + 1)
        self.n_augs = n_augs
        self.aug_transforms = []

        for i in range(n_augs):
            scaling_vec = torch.tensor(np.random.normal(loc=0, scale=0.5, size=n_mel_channels), requires_grad=False).float()
            self.register_buffer('aug_'+str(i), scaling_vec)
        self.aug_probs = aug_probs

    def sample_aug_idx(self):
        return int(np.random.choice(np.arange(self.n_augs+1),1, p=self.aug_probs))

    def forward(self, z, aug_idx=0):
        """
        z : B x n_mel_channels x out_len
        """
        if aug_idx == 0:
            return z
        else:
            return z + getattr(self, 'aug_' + str(aug_idx-1)).reshape(1, -1, 1)
