import torch
from torch import nn
from torch.nn import functional as F
from hydra.utils import instantiate

from nemo.collections.tts.helpers.common import get_mask_from_lengths
from nemo.collections.tts.helpers.common import ConvNorm, Invertible1x1Conv
from nemo.collections.tts.helpers.common import PositionalEmbedding, AffineTransformationLayer
from nemo.collections.tts.helpers.common import ConvLSTMLinear, WNFEV, WNTFEV
from nemo.collections.tts.helpers.flowtron import AR_Step, AR_Back_Step


def get_FP_model(config):
    name = config['name']
    hparams = config['hparams']
    if name == 'dp':
        model = FP(**hparams)
        #model = instantiate(hparams)
    elif name == 'dpfp':
        model = FPFP(**hparams)
    elif name == 'dpfa':
        model = FPFA(**hparams)

    return model


class FeatureProcessing():
    def __init__(self, take_log_of_input=False):
        super().__init__()
        self.take_log_of_input = take_log_of_input

    def normalize(self, x):
        if self.take_log_of_input:
            x = torch.log(x + 1)
        return x

    def denormalize(self, x):
        if self.take_log_of_input:
            x = torch.exp(x) - 1
        return x


class FP(nn.Module):
    def __init__(self, arch, n_speaker_dim, n_text_dim, reduction_factor,
                 take_log_of_input, arch_hparams):
        super().__init__()
        self.feature_processing = FeatureProcessing(take_log_of_input)

        n_txt_reduced_dim = int(n_text_dim / reduction_factor)
        if reduction_factor > 1:
            fn = ConvNorm(n_text_dim, n_txt_reduced_dim, kernel_size=3)
            fn = torch.nn.utils.weight_norm(fn.conv)
            self.txt_proj = fn

        in_dim = n_txt_reduced_dim + n_speaker_dim
        arch_hparams['in_dim'] = in_dim
        #arch_hparams.in_dim = in_dim

        if arch == 'convlstmlinear':
            self.feat_pred_fn = ConvLSTMLinear(**arch_hparams)
        elif arch == 'wnfev':
            self.feat_pred_fn = WNFEV(**arch_hparams)
        elif arch == 'wntfev':
            self.feat_pred_fn = WNTFEV(**arch_hparams)

    def forward(self, txt_enc, spk_emb, x, lens):
        if x is not None:
            x = self.feature_processing.normalize(x)

        if hasattr(self, 'txt_proj'):
            txt_enc = torch.relu(self.txt_proj(txt_enc))

        spk_emb_expanded = spk_emb[..., None].expand(-1, -1, txt_enc.shape[2])
        context = torch.cat((txt_enc, spk_emb_expanded), 1)

        x_hat = self.feat_pred_fn(context, lens)

        outputs = {'x_hat': x_hat, 'x': x}
        return outputs

    def infer(self, z, txt_enc, spk_emb):
        x_hat = self.forward(txt_enc, spk_emb, None, None)['x_hat']
        x_hat = self.feature_processing.denormalize(x_hat)
        return x_hat


class FPFP(torch.nn.Module):
    def __init__(self, n_in_dim, n_speaker_dim, n_text_dim, n_flows,
                 n_group_size, n_layers, reduction_factor,
                 with_dilation, kernel_size, scaling_fn, reduction_norm,
                 use_positional_embedding, take_log_of_input=False,
                 p_dropout=0.0, affine_model='simple_conv', n_channels=1024):
        super(FPFP, self).__init__()
        assert(n_group_size % 2 == 0)
        self.n_flows = n_flows
        self.n_group_size = n_group_size
        self.transforms = torch.nn.ModuleList()
        self.convinv = torch.nn.ModuleList()
        self.reduction_factor = reduction_factor
        self.n_speaker_dim = n_speaker_dim
        self.scaling_fn = scaling_fn
        self.feature_processing = FeatureProcessing(take_log_of_input)
        n_txt_reduced_dim = int(n_text_dim / reduction_factor)
        context_dim = n_txt_reduced_dim * n_group_size + n_speaker_dim

        self.use_positional_embedding = use_positional_embedding
        if use_positional_embedding:
            self.pos_emb = PositionalEmbedding(n_txt_reduced_dim, max_len=5000)

        if self.n_group_size > 1:
            self.unfold_params = {'kernel_size': (n_group_size, 1),
                                  'stride': n_group_size,
                                  'padding': 0, 'dilation': 1}
            self.unfold = nn.Unfold(**self.unfold_params)

        if self.reduction_factor > 1:
            fn = ConvNorm(n_text_dim, n_txt_reduced_dim, kernel_size=3)
            if reduction_norm == 'weightnorm':
                fn = torch.nn.utils.weight_norm(fn.conv, name='weight')
            elif reduction_norm == 'instancenorm':
                fn = nn.Sequential(
                    fn, nn.InstanceNorm1d(n_txt_reduced_dim, affine=True))
            self.txt_proj = fn

        for k in range(n_flows):
            self.convinv.append(Invertible1x1Conv(n_in_dim * n_group_size))
            self.transforms.append(AffineTransformationLayer(
                n_in_dim * n_group_size, context_dim, n_layers,
                with_dilation=with_dilation, kernel_size=kernel_size,
                scaling_fn=scaling_fn, p_dropout=p_dropout,
                affine_model=affine_model, n_channels=n_channels))

    def pad_inputs(self, text, durations):
        """ replication pad to multiple of n_group_size """
        padding = 0
        if text.shape[2] % self.n_group_size != 0:
            padding = self.n_group_size - text.shape[2] % self.n_group_size
            pad_fn = nn.ReplicationPad1d((0, padding))
            text = pad_fn(text)
            durations = F.pad(durations, (0, padding, 0, 0), 'constant', 0)

        return text, durations, padding

    def fold(self, data):
        """Inverse of the self.unfold(data.unsqueeze(-1)) operation used for
        the grouping or "squeeze" operation on input

        Args:
            data: B x C x T tensor of temporal data
        """
        output_size = (data.shape[2]*self.n_group_size, 1)
        data = nn.functional.fold(
            data, output_size=output_size, **self.unfold_params).squeeze(-1)
        return data

    def preprocess_context(self, txt_emb, speaker_vecs):
        if self.use_positional_embedding:
            txt_emb = self.pos_emb(txt_emb)
        if self.n_group_size > 1:
            txt_emb = self.unfold(txt_emb[..., None])
        speaker_vecs = speaker_vecs[..., None].expand(-1, -1, txt_emb.shape[2])
        context = torch.cat((txt_emb, speaker_vecs), 1)
        return context

    def forward(self, txt_emb, spk_emb, x, lens):
        """x<tensor>: duration or pitch or energy average"""
        assert(txt_emb.size(2) >= x.size(1))

        if len(x.shape) == 2:
            # add channel dimension
            x = x[:, None]
        x = self.feature_processing.normalize(x)

        if hasattr(self, 'txt_proj'):
            txt_emb = torch.relu(self.txt_proj(txt_emb))

        # pad to support larger groups
        txt_emb, x, padding = self.pad_inputs(txt_emb, x)

        # lens including padded values
        lens_groupped = torch.ceil(lens / self.n_group_size).long()

        context = self.preprocess_context(txt_emb, spk_emb)
        x = self.unfold(x[..., None])

        log_s_list, log_det_W_list = [], []
        for k in range(self.n_flows):
            x, log_det_W = self.convinv[k](x)
            x, log_s = self.transforms[k](x, context, lens=lens_groupped)
            log_det_W_list.append(log_det_W)
            log_s_list.append(log_s)

        # remove padded data
        if padding > 0:
            x = x[..., :-padding]
            log_s_list = [l[:, :, :-padding] for l in log_s_list]

        # prepare outputs
        outputs = {'z': x,
                   'log_det_W_list': log_det_W_list,
                   'log_s_list': log_s_list}

        return outputs

    def infer(self, z, txt_enc, spk_emb):
        if hasattr(self, 'txt_proj'):
            txt_enc = torch.relu(self.txt_proj(txt_enc))

        # pad to support larger groups
        txt_enc, z, padding = self.pad_inputs(txt_enc, z)

        context = self.preprocess_context(txt_enc, spk_emb)
        z = self.unfold(z[..., None])
        for k in reversed(range(self.n_flows)):
            z = self.transforms[k].forward(z, context, reverse=True)
            z = self.convinv[k](z, reverse=True)

        # z mapped to input domain
        x_hat = self.fold(z)

        # remove padded data
        if padding > 0:
            x_hat = x_hat[..., :-padding]

        x_hat = self.feature_processing.denormalize(x_hat)
        return x_hat


class FPFA(torch.nn.Module):
    def __init__(self, n_in_dim, n_speaker_dim, n_text_dim, n_flows, n_hidden,
                 n_lstm_layers, scaling_fn='exp', reduction_factor=1,
                 reduction_norm='instancenorm', use_positional_embedding=False,
                 take_log_of_input=False, p_dropout=0.0, setup='',
                 with_conv_in="", conv_in_nonlinearity="", kernel_size=1,
                 n_convs=1, scaling=1.0):
        super(FPFA, self).__init__()
        self.flows = torch.nn.ModuleList()
        self.reduction_factor = reduction_factor
        self.n_speaker_dim = n_speaker_dim
        self.feature_processing = FeatureProcessing(take_log_of_input)
        self.n_in_dim = n_in_dim
        self.setup = setup
        self.scaling = scaling
        n_out_dims = None
        if setup == 'alternate':
            n_out_dims = 2
        n_txt_reduced_dim = int(n_text_dim / reduction_factor)

        self.use_positional_embedding = use_positional_embedding
        if use_positional_embedding:
            self.pos_emb = PositionalEmbedding(n_txt_reduced_dim, max_len=5000)

        for i in range(n_flows):
            if i % 2 == 0:
                self.flows.append(AR_Step(
                    n_in_dim, n_speaker_dim, n_txt_reduced_dim, n_hidden,
                    n_lstm_layers, scaling_fn, p_dropout, n_out_dims,
                    with_conv_in, conv_in_nonlinearity, kernel_size, n_convs))
            else:
                self.flows.append(AR_Back_Step(
                    n_in_dim, n_speaker_dim, n_txt_reduced_dim, n_hidden,
                    n_lstm_layers, scaling_fn, p_dropout, n_out_dims,
                    with_conv_in, conv_in_nonlinearity, kernel_size, n_convs))

        if self.reduction_factor > 1:
            fn = ConvNorm(n_text_dim, n_txt_reduced_dim, kernel_size=3)
            if reduction_norm == 'weightnorm':
                fn = torch.nn.utils.weight_norm(fn.conv, name='weight')
            elif reduction_norm == 'instancenorm':
                fn = nn.Sequential(
                    fn, nn.InstanceNorm1d(n_txt_reduced_dim, affine=True))
            self.txt_proj = fn

    def forward(self, txt_emb, spk_emb, x, lens):
        """x<tensor>: duration or pitch or energy average"""
        x[:, 0] = x[:, 0] / self.scaling
        if len(x.shape) == 2:
            # add channel dimension
            x = x[:, None]
        x = x.permute(2, 0, 1)
        x = self.feature_processing.normalize(x)
        if hasattr(self, 'txt_proj'):
            txt_emb = torch.relu(self.txt_proj(txt_emb))

        context = torch.cat(
            (txt_emb, spk_emb[..., None].expand(-1, -1, txt_emb.shape[2])), 1)

        context = context.permute(2, 0, 1)

        log_s_list = []
        mask = ~get_mask_from_lengths(lens)[..., None]
        affine_dims = None
        for i, flow in enumerate(self.flows):
            if self.setup == 'alternate':
                affine_dims = [1, 2]
                if i % 2:
                    affine_dims = [0, 2]

            x, log_s = flow(x, context, mask, lens, lens, affine_dims)
            log_s_list.append(log_s)

        # x mapped to z
        x = x.permute(1, 2, 0)
        log_s_list = [log_s.permute(1, 2, 0) for log_s in log_s_list]
        outputs = {'z': x, 'log_s_list': log_s_list, 'log_det_W_list': []}
        return outputs

    def infer(self, z, txt_emb, spk_emb):
        z = z.permute(2, 0, 1)

        if hasattr(self, 'txt_proj'):
            txt_emb = torch.relu(self.txt_proj(txt_emb))

        context = torch.cat(
            (txt_emb, spk_emb[..., None].expand(-1, -1, txt_emb.shape[2])), 1)

        context = context.permute(2, 0, 1)
        affine_dims = None
        for i, flow in enumerate(reversed(self.flows)):
            if self.setup == 'alternate':
                affine_dims = [0, 2]
                if i % 2:
                    affine_dims = [1, 2]
            z = flow.infer(z, context, affine_dims)

        x_hat = z.permute(1, 2, 0)
        x_hat[:, 0] = x_hat[:, 0] * self.scaling
        x_hat = self.feature_processing.denormalize(x_hat)
        return x_hat
