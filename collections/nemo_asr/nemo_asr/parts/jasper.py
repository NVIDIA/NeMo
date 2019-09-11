# Taken straight from Patter https://github.com/ryanleary/patter
# TODO: review, and copyright and fix/add comments
import torch
import torch.nn as nn

jasper_activations = {
    "hardtanh": nn.Hardtanh,
    "relu": nn.ReLU,
    "selu": nn.SELU,
}


def init_weights(m, mode='xavier_uniform'):
    if isinstance(m, nn.Conv1d) or isinstance(m, MaskedConv1d):
        if mode == 'xavier_uniform':
            nn.init.xavier_uniform_(m.weight, gain=1.0)
        elif mode == 'xavier_normal':
            nn.init.xavier_normal_(m.weight, gain=1.0)
        elif mode == 'kaiming_uniform':
            nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
        elif mode == 'kaiming_normal':
            nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
        else:
            raise ValueError("Unknown Initialization mode: {0}".format(mode))
    elif isinstance(m, nn.BatchNorm1d):
        if m.track_running_stats:
            m.running_mean.zero_()
            m.running_var.fill_(1)
            m.num_batches_tracked.zero_()
        if m.affine:
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)


def get_same_padding(kernel_size, stride, dilation):
    if stride > 1 and dilation > 1:
        raise ValueError("Only stride OR dilation may be greater than 1")
    if dilation > 1:
        return (dilation * kernel_size) // 2 - 1
    return kernel_size // 2


class MaskedConv1d(nn.Conv1d):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, heads=-1, bias=False,
                 use_mask=True):

        if not (heads == -1 or groups == in_channels):
            raise ValueError("Only use heads for depthwise convolutions")

        if heads != -1:
            self.real_out_channels = out_channels
            in_channels = heads
            out_channels = heads
            groups = heads

        super(MaskedConv1d, self).__init__(in_channels, out_channels,
                                           kernel_size,
                                           stride=stride,
                                           padding=padding, dilation=dilation,
                                           groups=groups, bias=bias)
        self.use_mask = use_mask
        self.heads = heads

    def get_seq_len(self, lens):
        return ((lens + 2 * self.padding[0] - self.dilation[0] * (
                self.kernel_size[0] - 1) - 1) / self.stride[0] + 1)

    def forward(self, x, lens):
        if self.use_mask:
            lens = lens.to(dtype=torch.long)
            max_len = x.size(2)
            mask = torch.arange(max_len).to(lens.device)\
                .expand(len(lens), max_len) >= lens.unsqueeze(1)
            x = x.masked_fill(
                mask.unsqueeze(1).type(torch.bool).to(device=x.device), 0
            )
            del mask
            lens = self.get_seq_len(lens)

        if self.heads != -1:
            sh = x.shape
            x = x.view(-1, self.heads, sh[-1])

        out, lens = super(MaskedConv1d, self).forward(x), lens

        if self.heads != -1:
            out = out.view(sh[0], self.real_out_channels, -1)

        return out, lens


class GroupShuffle(nn.Module):

    def __init__(self, groups, channels):
        super(GroupShuffle, self).__init__()

        self.groups = groups
        self.channels_per_group = channels // groups

    def forward(self, x):
        sh = x.shape

        x = x.view(-1, self.groups, self.channels_per_group, sh[-1])

        x = torch.transpose(x, 1, 2).contiguous()

        x = x.view(-1, self.groups * self.channels_per_group, sh[-1])

        return x


class JasperBlock(nn.Module):

    def __init__(self, inplanes, planes, repeat=3, kernel_size=11, stride=1,
                 dilation=1, padding='same', dropout=0.2, activation=None,
                 residual=True, groups=1, separable=False,
                 heads=-1, tied=False, normalization="batch",
                 norm_groups=1, residual_mode='add',
                 residual_panes=[], conv_mask=False):
        super(JasperBlock, self).__init__()

        if padding != "same":
            raise ValueError("currently only 'same' padding is supported")

        padding_val = get_same_padding(kernel_size[0], stride[0], dilation[0])
        self.conv_mask = conv_mask
        self.separable = separable
        self.residual_mode = residual_mode

        self.conv = nn.ModuleList()
        inplanes_loop = inplanes

        if tied:
            rep_layer = self._get_conv_bn_layer(
                inplanes_loop,
                planes,
                kernel_size=kernel_size,
                stride=stride,
                dilation=dilation,
                padding=padding_val,
                groups=groups,
                heads=heads,
                separable=separable,
                normalization=normalization,
                norm_groups=norm_groups)

        for _ in range(repeat - 1):
            if tied:
                self.conv.extend(rep_layer)
            else:
                self.conv.extend(
                    self._get_conv_bn_layer(
                        inplanes_loop,
                        planes,
                        kernel_size=kernel_size,
                        stride=stride,
                        dilation=dilation,
                        padding=padding_val,
                        groups=groups,
                        heads=heads,
                        separable=separable,
                        normalization=normalization,
                        norm_groups=norm_groups))

            self.conv.extend(
                self._get_act_dropout_layer(
                    drop_prob=dropout,
                    activation=activation))

            inplanes_loop = planes

        if tied:
            self.conv.extend(rep_layer)
        else:
            self.conv.extend(
                self._get_conv_bn_layer(
                    inplanes_loop,
                    planes,
                    kernel_size=kernel_size,
                    stride=stride,
                    dilation=dilation,
                    padding=padding_val,
                    groups=groups,
                    heads=heads,
                    separable=separable,
                    normalization=normalization,
                    norm_groups=norm_groups))

        self.res = nn.ModuleList() if residual else None
        res_panes = residual_panes.copy()
        self.dense_residual = residual
        if residual:
            if len(residual_panes) == 0:
                res_panes = [inplanes]
                self.dense_residual = False
            for ip in res_panes:
                self.res.append(
                    nn.ModuleList(
                        modules=self._get_conv_bn_layer(
                            ip,
                            planes,
                            kernel_size=1,
                            normalization=normalization,
                            norm_groups=norm_groups)))
        self.out = nn.Sequential(
            *
            self._get_act_dropout_layer(
                drop_prob=dropout,
                activation=activation))

    def _get_conv_bn_layer(self, in_channels, out_channels, kernel_size=11,
                           stride=1, dilation=1, padding=0, bias=False,
                           groups=1, heads=-1, separable=False,
                           normalization="batch", norm_groups=1):
        if norm_groups == -1:
            norm_groups = out_channels

        if separable:
            layers = [
                MaskedConv1d(in_channels, in_channels, kernel_size,
                             stride=stride,
                             dilation=dilation, padding=padding, bias=bias,
                             groups=in_channels, heads=heads,
                             use_mask=self.conv_mask),
                MaskedConv1d(in_channels, out_channels, kernel_size=1,
                             stride=1,
                             dilation=1, padding=0, bias=bias, groups=groups,
                             use_mask=self.conv_mask)
            ]
        else:
            layers = [
                MaskedConv1d(in_channels, out_channels, kernel_size,
                             stride=stride,
                             dilation=dilation, padding=padding, bias=bias,
                             groups=groups,
                             use_mask=self.conv_mask)
            ]

        if normalization == "group":
            layers.append(nn.GroupNorm(
                num_groups=norm_groups, num_channels=out_channels))
        elif normalization == "instance":
            layers.append(nn.GroupNorm(
                num_groups=out_channels, num_channels=out_channels))
        elif normalization == "layer":
            layers.append(nn.GroupNorm(
                num_groups=1, num_channels=out_channels))
        elif normalization == "batch":
            layers.append(nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.1))
        else:
            raise ValueError(
                    f"Normalization method ({normalization}) does not match"
                    f" one of [batch, layer, group, instance].")

        if groups > 1:
            layers.append(GroupShuffle(groups, out_channels))
        return layers

    def _get_act_dropout_layer(self, drop_prob=0.2, activation=None):
        if activation is None:
            activation = nn.Hardtanh(min_val=0.0, max_val=20.0)
        layers = [
            activation,
            nn.Dropout(p=drop_prob)
        ]
        return layers

    def forward(self, input_):

        xs, lens_orig = input_

        # compute forward convolutions
        out = xs[-1]

        lens = lens_orig
        for i, l in enumerate(self.conv):
            # if we're doing masked convolutions, we need to pass in and
            # possibly update the sequence lengths
            # if (i % 4) == 0 and self.conv_mask:
            if isinstance(l, MaskedConv1d):
                out, lens = l(out, lens)
            else:
                out = l(out)

        # compute the residuals
        if self.res is not None:
            for i, layer in enumerate(self.res):
                res_out = xs[i]
                for j, res_layer in enumerate(layer):
                    if isinstance(res_layer, MaskedConv1d):
                        res_out, _ = res_layer(res_out, lens_orig)
                    else:
                        res_out = res_layer(res_out)

                if self.residual_mode == 'add':
                    out = out + res_out
                else:
                    out = torch.max(out, res_out)

        # compute the output
        out = self.out(out)
        if self.res is not None and self.dense_residual:
            return xs + [out], lens

        return [out], lens
