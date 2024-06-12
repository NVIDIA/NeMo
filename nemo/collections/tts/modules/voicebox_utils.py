import math
from typing import Optional, Dict
from random import random
from functools import partial, wraps
from collections import namedtuple

import einops
import torch
from torch import nn, Tensor
from torch.nn import Module
import torch.nn.functional as F


from nemo.core.classes import NeuralModule, typecheck
from nemo.core.neural_types import FloatType, LengthsType, NeuralType, SpectrogramType

__all__ = ['TransformerUNet']

# constants

FlashAttentionConfig = namedtuple('FlashAttentionConfig', ['enable_flash', 'enable_math', 'enable_mem_efficient'])

def once(fn):
    called = False
    @wraps(fn)
    def inner(x):
        nonlocal called
        if called:
            return
        called = True
        return fn(x)
    return inner

print_once = once(print)


# helper functions

def exists(val):
    return val is not None

def identity(t):
    return t

def default(val, d):
    return val if exists(val) else d

def divisible_by(num, den):
    return (num % den) == 0

def is_odd(n):
    return not divisible_by(n, 2)

def coin_flip():
    return random() < 0.5

def pack_one(t, pattern):
    return einops.pack([t], pattern)

def unpack_one(t, ps, pattern):
    return einops.unpack(t, ps, pattern)[0]

# tensor helpers

def prob_mask_like(shape, prob, device):
    if prob == 1:
        return torch.ones(shape, device = device, dtype = torch.bool)
    elif prob == 0:
        return torch.zeros(shape, device = device, dtype = torch.bool)
    else:
        return torch.zeros(shape, device = device).float().uniform_(0, 1) < prob

def reduce_masks_with_and(*masks):
    masks = [*filter(exists, masks)]

    if len(masks) == 0:
        return None

    mask, *rest_masks = masks

    for rest_mask in rest_masks:
        mask = mask & rest_mask

    return mask

def interpolate_1d(t, length, mode = 'bilinear'):
    " pytorch does not offer interpolation 1d, so hack by converting to 2d "

    dtype = t.dtype
    t = t.float()

    implicit_one_channel = t.ndim == 2
    if implicit_one_channel:
        t = einops.rearrange(t, 'b n -> b 1 n')

    t = einops.rearrange(t, 'b d n -> b d n 1')
    t = F.interpolate(t, (length, 1), mode = mode)
    t = einops.rearrange(t, 'b d n 1 -> b d n')

    if implicit_one_channel:
        t = einops.rearrange(t, 'b 1 n -> b n')

    t = t.to(dtype)
    return t

def curtail_or_pad(t, target_length):
    length = t.shape[-2]

    if length > target_length:
        t = t[..., :target_length, :]
    elif length < target_length:
        t = F.pad(t, (0, 0, 0, target_length - length), value = 0.)

    return t

# mask construction helpers

def mask_from_start_end_indices(
    seq_len: int,
    start: Tensor,
    end: Tensor
):
    assert start.shape == end.shape
    device = start.device

    seq = torch.arange(seq_len, device = device, dtype = torch.long)
    seq = seq.reshape(*((-1,) * start.ndim), seq_len)
    seq = seq.expand(*start.shape, seq_len)

    mask = seq >= start[..., None].long()
    mask &= seq < end[..., None].long()
    return mask

def mask_from_frac_lengths(
    seq_len: int,
    frac_lengths: Tensor
):
    device = frac_lengths.device

    lengths = (frac_lengths * seq_len).long()
    max_start = seq_len - lengths

    rand = torch.zeros_like(frac_lengths, device = device).float().uniform_(0, 1)
    start = (max_start * rand).clamp(min = 0)
    end = start + lengths

    return mask_from_start_end_indices(seq_len, start, end)

# sinusoidal positions

class LearnedSinusoidalPosEmb(Module):
    """ used by @crowsonkb """

    def __init__(self, dim):
        super().__init__()
        assert divisible_by(dim, 2)
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim))

    def forward(self, x):
        x = einops.rearrange(x, 'b -> b 1')
        freqs = x * einops.rearrange(self.weights, 'd -> 1 d') * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim = -1)
        return fouriered

# convolutional positional generating module

class ConvPositionEmbed(Module):
    def __init__(
        self,
        dim,
        *,
        kernel_size,
        groups = None
    ):
        super().__init__()
        assert is_odd(kernel_size)
        groups = default(groups, dim) # full depthwise conv by default

        self.dw_conv1d = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size, groups = groups, padding = kernel_size // 2),
            nn.GELU()
        )

    def forward(self, x, mask = None):

        if exists(mask):
            mask = mask[..., None]
            x = x.masked_fill(mask, 0.)

        x = einops.rearrange(x, 'b n c -> b c n')
        x = self.dw_conv1d(x)
        out = einops.rearrange(x, 'b c n -> b n c')

        if exists(mask):
            out = out.masked_fill(mask, 0.)

        return out

# norms

class RMSNorm(Module):
    def __init__(
        self,
        dim
    ):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return F.normalize(x, dim = -1) * self.scale * self.gamma

class AdaptiveRMSNorm(Module):
    def __init__(
        self,
        dim,
        cond_dim = None
    ):
        super().__init__()
        cond_dim = default(cond_dim, dim)
        self.scale = dim ** 0.5

        self.to_gamma = nn.Linear(cond_dim, dim)
        self.to_beta = nn.Linear(cond_dim, dim)

        # init to identity

        nn.init.zeros_(self.to_gamma.weight)
        nn.init.ones_(self.to_gamma.bias)

        nn.init.zeros_(self.to_beta.weight)
        nn.init.zeros_(self.to_beta.bias)

    def forward(self, x, *, cond):
        normed = F.normalize(x, dim = -1) * self.scale

        gamma, beta = self.to_gamma(cond), self.to_beta(cond)
        gamma, beta = map(lambda t: einops.rearrange(t, 'b d -> b 1 d'), (gamma, beta))

        return normed * gamma + beta

class GEGLU(Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim = -1)
        return F.gelu(gate) * x

def FeedForward(dim, mult = 4, dropout = 0.):
    dim_inner = int(dim * mult * 2 / 3)
    return nn.Sequential(
        nn.Linear(dim, dim_inner * 2),
        GEGLU(),
        nn.Dropout(dropout),
        nn.Linear(dim_inner, dim)
    )

# transformer

class Transformer(Module):
    def __init__(
        self,
        dim,
        *,
        depth,
        heads = 8,
        ff_mult = 4,
        attn_dropout = 0.,
        ff_dropout = 0.,
        max_positions = 3000, 
        adaptive_rmsnorm = False,
        adaptive_rmsnorm_cond_dim_in = None,
        use_unet_skip_connection = False,
        skip_connect_scale = None,
    ):
        super().__init__()
        assert divisible_by(depth, 2)
        self.layers = nn.ModuleList([])
        self.init_alibi(max_positions=max_positions, heads=heads)

        if adaptive_rmsnorm:
            rmsnorm_klass = partial(AdaptiveRMSNorm, cond_dim = adaptive_rmsnorm_cond_dim_in)
        else:
            rmsnorm_klass = RMSNorm

        self.skip_connect_scale = default(skip_connect_scale, 2 ** -0.5)

        for ind in range(depth):
            layer = ind + 1
            has_skip = use_unet_skip_connection and layer > (depth // 2)

            self.layers.append(nn.ModuleList([
                nn.Linear(dim * 2, dim) if has_skip else None,
                rmsnorm_klass(dim = dim),
                torch.nn.MultiheadAttention(
                    embed_dim=dim, 
                    num_heads=heads, 
                    dropout = attn_dropout,
                    batch_first=True,
                ),
                rmsnorm_klass(dim = dim),
                FeedForward(dim = dim, mult = ff_mult, dropout = ff_dropout)
            ]))

        self.final_norm = RMSNorm(dim)

    def init_alibi(
        self,
        max_positions: int,
        heads: int,
    ):
        def get_slopes(n):
            def get_slopes_power_of_2(n):
                start = 2 ** (-(2 ** -(math.log2(n) - 3)))
                ratio = start
                return [start * ratio**i for i in range(n)]
            if math.log2(n).is_integer():
                return get_slopes_power_of_2(n)
            else:
                closest_power_of_2 = 2 ** math.floor(math.log2(n))
                return (
                    get_slopes_power_of_2(closest_power_of_2)
                    + get_slopes(2 * closest_power_of_2)[0::2][: n - closest_power_of_2]
                )

        maxpos = max_positions
        attn_heads = heads
        self.slopes = nn.Parameter(
            torch.Tensor(get_slopes(attn_heads)).view(attn_heads, 1, 1)
        )
        
        pos_bias = (
            torch.abs(
                torch.arange(maxpos).unsqueeze(0) - torch.arange(maxpos).unsqueeze(1)
            )
            * -1
        ).view(1, maxpos, maxpos)
        self.register_buffer('pos_bias', pos_bias, persistent=False)
        return

    def forward(
        self,
        x,
        key_padding_mask: Optional[torch.Tensor] = None,
        adaptive_rmsnorm_cond = None
    ):
        batch_size, seq_len, *_ = x.shape

        # keep track of skip connections

        skip_connects = []

        alibi_bias = self.get_alibi_bias(batch_size=batch_size, seq_len=seq_len)

        # adaptive rmsnorm

        rmsnorm_kwargs = dict()
        if exists(adaptive_rmsnorm_cond):
            rmsnorm_kwargs = dict(cond = adaptive_rmsnorm_cond)

        # going through the attention layers

        for skip_combiner, attn_prenorm, attn, ff_prenorm, ff in self.layers:

            # in the paper, they use a u-net like skip connection
            # unclear how much this helps, as no ablations or further numbers given besides a brief one-two sentence mention

            if not exists(skip_combiner):
                skip_connects.append(x)
            else:
                skip_connect = skip_connects.pop() * self.skip_connect_scale
                x = torch.cat((x, skip_connect), dim = -1)
                x = skip_combiner(x)

            attn_input = attn_prenorm(x, **rmsnorm_kwargs)
            if key_padding_mask is not None:
                # Since we've Alibi_bias as a float-type attn_mask, the padding_mask need to be a float-type tensor, too
                float_key_padding_mask = key_padding_mask.float()
                float_key_padding_mask = float_key_padding_mask.masked_fill(key_padding_mask, float('-inf'))
            else:
                float_key_padding_mask = None 

            x, _ = attn(
                query=attn_input, 
                key=attn_input,
                value=attn_input,
                key_padding_mask = float_key_padding_mask,
                need_weights=False,
                attn_mask = alibi_bias
            )

            ff_input = ff_prenorm(x, **rmsnorm_kwargs) 
            x = ff(ff_input) + x

        return self.final_norm(x)

    def get_alibi_bias(self, batch_size: int, seq_len: int):
        bias = self.pos_bias[:, :seq_len, :seq_len]
        alibi_bias = bias.float() * self.slopes.float()
        alibi_bias = alibi_bias.repeat(batch_size, 1, 1)

        return alibi_bias

class TransformerUNet(NeuralModule):
    def __init__(
        self,
        in_channels = 1,
        out_channels = 1,
        freq_dim = 256,
        dim = 1024,
        depth = 24,
        heads = 16,
        ff_mult = 4,
        ff_dropout = 0.,
        attn_dropout = 0.,
        max_positions = 3000,
        time_hidden_dim = None,
        conv_pos_embed_kernel_size = 31,
        conv_pos_embed_groups = None,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        dim_in = freq_dim * in_channels * 2 

        time_hidden_dim = default(time_hidden_dim, dim * 4)
        self.proj_in = nn.Linear(dim_in, dim)

        self.sinu_pos_emb = nn.Sequential(
            LearnedSinusoidalPosEmb(dim),
            nn.Linear(dim, time_hidden_dim),
            nn.SiLU()
        )


        self.conv_embed = ConvPositionEmbed(
            dim = dim,
            kernel_size = conv_pos_embed_kernel_size,
            groups = conv_pos_embed_groups
        )

        self.transformer = Transformer(
            dim = dim,
            depth = depth,
            heads = heads,
            ff_mult = ff_mult,
            ff_dropout = ff_dropout,
            attn_dropout= attn_dropout,
            max_positions=max_positions,
            adaptive_rmsnorm = True,
            adaptive_rmsnorm_cond_dim_in = time_hidden_dim,
            use_unet_skip_connection = True,
        )

        dim_out = freq_dim * out_channels * 2

        self.proj_out = nn.Linear(dim, dim_out)

    @property
    def input_types(self) -> Dict[str, NeuralType]:
        """Returns definitions of module output ports.
        """
        return {
            "input": NeuralType(('B', 'C', 'D', 'T'), SpectrogramType()),
            "input_length": NeuralType(('B',), LengthsType(), optional=True),
            "condition": NeuralType(('B',), FloatType(), optional=True),
        }

    @property
    def output_types(self) -> Dict[str, NeuralType]:
        """Returns definitions of module output ports.
        """
        return {
            "output": NeuralType(('B', 'C', 'D', 'T'), SpectrogramType()),
            "output_length": NeuralType(('B',), LengthsType(), optional=True),
        }


    @staticmethod
    def _get_key_padding_mask(input_length, max_length):
        """
        Return the self_attention masking according to the input length.
        0 indicates the frame is in the valid range, while 1 indicates the frame is a padding frame.
        """
        key_padding_mask = torch.arange(max_length).expand(len(input_length), max_length).to(input_length.device)
        key_padding_mask = key_padding_mask >= input_length.unsqueeze(1)
        return key_padding_mask
    
    @typecheck()
    def forward(
        self,
        input,
        input_length=None,
        condition=None
    ):
        # Stack real and imaginary components
        B, C_in, D, T = input.shape
        if C_in != self.in_channels:
            raise RuntimeError(f'Unexpected input channel size {C_in}, expected {self.in_channels}')
        
        input_real_imag = torch.stack([input.real, input.imag], dim=2)
        input_real_imag = input_real_imag.permute(0, 4, 1, 2, 3)
        input = einops.rearrange(input_real_imag, 'B T C RI F -> B T (C RI F)')

        x = self.proj_in(input)
        key_padding_mask = self._get_key_padding_mask(input_length, max_length=T)
        x = self.conv_embed(x, mask = key_padding_mask) + x

        if not exists(condition):
            raise NotImplementedError
        
        time_emb = self.sinu_pos_emb(condition)

        # attend

        x = self.transformer(
            x,
            key_padding_mask = key_padding_mask,
            adaptive_rmsnorm_cond = time_emb
        )

        output = self.proj_out(x)
        output = output.reshape(B, T, self.out_channels, 2, D)
        output = output.permute(0, 2, 4, 1, 3)
        output = torch.view_as_complex(output.contiguous())

        return output, input_length