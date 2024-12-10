import math
from typing import Optional, Dict, Union
from random import random
from functools import partial, wraps
from packaging import version
from collections import namedtuple

import einops
import torch
from torch import nn, Tensor, einsum
from torch.nn import Module
import torch.nn.functional as F
from torch.cuda.amp import autocast

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

def generate_mask_from_repeats(repeats: Tensor):
    repeats = repeats.int()
    device = repeats.device

    lengths = repeats.sum(dim = -1)
    max_length = lengths.amax().item()
    cumsum = repeats.cumsum(dim = -1)
    cumsum_exclusive = F.pad(cumsum, (1, -1), value = 0.)

    seq = torch.arange(max_length, device = device)
    seq = einops.repeat(seq, '... j -> ... i j', i = repeats.shape[-1])

    cumsum = einops.rearrange(cumsum, '... i -> ... i 1')
    cumsum_exclusive = einops.rearrange(cumsum_exclusive, '... i -> ... i 1')

    lengths = einops.rearrange(lengths, 'b -> b 1 1')
    mask = (seq < cumsum) & (seq >= cumsum_exclusive) & (seq < lengths)
    return mask

def is_probably_audio_from_shape(t):
    return exists(t) and (t.ndim == 2 or (t.ndim == 3 and t.shape[1] == 1))


def maximum_path(value, mask, const=None):
    device = value.device
    dtype = value.dtype
    if not exists(const):
        const = torch.tensor(float('-inf')).to(device)  # Patch for Sphinx complaint
    value = value * mask

    b, t_x, t_y = value.shape
    direction = torch.zeros(value.shape, dtype=torch.int64, device=device)
    v = torch.zeros((b, t_x), dtype=torch.float32, device=device)
    x_range = torch.arange(t_x, dtype=torch.float32, device=device).view(1, -1)

    for j in range(t_y):
        v0 = pad_tensor(v, ((0, 0), (1, 0)), value = const)[:, :-1]
        v1 = v
        max_mask = v1 >= v0
        v_max = torch.where(max_mask, v1, v0)
        direction[:, :, j] = max_mask

        index_mask = x_range <= j
        v = torch.where(index_mask.view(1,-1), v_max + value[:, :, j], const)

    direction = torch.where(mask.bool(), direction, 1)

    path = torch.zeros(value.shape, dtype=torch.float32, device=device)
    index = mask[:, :, 0].sum(1).long() - 1
    index_range = torch.arange(b, device=device)

    for j in reversed(range(t_y)):
        path[index_range, index, j] = 1
        index = index + direction[index_range, index, j] - 1

    path = path * mask.float()
    path = path.to(dtype=dtype)
    return path

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

# rotary positional embeddings

class RotaryEmbedding(Module):
    def __init__(self, dim, theta = 50000):
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    @property
    def device(self):
        return self.inv_freq.device

    @autocast(enabled = False)
    def forward(self, t: Union[int, Tensor]):
        if not torch.is_tensor(t):
            t = torch.arange(t, device = self.device)

        t = t.type_as(self.inv_freq)
        freqs = torch.einsum('i , j -> i j', t, self.inv_freq)
        freqs = torch.cat((freqs, freqs), dim = -1)
        return freqs

def rotate_half(x):
    x1, x2 = x.chunk(2, dim = -1)
    return torch.cat((-x2, x1), dim = -1)

@autocast(enabled = False)
def apply_rotary_pos_emb(pos, t):
    return t * pos.cos() + rotate_half(t) * pos.sin()

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

# attention

class MultiheadRMSNorm(Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(heads, 1, dim))

    def forward(self, x):
        return F.normalize(x, dim = -1) * self.gamma * self.scale

class Attend(nn.Module):
    def __init__(
        self,
        dropout = 0.,
        flash = False,
        scale = None
    ):
        super().__init__()
        self.dropout = dropout
        self.attn_dropout = nn.Dropout(dropout)

        self.scale = scale

        self.flash = flash
        assert not (flash and version.parse(torch.__version__) < version.parse('2.0.0')), 'in order to use flash attention, you must be using pytorch 2.0 or above'

        # determine efficient attention configs for cuda and cpu

        self.cpu_config = FlashAttentionConfig(True, True, True)
        self.cuda_config = None

        if not torch.cuda.is_available() or not flash:
            return

        device_properties = torch.cuda.get_device_properties(torch.device('cuda'))

        if device_properties.major == 8 and device_properties.minor == 0:
            print_once('A100 GPU detected, using flash attention if input tensor is on cuda')
            self.cuda_config = FlashAttentionConfig(True, False, False)
        else:
            print_once('Non-A100 GPU detected, using math or mem efficient attention if input tensor is on cuda')
            self.cuda_config = FlashAttentionConfig(False, True, True)

    def flash_attn(self, q, k, v, mask = None):
        _, heads, q_len, dim_head, k_len, is_cuda, device = *q.shape, k.shape[-2], q.is_cuda, q.device

        # if scale is given, divide by the default scale that sdpa uses

        if exists(self.scale):
            q = q * (self.scale / (dim_head ** -0.5))

        # Check if mask exists and expand to compatible shape
        # The mask is B L, so it would have to be expanded to B H N L

        if exists(mask):
            mask = mask.expand(-1, heads, q_len, -1)

        # Check if there is a compatible device for flash attention

        config = self.cuda_config if is_cuda else self.cpu_config

        # pytorch 2.0 flash attn: q, k, v, mask, dropout, softmax_scale

        with torch.backends.cuda.sdp_kernel(**config._asdict()):
            out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask = mask,
                dropout_p = self.dropout if self.training else 0.
            )

        return out

    def forward(self, q, k, v, mask = None):
        """
        einstein notation
        b - batch
        h - heads
        n, i, j - sequence length (base sequence length, source, target)
        d - feature dimension
        """

        q_len, k_len, device = q.shape[-2], k.shape[-2], q.device

        scale = default(self.scale, q.shape[-1] ** -0.5)

        if exists(mask) and mask.ndim != 4:
            mask = einops.rearrange(mask, 'b j -> b 1 1 j')

        if self.flash:
            return self.flash_attn(q, k, v, mask = mask)

        # similarity

        sim = einsum(f"b h i d, b h j d -> b h i j", q, k) * scale

        # key padding mask

        if exists(mask):
            sim = sim.masked_fill(~mask, -torch.finfo(sim.dtype).max)

        # attention

        attn = sim.softmax(dim=-1)
        attn = self.attn_dropout(attn)

        # aggregate values

        out = einsum(f"b h i j, b h j d -> b h i d", attn, v)

        return out


class Attention(Module):
    def __init__(
        self,
        dim,
        dim_head = 64,
        heads = 8,
        dropout = 0,
        flash = False,
        qk_norm = False,
        qk_norm_scale = 10
    ):
        super().__init__()
        self.heads = heads
        dim_inner = dim_head * heads

        scale = qk_norm_scale if qk_norm else None

        self.attend = Attend(dropout, flash = flash, scale = scale)

        self.qk_norm = qk_norm

        if qk_norm:
            self.q_norm = MultiheadRMSNorm(dim_head, heads = heads)
            self.k_norm = MultiheadRMSNorm(dim_head, heads = heads)

        self.to_qkv = nn.Linear(dim, dim_inner * 3, bias = False)
        self.to_out = nn.Linear(dim_inner, dim, bias = False)

    def forward(self, x, mask = None, rotary_emb = None):
        h = self.heads

        q, k, v = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: einops.rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))

        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        if exists(rotary_emb):
            q, k = map(lambda t: apply_rotary_pos_emb(rotary_emb, t), (q, k))

        out = self.attend(q, k, v, mask = mask)

        out = einops.rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

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
        dim_head = 64,
        heads = 8,
        ff_mult = 4,
        attn_dropout = 0.,
        ff_dropout = 0.,
        num_register_tokens = 0.,
        attn_flash = False,
        adaptive_rmsnorm = False,
        adaptive_rmsnorm_cond_dim_in = None,
        use_unet_skip_connection = False,
        skip_connect_scale = None,
        attn_qk_norm = False,
        use_gateloop_layers = False,
        gateloop_use_jax = False,
    ):
        super().__init__()
        assert divisible_by(depth, 2)
        self.layers = nn.ModuleList([])

        self.rotary_emb = RotaryEmbedding(dim = dim_head)

        self.num_register_tokens = num_register_tokens
        self.has_register_tokens = num_register_tokens > 0

        if self.has_register_tokens:
            self.register_tokens = nn.Parameter(torch.randn(num_register_tokens, dim))

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
                GateLoop(dim = dim, use_jax_associative_scan = gateloop_use_jax) if use_gateloop_layers else None,
                rmsnorm_klass(dim = dim),
                Attention(dim = dim, dim_head = dim_head, heads = heads, dropout = attn_dropout, flash = attn_flash, qk_norm = attn_qk_norm),
                rmsnorm_klass(dim = dim),
                FeedForward(dim = dim, mult = ff_mult, dropout = ff_dropout)
            ]))

        self.final_norm = RMSNorm(dim)

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(
        self,
        x,
        mask = None,
        adaptive_rmsnorm_cond = None
    ):
        batch, seq_len, *_ = x.shape

        # add register tokens to the left

        if self.has_register_tokens:
            register_tokens = repeat(self.register_tokens, 'n d -> b n d', b = batch)

            x, ps = pack([register_tokens, x], 'b * d')

            if exists(mask):
                mask = F.pad(mask, (self.num_register_tokens, 0), value = True)

        # keep track of skip connections

        skip_connects = []

        # rotary embeddings

        positions = seq_len

        if self.has_register_tokens:
            main_positions = torch.arange(seq_len, device = self.device, dtype = torch.long)
            register_positions = torch.full((self.num_register_tokens,), -10000, device = self.device, dtype = torch.long)
            positions = torch.cat((register_positions, main_positions))

        rotary_emb = self.rotary_emb(positions)

        # adaptive rmsnorm

        rmsnorm_kwargs = dict()
        if exists(adaptive_rmsnorm_cond):
            rmsnorm_kwargs = dict(cond = adaptive_rmsnorm_cond)

        # going through the attention layers

        for skip_combiner, maybe_gateloop, attn_prenorm, attn, ff_prenorm, ff in self.layers:

            # in the paper, they use a u-net like skip connection
            # unclear how much this helps, as no ablations or further numbers given besides a brief one-two sentence mention

            if not exists(skip_combiner):
                skip_connects.append(x)
            else:
                skip_connect = skip_connects.pop() * self.skip_connect_scale
                x = torch.cat((x, skip_connect), dim = -1)
                x = skip_combiner(x)

            if exists(maybe_gateloop):
                x = maybe_gateloop(x) + x

            attn_input = attn_prenorm(x, **rmsnorm_kwargs)
            x = attn(attn_input, mask = mask, rotary_emb = rotary_emb) + x

            ff_input = ff_prenorm(x, **rmsnorm_kwargs) 
            x = ff(ff_input) + x

        # remove the register tokens

        if self.has_register_tokens:
            _, x = unpack(x, ps, 'b * d')

        return self.final_norm(x)

class _Transformer(Module):
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
        **kwargs,
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
            attn.train()

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

            attn_output, _ = attn(
                query=attn_input, 
                key=attn_input,
                value=attn_input,
                key_padding_mask = float_key_padding_mask,
                need_weights=False,
                attn_mask = alibi_bias
            )
            x = attn_output + x

            ff_input = ff_prenorm(x, **rmsnorm_kwargs) 
            x = ff(ff_input) + x

        return self.final_norm(x)

    def get_alibi_bias(self, batch_size: int, seq_len: int):
        bias = self.pos_bias[:, :seq_len, :seq_len]
        alibi_bias = bias.float() * self.slopes.float()
        alibi_bias = alibi_bias.repeat(batch_size, 1, 1)

        return alibi_bias
