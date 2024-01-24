import math

import torch
import torch.nn.functional as F

from einops import rearrange

from fftconv import fftconv_fwd, fftconv_bwd

@torch.jit.script
def _mul_sum(y, q):
    return (y * q).sum(dim=1)

# reference convolution with residual connection
def fftconv_ref(u, k, D, dropout_mask, gelu=True, k_rev=None):
    seqlen = u.shape[-1]
    fft_size = 2 * seqlen
    k_f = torch.fft.rfft(k, n=fft_size) / fft_size
    if k_rev is not None:
        k_rev_f = torch.fft.rfft(k_rev, n=fft_size) / fft_size
        k_f = k_f + k_rev_f.conj()
    u_f = torch.fft.rfft(u.to(dtype=k.dtype), n=fft_size)
    y = torch.fft.irfft(u_f * k_f, n=fft_size, norm='forward')[..., :seqlen]
    out = y + u * D.unsqueeze(-1)
    if gelu:
        out = F.gelu(out)
    if dropout_mask is not None:
        return (out * rearrange(dropout_mask, 'b H -> b H 1')).to(dtype=u.dtype)
    else:
        return out.to(dtype=u.dtype)

# reference H3 forward pass
def fftconv_h3_ref(k, ssm_kernel, D, q, v, head_dim=1, ssm_kernel_rev=None):
    seqlen = k.shape[-1]
    fft_size = 2 * seqlen
    kv = (rearrange(k, 'b (h d1) l -> b d1 1 h l', d1=head_dim)
            * rearrange(v, 'b (h d2) l -> b 1 d2 h l', d2=head_dim))  # b d1 d2 h l
    kv_f = torch.fft.rfft(kv.to(dtype=ssm_kernel.dtype), n=fft_size) / fft_size
    ssm_kernel_f = torch.fft.rfft(ssm_kernel, n=fft_size)  # h L+1
    if ssm_kernel_rev is not None:
        ssm_kernel_rev_f = torch.fft.rfft(ssm_kernel_rev, n=fft_size)  # h L+1
        ssm_kernel_f = ssm_kernel_f + ssm_kernel_rev_f.conj()
    y = torch.fft.irfft(kv_f * ssm_kernel_f, n=fft_size, norm='forward')[..., :seqlen]  # b d1 d2 h l
    out = y + kv * D.unsqueeze(-1)  # b d1 d2 h l
    q = rearrange(q, 'b (h d1) l -> b d1 1 h l', d1=head_dim)
    if head_dim > 1:
        out = _mul_sum(out, q)
        return rearrange(out, 'b d2 h l -> b (h d2) l').to(dtype=k.dtype)
    else:
        return rearrange(out * q, 'b 1 1 h l -> b h l').to(dtype=k.dtype)


class FFTConvFunc(torch.autograd.Function):

    @staticmethod
    def forward(ctx, u, k, D, dropout_mask=None, gelu=True, force_fp16_output=False,
                output_hbl_layout=False, v=None, head_dim=1, q=None, fftfp16=False, k_rev=None):
        seqlen = u.shape[-1]
        fft_size = max(2 * 2 ** int(math.ceil(math.log2(seqlen))), 16)
        k_f = torch.fft.rfft(k, n=fft_size)
        if k_rev is not None:
            k_f = k_f + torch.fft.rfft(k_rev, n=fft_size).conj()
        if u.stride(-1) != 1:
            u = u.contiguous()
        k_f = k_f.contiguous()
        D = D.contiguous()
        if v is not None and v.stride(-1) != 1:
            v = v.contiguous()
        if q is not None and q.stride(-1) != 1:
            q = q.contiguous()
        if dropout_mask is not None:
            dropout_mask = dropout_mask.contiguous()
        ctx.save_for_backward(u, k_f, D, dropout_mask, v, q)
        ctx.output_hbl_layout = output_hbl_layout
        ctx.head_dim = head_dim
        ctx.gelu = gelu
        ctx.fftfp16 = fftfp16
        ctx.has_k_rev = k_rev is not None
        out = fftconv_fwd(u, k_f, D, v, head_dim, q, dropout_mask, gelu, False, False, fft_size, force_fp16_output, output_hbl_layout, fftfp16)
        return out

    @staticmethod
    def backward(ctx, dout):
        if ctx.output_hbl_layout:
            dout = rearrange(rearrange(dout, 'b h l -> h b l').contiguous(), 'h b l -> b h l')
        else:
            dout = dout.contiguous()
        u, k_f, D, dropout_mask, v, q = ctx.saved_tensors
        seqlen = u.shape[-1]
        fft_size = max(2 * 2 ** int(math.ceil(math.log2(seqlen))), 16)
        du, dk_f, dD, dv, dq = fftconv_bwd(dout, u, k_f, D, v, ctx.head_dim, q, dropout_mask, ctx.gelu, False, False, fft_size,
                                   ctx.output_hbl_layout, ctx.fftfp16)
        dk = torch.fft.irfft(dk_f, n=fft_size, norm='forward')[..., :seqlen]
        dk_rev = (None if not ctx.has_k_rev
                  else torch.fft.irfft(dk_f.conj(), n=fft_size, norm='forward')[..., :seqlen])
        if v is not None:
            dv = dv.to(dtype=v.dtype)  # We do atomicAdd in fp32 so might need to convert to fp16
        return du, dk, dD, None, None, None, None, dv if v is not None else None, None, dq if q is not None else None, None, dk_rev

def fftconv_func(u, k, D, dropout_mask=None, gelu=True, force_fp16_output=False,
                 output_hbl_layout=False, v=None, head_dim=1, q=None, fftfp16=False, k_rev=None):
    return FFTConvFunc.apply(u, k, D, dropout_mask, gelu, force_fp16_output,
                             output_hbl_layout, v, head_dim, q, fftfp16, k_rev)
