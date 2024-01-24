import torch
import torch.nn.functional as F

from einops import rearrange

from fftconv import fftconv_fwd, fftconv_bwd


def fftconv_ref(u, k, D, dropout_mask):
    seqlen = u.shape[-1]
    fft_size = 2 * seqlen
    k_f = torch.fft.rfft(k, n=fft_size) / fft_size
    u_f = torch.fft.rfft(u.to(dtype=k.dtype), n=fft_size)
    y = torch.fft.irfft(u_f * k_f, n=fft_size, norm='forward')[..., :seqlen]
    out = y + u * D.unsqueeze(-1)
    return (F.gelu(out) * rearrange(dropout_mask, 'b H -> b H 1')).to(dtype=u.dtype)


def fftconv_fast(u, k, D, dropout_mask):
    """Fuse padding + rfft + pointwise mult + ifft + multiply with D + gelu + dropout
    """
    seqlen = u.shape[-1]
    fft_size = 2 * seqlen
    k_f = torch.fft.rfft(k, n=fft_size)
    out = fftconv_fwd(u, k_f, D, dropout_mask, fft_size)
    return out


def fftconv_fast_bwd(dout, u, k, D, dropout_mask=None):
    seqlen = u.shape[-1]
    fft_size = 2 * seqlen
    k_f = torch.fft.rfft(k, n=fft_size)
    dx, dk_f, dD = fftconv_bwd(dout, u, k_f, D, dropout_mask, fft_size)
    dk = torch.fft.irfft(dk_f, n=fft_size, norm='forward')[..., :seqlen]
    return dx, dk, dD


device = 'cuda'
dtype = torch.float32
# dtype = torch.float16
batch_size = 64
H = 256
fft_size = 2048
seqlen = 1024
dropout_prob = 0.37

torch.manual_seed(0)
u = torch.randn(batch_size, H, seqlen, device=device, dtype=dtype, requires_grad=True)
k = torch.randn(H, seqlen, device=device, requires_grad=True)
D = torch.randn(H, device=device, requires_grad=True)
dropout_mask = F.dropout(torch.ones(batch_size, H, device=device), dropout_prob)

out = fftconv_ref(u, k, D, dropout_mask)
out = fftconv_fast(u, k, D, dropout_mask)
g = torch.randn_like(out)
fftconv_fast_bwd(g, u, k, D, dropout_mask)
