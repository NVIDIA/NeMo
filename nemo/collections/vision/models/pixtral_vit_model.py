import os
import torch
from torch import nn
import torch.nn.functional as F
from transformers import AutoConfig
from transformers.utils import ModelOutput
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple
from safetensors.torch import load_file

def remove_padding(padded_image, padding_val=-100):
    valid_rows = ~(padded_image == padding_val).all(dim=(1, 3)).squeeze(0)  # Shape: (H,)
    valid_cols = ~(padded_image == padding_val).all(dim=(1, 2)).squeeze(0)  # Shape: (W,)
    non_padded_image = padded_image[:, :, valid_rows, :][:, :, :, valid_cols]
    return non_padded_image

@dataclass
class BaseModelOutput(ModelOutput):
    """
    Base class for model's outputs, with potential hidden states and attentions.

    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    last_hidden_state: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None

def generate_block_attention_mask(patch_embeds_list, tensor):
    dtype = tensor.dtype
    device = tensor.device
    seq_len = tensor.shape[1]
    d_min = torch.finfo(dtype).min
    causal_mask = torch.full((seq_len, seq_len), fill_value=d_min, dtype=dtype, device=device)

    block_end_idx = torch.tensor(patch_embeds_list).cumsum(-1)
    block_start_idx = torch.tensor([0] + patch_embeds_list[:-1]).cumsum(-1)
    for start, end in zip(block_start_idx, block_end_idx):
        causal_mask[start:end, start:end] = 0

    causal_mask = causal_mask[None, None, :, :].expand(tensor.shape[0], 1, -1, -1)
    return causal_mask

class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        PixtralRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"

def _reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    freqs_cis: real - (seq_len, head_dim / 2)
    x: real - (bsz, seq_len, ..., head_dim / 2)
    """
    ndim = x.ndim
    assert ndim > 1
    assert freqs_cis.shape == (x.shape[1], x.shape[-1]), (
        "freqs_cis shape", freqs_cis.shape, "does not match required shape", (x.shape[1], x.shape[-1]),
    )
    shape = [1] * ndim
    shape[1] = x.shape[1]
    shape[-1] = x.shape[-1]
    return freqs_cis.view(*shape)

def precompute_freqs_cis_2d(
    dim: int,
    height: int,
    width: int,
    theta: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))

    h = torch.arange(height, device=freqs.device)
    w = torch.arange(width, device=freqs.device)

    freqs_h = torch.outer(h, freqs[::2]).float()
    freqs_w = torch.outer(w, freqs[1::2]).float()
    freqs_2d = torch.cat(
        [
            freqs_h[:, None, :].repeat(1, width, 1),
            freqs_w[None, :, :].repeat(height, 1, 1),
        ],
        dim=-1,
    )

    freqs_cis = torch.polar(torch.ones_like(freqs_2d), freqs_2d)
    freqs_cis_real = freqs_cis.real
    freqs_cis_imag = freqs_cis.imag
    return freqs_cis_real, freqs_cis_imag

def apply_rotary_emb_vit(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: Tuple[torch.Tensor, torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    # Reshape xq and xk to separate real and imaginary parts
    xq_reshaped = xq.reshape(*xq.shape[:-1], -1, 2)
    xk_reshaped = xk.reshape(*xk.shape[:-1], -1, 2)
    
    # Extract real and imaginary parts
    xq_real = xq_reshaped[..., 0]
    xq_imag = xq_reshaped[..., 1]
    xk_real = xk_reshaped[..., 0]
    xk_imag = xk_reshaped[..., 1]
    
    # Extract freqs_cis real and imaginary parts
    freqs_cis_real, freqs_cis_imag = freqs_cis
    #assert freqs_cis_real.shape == freqs_cis_imag.shape, "freqs_cis real and imaginary parts must have the same shape"
    
    # Reshape freqs_cis for broadcasting
    freqs_cis_real = _reshape_for_broadcast(freqs_cis_real, xq_real)
    freqs_cis_imag = _reshape_for_broadcast(freqs_cis_imag, xq_real)
    
    # Perform complex multiplication using real arithmetic
    xq_out_real = xq_real * freqs_cis_real - xq_imag * freqs_cis_imag
    xq_out_imag = xq_real * freqs_cis_imag + xq_imag * freqs_cis_real
    xk_out_real = xk_real * freqs_cis_real - xk_imag * freqs_cis_imag
    xk_out_imag = xk_real * freqs_cis_imag + xk_imag * freqs_cis_real
    
    # Stack real and imaginary parts back together
    xq_out = torch.stack([xq_out_real, xq_out_imag], dim=-1)
    xk_out = torch.stack([xk_out_real, xk_out_imag], dim=-1)
    
    # Flatten the last two dimensions to match original shape
    xq_out = xq_out.flatten(-2)
    xk_out = xk_out.flatten(-2)
    return xq_out.type_as(xq), xk_out.type_as(xk)

class FeedForward(nn.Module):

    def __init__(self, args):
        super().__init__()
        assert args.intermediate_size is not None
        self.w1 = nn.Linear(args.hidden_size,
                            args.intermediate_size,
                            bias=False)
        self.w2 = nn.Linear(args.intermediate_size,
                            args.hidden_size,
                            bias=False)
        self.w3 = nn.Linear(args.hidden_size,
                            args.intermediate_size,
                            bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

class Attention(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        assert not args.hidden_size % args.num_attention_heads
        self.n_heads = args.num_attention_heads
        self.head_dim = args.hidden_size // args.num_attention_heads
        self.scaling = (self.head_dim) ** -0.5
        self.wq = nn.Linear(args.hidden_size, args.hidden_size, bias=False)
        self.wk = nn.Linear(args.hidden_size, args.hidden_size, bias=False)
        self.wv = nn.Linear(args.hidden_size, args.hidden_size, bias=False)
        self.wo = nn.Linear(args.hidden_size, args.hidden_size, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        freqs_cis: torch.Tensor,
    ) -> torch.Tensor:
        batch, patches, _ = x.shape

        q, k, v = self.wq(x), self.wk(x), self.wv(x)

        q = q.reshape(batch, patches, self.n_heads, self.head_dim)
        k = k.reshape(batch, patches, self.n_heads, self.head_dim)
        v = v.reshape(batch, patches, self.n_heads, self.head_dim)

        q, k = apply_rotary_emb_vit(q, k, freqs_cis=freqs_cis)

        # Reshape to (batch * n_heads, patches, head_dim)
        q = q.permute(0, 2, 1, 3).reshape(batch * self.n_heads, patches, self.head_dim)
        k = k.permute(0, 2, 1, 3).reshape(batch * self.n_heads, patches, self.head_dim)
        v = v.permute(0, 2, 1, 3).reshape(batch * self.n_heads, patches, self.head_dim)

        # Native PyTorch
        out = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask.squeeze(0))
        out = out.reshape(batch, self.n_heads, patches, self.head_dim)
        out = out.permute(0, 2, 1, 3).reshape(batch, patches, self.n_heads * self.head_dim)
        
        return self.wo(out)


class TransformerBlock(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.attention = Attention(args)
        self.feed_forward = FeedForward(args)
        self.attention_norm = RMSNorm(args.hidden_size, eps=1e-5)
        self.ffn_norm = RMSNorm(args.hidden_size, eps=1e-5)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        freqs_cis: torch.Tensor,
    ) -> torch.Tensor:
        r = self.attention.forward(self.attention_norm(x),
                                   mask=mask,
                                   freqs_cis=freqs_cis)
        h = x + r
        r = self.feed_forward.forward(self.ffn_norm(h))
        out = h + r
        return out


class Transformer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.layers = torch.nn.ModuleList()
        for _ in range(args.num_hidden_layers):
            self.layers.append(TransformerBlock(args))

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        freqs_cis: Optional[torch.Tensor],
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, mask=mask, freqs_cis=freqs_cis)
        return x


def position_meshgrid(patch_embeds_list: list[torch.Tensor], ) -> torch.Tensor:
    positions = torch.cat([
        torch.stack(
            torch.meshgrid(
                torch.arange(p.shape[-2]),
                torch.arange(p.shape[-1]),
                indexing="ij",
            ),
            dim=-1,
        ).reshape(-1, 2) for p in patch_embeds_list
    ])
    return positions

class PixtralVisionModel(nn.Module):
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.patch_conv = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=config.hidden_size,
            kernel_size=config.patch_size,
            stride=config.patch_size,
            bias=False,
        )
        self.ln_pre = RMSNorm(config.hidden_size, eps=1e-5)
        self.transformer = Transformer(config)

        head_dim = self.config.hidden_size // self.config.num_attention_heads
        assert head_dim % 2 == 0, "ROPE requires even head_dim"
        self._freqs_cis: Optional[torch.Tensor] = None

    @property
    def max_patches_per_side(self) -> int:
        return self.config.image_size // self.config.patch_size

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @property
    def dtype(self) -> torch.device:
        return next(self.parameters()).dtype

    @property
    def freqs_cis(self) -> torch.Tensor:
        if self._freqs_cis is None:
            self._freqs_cis = precompute_freqs_cis_2d(
                dim=self.config.hidden_size // self.config.num_attention_heads,
                height=self.max_patches_per_side,
                width=self.max_patches_per_side,
                theta=self.config.rope_theta,
            )
    
        if self._freqs_cis[0].device != self.device:
            self._freqs_cis = [self._freqs_cis[0].to(device=self.device), self._freqs_cis[1].to(device=self.device)]
    
        return self._freqs_cis

    def forward(
        self,
        images: List[torch.Tensor],
        output_hidden_states: bool = True,
    ) -> torch.Tensor:
        """
        Args:
            images: a tensor of  of N images of variable sizes, 
                each of shape (N, C, H, W)
        Returns:
            image_features: tensor of token features for 
                all tokens of all images of shape (N_toks, D)
        """
        # pass images through initial convolution independently
        #patch_embeds_list = [
        #    self.patch_conv(img.unsqueeze(0).to(self.dtype)) for img in images
        #]
        patch_embeds_list = [
            self.patch_conv(remove_padding(images[:, i].to(self.dtype))) for i in range(images.shape[1])
        ]
        
        # flatten to a single sequence
        patch_embeds = torch.cat(
            [p.flatten(2).permute(0, 2, 1) for p in patch_embeds_list], dim=1)
        patch_embeds = self.ln_pre(patch_embeds)

        # positional embeddings
        positions = position_meshgrid(patch_embeds_list).to(self.device)
        freqs_cis = (self.freqs_cis[0][positions[:, 0], positions[:, 1]], self.freqs_cis[1][positions[:, 0], positions[:, 1]])

        # Attention mask for multiple images
        mask = generate_block_attention_mask([p.shape[-2] * p.shape[-1] for p in patch_embeds_list], patch_embeds)

        out = self.transformer(patch_embeds, mask=mask, freqs_cis=freqs_cis)

        # remove batch dimension of the single sequence
        return BaseModelOutput(last_hidden_state=out, hidden_states=[out], attentions=())
    
    @classmethod
    def from_pretrained(cls, model_name: str, torch_dtype: torch.dtype = torch.float16, freeze: bool = True) -> "PixtralVisionModel":
        # Get the config
        config = AutoConfig.from_pretrained(model_name)
        
        # Instantiate the model
        model = cls(config).to(torch_dtype)
        
        # Load the checkpoint
        safetensor_file = "model.safetensors"
        ckpt_file = os.path.join(model_name, safetensor_file)
        state_dict = load_file(ckpt_file)
        model.load_state_dict(state_dict)

        # Optionally freeze parameters
        if freeze:
            for param in model.parameters():
                param.requires_grad = False
            model = model.eval()
        
        return model