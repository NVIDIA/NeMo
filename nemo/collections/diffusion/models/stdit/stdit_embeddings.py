import functools
import math
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from megatron.core.parallel_state import get_context_parallel_rank, get_context_parallel_world_size
from torch import nn

# reference : https://github.com/hpcaitech/Open-Sora/blob/main/opensora/models/layers/blocks.py

## gelu function
approx_gelu = lambda: nn.GELU(approximate="tanh")

# t-embedding part:
# ======================================================================
# --------------------------- TimeStepEmbeddding -----------------------
# ======================================================================


class TimestepEmbedder(nn.Module):
    """
    TimestepEmbedding : Embeds scalar timesteps into embedding representations
    with an optional random seed for syncronization.

    Args:
        hidden_size (int): Dimension of mlp embedding.
        frequency_embedding_size (int):  Dimension of timestep embedding.
        seed (int, optional): Random seed for initializing the embedding layers.
                              If None, no specific seed is set.
    """

    def __init__(self, hidden_size, frequency_embedding_size=256, seed=None):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

        if seed is not None:
            with torch.random.fork_rng():
                torch.manual_seed(seed)
                self.mlp[0].reset_parameters()
                self.mlp[2].reset_parameters()

        setattr(self.mlp[0].weight, "pipeline_parallel", True)
        setattr(self.mlp[0].bias, "pipeline_parallel", True)
        setattr(self.mlp[2].weight, "pipeline_parallel", True)
        setattr(self.mlp[2].bias, "pipeline_parallel", True)

    @staticmethod
    def timestep_embedding(t: torch.Tensor, dim, max_period=10000) -> torch.Tensor:
        """
        Create sinusoidal timestep embeddings.

        Args:
            t (Tensor): a 1-D Tensor of N indices, one per batch element with shape (N)
                        These may be fractional.
            dim (int): the dimension of the output.
            max_period (int): controls the minimum frequency of the embeddings.

        Returns:
            torch.Tensor: Tensor of positional embeddings with shape (N, dim)
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32, device=torch.cuda.current_device())
            / half
        )
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t: torch.Tensor, dtype) -> torch.Tensor:
        """
        For timesteps: timestep_embedding + mlp_embedding
        Args:
            t(torch.Tensor): Input tensor of shape (B)
        Returns:
            torch.Tensor: Output tensor of shape (B, D)
        """
        # timestep_embedding
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        if t_freq.dtype != dtype:
            t_freq = t_freq.to(dtype)

        # mlp_embedding
        t_emb = self.mlp(t_freq)
        return t_emb


# ======================================================================
# --------------------------- SizeEmbedding ----------------------------
# ======================================================================


class SizeEmbedder(TimestepEmbedder):
    """
    SizeEmbedder is a subclass of TimestepEmbedder. Embeds scalar fps into embedding representations
    with an optional random seed for syncronization.

    Args:
        hidden_size (int): Dimension of mlp embedding.
        frequency_embedding_size (int):  Dimension of timestep embedding.
        seed (int, optional): Random seed for initializing the embedding layers.
                              If None, no specific seed is set.
    """

    def __init__(self, hidden_size, frequency_embedding_size=256, seed=None):
        super().__init__(hidden_size=hidden_size, frequency_embedding_size=frequency_embedding_size)
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size
        self.outdim = hidden_size

        if seed is not None:
            with torch.random.fork_rng():
                torch.manual_seed(seed)
                self.mlp[0].reset_parameters()
                self.mlp[2].reset_parameters()

        setattr(self.mlp[0].weight, "pipeline_parallel", True)
        setattr(self.mlp[0].bias, "pipeline_parallel", True)
        setattr(self.mlp[2].weight, "pipeline_parallel", True)
        setattr(self.mlp[2].bias, "pipeline_parallel", True)

    def forward(self, fps: torch.Tensor, batch_size) -> torch.Tensor:
        """
        fps_embedding part: timestep_embedding & mlp_embedding
        Args:
            - fps(torch.Tensor) : The input tensor(fps tensor) of shape(B, 1)
            - batch_size :  The number of batch size
        Return
            - fps_emb(torch.Tensor) : The out tenosr of shape (B, D)
        """

        if fps.ndim == 1:
            fps = fps[:, None]
        assert fps.ndim == 2
        if fps.shape[0] != batch_size:
            fps = fps.repeat(batch_size // fps.shape[0], 1)
            assert fps.shape[0] == batch_size

        batch, dim = fps.shape[0], fps.shape[1]
        fps = rearrange(fps, "b d -> (b d)")
        fps_freq = self.timestep_embedding(fps, self.frequency_embedding_size).to(self.dtype)
        fps_emb = self.mlp(fps_freq)
        fps_emb = rearrange(fps_emb, "(b d) d2 -> b (d d2)", b=batch, d=dim, d2=self.outdim)
        return fps_emb

    @property
    def dtype(self):
        return next(self.parameters()).dtype


# ======================================================================
# --------------------------- TblockEmbedding --------------------------
# ======================================================================


class TblockEmbedder(nn.Module):
    """
    Tblock embedder: embedding part for (timestep_embedding + fps_embedding)

    Args:
        hidden_size: the dimension of the input.
        chunk_size: (chunk_size * hidden_size) is the dimension of the output
        seed (int, optional): Random seed for initializing the embedding layers.
                              If None, no specific seed is set.
    """

    def __init__(self, hidden_size, chunk_size=6, seed=None):
        super().__init__()
        self.t_proj = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, chunk_size * hidden_size, bias=True))

        if seed is not None:
            with torch.random.fork_rng():
                torch.manual_seed(seed)
                self.t_proj[1].reset_parameters()

        setattr(self.t_proj[1].weight, "pipeline_parallel", True)
        setattr(self.t_proj[1].bias, "pipeline_parallel", True)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        embedding part for (timestep_embedding + fps_embedding)
        Args:
            t(torch.Tensor): The input tensor of shape(B, D)
        Return:
            torch.Tensor: The input tensor of shape(B, chunk_size * D)
        """
        t_emb = self.t_proj(t)
        return t_emb


# x-embedding part:
# ======================================================================
# --------------------------- PosEmbedding2d --------------------------
# ======================================================================


class PositionEmbedding2D(nn.Module):
    """
    Position Embedding in spatial dimension.
    """

    def __init__(self, dim: int) -> None:
        """
        Args:
            dim(int): the hidden state dimension size
        """
        super().__init__()
        self.dim = dim
        assert dim % 4 == 0, "dim must be divisible by 4"
        half_dim = dim // 2
        inv_freq = 1.0 / (10000 ** (torch.arange(0, half_dim, 2).float() / half_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def _get_sin_cos_emb(self, t: torch.Tensor) -> torch.Tensor:
        out = torch.einsum("i,d->id", t, self.inv_freq)
        emb_cos = torch.cos(out)
        emb_sin = torch.sin(out)
        return torch.cat((emb_sin, emb_cos), dim=-1)

    @functools.lru_cache(maxsize=512)
    def _get_cached_emb(
        self,
        device: torch.device,
        dtype: torch.dtype,
        h: int,
        w: int,
        scale: float = 1.0,
        base_size: Optional[int] = None,
    ):
        """
        Args:
            device: torch tensor in which device
            dtype: torch tensor dtype
            h(int): the original height dimension in video frame
            w(int): the original weight dimension in video frame
            scale(float): scale ratio in spatial dimension part
            base_size(int, optional): sqrt of spatial dimension
        Return:
            torch tensor with original dimension with shape(1, h * w, D)
        """

        grid_h = torch.arange(h, device=torch.cuda.current_device()) / scale
        grid_w = torch.arange(w, device=torch.cuda.current_device()) / scale
        if base_size is not None:
            grid_h *= base_size / h
            grid_w *= base_size / w
        grid_h, grid_w = torch.meshgrid(
            grid_w,
            grid_h,
            indexing="ij",
        )  # here w goes first
        grid_h = grid_h.t().reshape(-1)
        grid_w = grid_w.t().reshape(-1)
        emb_h = self._get_sin_cos_emb(grid_h)
        emb_w = self._get_sin_cos_emb(grid_w)
        return torch.concat([emb_h, emb_w], dim=-1).unsqueeze(0).to(dtype)

    def pos_select_in_cp_rank(self, pos_origin: torch.Tensor, h: int, w: int) -> torch.Tensor:
        """
        Position embedding select in cp rank due to latent split in cp group
        Args:
            pos_origin(torch.Tensor): input tensor with shape(1, h * w, D)
        Return:
            tensor with shape(1, h / cp_size * w, D)
        """
        cp_size = get_context_parallel_world_size()
        cp_rank = get_context_parallel_rank()
        if cp_size == 1:
            return pos_origin
        # split in h dimension in latent
        assert h % cp_size == 0
        pos_emb_per_cp_rank = torch.chunk(pos_origin, cp_size, dim=1)[cp_rank].contiguous()
        return pos_emb_per_cp_rank

    def forward(
        self,
        x: torch.Tensor,
        h: int,
        w: int,
        scale: Optional[float] = 1.0,
        base_size: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: input tensor with shape(B, T, S, D). Here just for tensor device & dtype
        """
        pos_emb = self._get_cached_emb(x.device, x.dtype, h, w, scale, base_size)
        pos_emb = self.pos_select_in_cp_rank(pos_emb, h, w)
        return pos_emb


# ======================================================================
# ----------------------------- PatchEmbed3D ---------------------------
# ======================================================================


class PatchEmbed3D(nn.Module):
    """Video latent to Patch Embedding part:
        - patchify embedding
        - with padding, conv3d, norm, flatten

    Args:
        - patch_size (int): Patch token size. Default: (1, 2, 2).
        - in_chans (int): Number of input video channels. Default: 3.
        - embed_dim (int): Number of linear projection output channels. Default: 96.
        - norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(
        self,
        patch_size=(1, 2, 2),
        in_chans=3,
        embed_dim=96,
        norm_layer=None,
        flatten=True,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.flatten = flatten
        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        # due to padding = 0, change nn.conv2d to nn.linear
        # self.proj = nn.Linear(in_chans * np.prod(patch_size), embed_dim)

        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        """
            Forward function of the PatchEmbed3D module.
        Parameters:
            x(torch.Tensor): The input tensor of shape(B, C, T, H, W)

        - Returns:
            torch.Tensor: The embedded patches as a tensor of shape(B, S, D)

        """
        # padding_set
        _, _, T, H, W = x.size()
        if W % self.patch_size[2] != 0:
            x = F.pad(x, (0, self.patch_size[2] - W % self.patch_size[2]))
        if H % self.patch_size[1] != 0:
            x = F.pad(x, (0, 0, 0, self.patch_size[1] - H % self.patch_size[1]))
        if T % self.patch_size[0] != 0:
            x = F.pad(x, (0, 0, 0, 0, 0, self.patch_size[0] - T % self.patch_size[0]))

        # proj
        x = self.proj(x)  # [B, C, T, H, W]

        # norm
        if self.norm is not None:
            T, Wh, Ww = x.size(2), x.size(3), x.size(4)
            x = x.flatten(2).transpose(1, 2)
            x = self.norm(x)
            x = x.transpose(1, 2).view(-1, self.embed_dim, T, Wh, Ww)

        # format reshape to [B, S, D]
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # [B, D, T, H, W] -> [B, S, D]
        return x


# y-embedding part
# ======================================================================
# --------------------------- CaptionEmbedder --------------------------
# ======================================================================
class CaptionEmbedder(nn.Module):
    """
    CaptionEmbedder part similar to LabelEmbedder.
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """

    def __init__(self, in_channels, hidden_size, uncond_prob, act_layer=approx_gelu, token_num=120, seed=None):
        super().__init__()
        self.y_proj = nn.Sequential(
            nn.Linear(in_channels, hidden_size, bias=True),
            act_layer(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.uncond_prob = uncond_prob
        self.seed = seed
        if seed is not None:
            with torch.random.fork_rng():
                torch.manual_seed(seed)
                self.y_proj[0].reset_parameters()
                self.y_proj[2].reset_parameters()
                self.register_buffer(
                    "y_embedding",
                    torch.randn(token_num, in_channels) / in_channels**0.5,
                )
        else:
            self.register_buffer(
                "y_embedding",
                torch.randn(token_num, in_channels) / in_channels**0.5,
            )

        setattr(self.y_proj[0].weight, "pipeline_parallel", True)
        setattr(self.y_proj[0].bias, "pipeline_parallel", True)
        setattr(self.y_proj[2].weight, "pipeline_parallel", True)
        setattr(self.y_proj[2].bias, "pipeline_parallel", True)

    def token_drop(self, caption, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            with torch.random.fork_rng():
                torch.manual_seed(self.seed)
                drop_ids = torch.rand(caption.shape[0]).cuda() < self.uncond_prob
        else:
            drop_ids = force_drop_ids == 1
        caption = torch.where(drop_ids[:, None, None, None], self.y_embedding, caption)
        return caption

    def forward(self, caption, train, force_drop_ids=None):
        if train:
            assert caption.shape[2:] == self.y_embedding.shape
        use_dropout = self.uncond_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            caption = self.token_drop(caption, force_drop_ids)
        caption = self.y_proj(caption)
        return caption


# ======================================================================
# ------------------------------ Final layer ---------------------------
# ======================================================================


def t2i_modulate(x, shift, scale):
    return x * (1 + scale) + shift


class T2IFinalLayer(nn.Module):
    """
    The final layer of STDiT.

    Args:
        hidden_size(int): the dimension of input tensor
        num_patch(int): the product of (patch_temporal * patch_height * patch_width)
        out_channels(int): output_channels
        d_t(int): temporal dimension in per rank
        d_s(int): spatial dimension in per rank
    """

    def __init__(self, hidden_size, num_patch, out_channels, d_t=None, d_s=None):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, num_patch * out_channels, bias=True)
        self.scale_shift_table = nn.Parameter(torch.randn(2, hidden_size) / hidden_size**0.5)
        self.out_channels = out_channels
        self.d_t = d_t
        self.d_s = d_s

    def t_mask_select(self, x_mask, x, masked_x, T, S):
        """
        t_mask_select which is speical in open-sora stditv3 model

        Args:
            x_mask(torch.Tensor): the input tensor mask with shape (B, T), true or false
            x(torch.Tensor): the input tensor from decoder with shape (B, (T, S), D)
            masked_x(torch.Tensor): the input tensor with mask which shape (B, (T, S), D)
            T(int): temporal dimension
            S(int): spatial dimension

        """
        x = rearrange(x, "B (T S) D -> B T S D", T=T, S=S)
        masked_x = rearrange(masked_x, "B (T S) D -> B T S D", T=T, S=S)
        x = torch.where(x_mask[:, :, None, None], x, masked_x)
        x = rearrange(x, "B T S D -> B (T S) D")
        return x

    def forward(self, x, t, x_mask=None, t0=None, T=None, S=None):
        """
        x(torch.Tensor): the input tensor from decoder
        t(torch.Tensor): timestep tensor with shape(B, D)
        x_mask(torch.Tensor, optional): the input tensor mask for mask some frames with shape
        t0(torch.Tensor, optional): t with timestep=0, timestep tensor with shape(B, D)
        T(int, optional): temporal dimension
        S(int, optional): spatial dimension
        """
        if T is None:
            T = self.d_t
        if S is None:
            S = self.d_s
        # modulate part
        shift, scale = (self.scale_shift_table[None] + t[:, None]).chunk(2, dim=1)
        x = t2i_modulate(self.norm_final(x), shift, scale)

        # x_mask part
        if x_mask is not None:
            shift_zero, scale_zero = (self.scale_shift_table[None] + t0[:, None]).chunk(2, dim=1)
            x_zero = t2i_modulate(self.norm_final(x), shift_zero, scale_zero)
            x = self.t_mask_select(x_mask, x, x_zero, T, S)

        # linear part
        x = self.linear(x)
        return x
