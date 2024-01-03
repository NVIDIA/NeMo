from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor

from nemo.collections.tts.losses.audio_codec_loss import MaskedCosineLoss
from nemo.collections.tts.parts.utils.distributed import broadcast_tensors
from nemo.collections.tts.parts.utils.helpers import mask_sequence_tensor
from nemo.core.classes.common import typecheck
from nemo.core.classes.module import NeuralModule
from nemo.core.neural_types.elements import EncodedRepresentation, Index, LengthsType, LossType
from nemo.core.neural_types.neural_type import NeuralType
from nemo.utils import logging
from nemo.utils.decorators import experimental


@experimental
class FiniteScalarQuantizer(NeuralModule):
    """
    Args:
        num_levels: number of levels for each dimension
        eps: small regularization constant for scaling

    References:
        Mentzer et al., Finite Scalar Quantization: VQ-VAE Made Simple (https://arxiv.org/abs/2309.15505v1)
    """

    def __init__(self, num_levels: List[int], eps: float = 1e-3):
        super().__init__()

        # index base per dimension of the input vector
        # this is used to convert between per-dimension indices and a codebook token index
        dim_base_index = torch.cumprod(torch.tensor([1] + num_levels[:-1]), dim=0, dtype=torch.int32)
        dim_base_index = rearrange(dim_base_index, 'D -> 1 D 1')
        self.register_buffer('dim_base_index', dim_base_index)

        # Register the number of levels for each dimension
        num_levels = torch.tensor(num_levels, dtype=torch.int32)
        num_levels = rearrange(num_levels, 'D -> 1 D 1')
        self.register_buffer('num_levels', num_levels)

        # Regularization
        self.eps = eps

        logging.debug('Initializing %s with', self.__class__.__name__)
        logging.debug('\t dim:           %s', self.dim)
        logging.debug('\t num_levels:    %s', self.num_levels)
        logging.debug('\t codebook_size: %s', self.codebook_size)
        logging.debug('\t eps:           %s', self.eps)

    @property
    def codebook_size(self):
        """Returns the size of the corresponding codebook."""
        return self.num_levels.prod().item()

    @property
    def dim(self):
        """Returns the dimension of the input vector."""
        return self.num_levels.numel()

    @property
    def codebook_dim(self):
        """Returns the dimension of the input vector.
        Keeping for compatiblitiy with the original RVQ implementation.
        """
        return self.dim

    @property
    def codes(self):
        """Returns the codebooks entries."""
        indices = torch.arange(self.codebook_size)
        # [D, B, T]
        indices = rearrange(indices, 'B -> 1 B 1')
        # [B, D, T]
        codes = self.decode(indices=indices, input_len=None)
        # Remove the time dimension
        codes = codes.squeeze(-1)
        return codes

    @property
    def codebook(self):
        """Returns the codebooks entries."""
        return self.codes

    @staticmethod
    def round(inputs: torch.Tensor, input_len: torch.Tensor) -> torch.Tensor:
        """Round with a straight-through estimator.
        """
        inputs_rounded = torch.round(inputs)
        return inputs + (inputs_rounded - inputs).detach()

    def compress(self, inputs: torch.Tensor, input_len: torch.Tensor) -> torch.Tensor:
        """Apply compression to the input, to limit to values.
        """
        output_scale = (self.num_levels - 1) / 2
        # scale down a bit to avoid rounding issues
        output_scale = output_scale * (1 - self.eps)
        # offset for even number of levels
        output_offset = torch.where(self.num_levels % 2 == 0, 0.5, 0)
        # shift for even number of levels
        input_shift = (output_offset / output_scale).tan()
        # compressed output
        output = output_scale * (inputs + input_shift).tanh() - output_offset
        return output

    @typecheck(
        input_types={
            "inputs": NeuralType(('B', 'D', 'T'), EncodedRepresentation()),
            "input_len": NeuralType(tuple('B'), LengthsType()),
        },
        output_types={"codes": NeuralType(('B', 'D', 'T'), Index())},
    )
    def inputs_to_codes(self, inputs: torch.Tensor, input_len: torch.Tensor) -> torch.Tensor:
        # apply compression
        compressed = self.compress(inputs=inputs, input_len=input_len)
        # apply rounding to nearest integer
        codes = self.round(inputs=compressed, input_len=input_len)
        # normalize to [-1, 1]
        scale = self.num_levels // 2
        codes = codes / scale
        return codes

    def codes_to_nonnegative(self, codes: torch.Tensor) -> torch.Tensor:
        """Convert values centered arouund zero to nonnegative values.
        """
        scale = offset = self.num_levels // 2
        return scale * codes + offset

    def nonnegative_to_codes(self, codes_nonnegative: torch.Tensor) -> torch.Tensor:
        """Convert nonnegative values to values centered arouund zero.
        """
        scale = offset = self.num_levels // 2
        return (codes_nonnegative - offset) / scale

    def codes_to_indices(self, codes: torch.Tensor) -> torch.Tensor:
        """Converts a code vector to a single index.
        """
        if codes.size(1) != self.dim:
            raise RuntimeError(
                f'Input code dimension {codes.size(1)} not matching the expected dimension {self.dim}, input codes shape {codes.shape}'
            )
        # convert code vectors to nonnegative values
        indices = self.codes_to_nonnegative(codes)
        # convert one nonnegative index per dimension to a single index per code vector
        indices = torch.sum(indices * self.dim_base_index, dim=1)
        return indices.to(torch.int32)

    # API of the RVQ
    @property
    def input_types(self):
        return {
            "inputs": NeuralType(('B', 'D', 'T'), EncodedRepresentation()),
            "input_len": NeuralType(tuple('B'), LengthsType(), optional=True),
        }

    @property
    def output_types(self):
        return {
            "dequantized": NeuralType(('B', 'D', 'T'), EncodedRepresentation()),
            "indices": NeuralType(('D', 'B', 'T'), Index()),
        }

    @typecheck()
    def forward(
        self, inputs: torch.Tensor, input_len: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        if inputs.size(1) != self.dim:
            raise RuntimeError(
                f'Input dimension {inputs.size(1)} not matching the expected dimension {self.dim}, inputs shape {inputs.shape}'
            )

        dequantized = self.inputs_to_codes(inputs=inputs, input_len=input_len)
        indices = self.codes_to_indices(codes=dequantized)

        if input_len is not None:
            # apply masking
            dequantized = mask_sequence_tensor(dequantized, input_len)
            indices = mask_sequence_tensor(indices, input_len)

        # only 1 codebook, but return in [D, B, T] format to match RVQ API
        indices = indices.unsqueeze(0)
        return dequantized, indices

    @typecheck(
        input_types={
            "inputs": NeuralType(('B', 'D', 'T'), EncodedRepresentation()),
            "input_len": NeuralType(tuple('B'), LengthsType(), optional=True),
        },
        output_types={"indices": NeuralType(('D', 'B', 'T'), Index())},
    )
    def encode(self, inputs: torch.Tensor, input_len: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Convert a continuous code vector to a single index.
        """
        _, indices = self(inputs=inputs, input_len=input_len)
        return indices

    @typecheck(
        input_types={
            "indices": NeuralType(('D', 'B', 'T'), Index()),
            "input_len": NeuralType(tuple('B'), LengthsType(), optional=True),
        },
        output_types={"dequantized": NeuralType(('B', 'D', 'T'), EncodedRepresentation()),},
    )
    def decode(self, indices: torch.Tensor, input_len: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Convert a single index to a continuous code vector.
        """
        if indices.size(0) > 1:
            # codebook dimension used for compatibility with RVQ
            raise ValueError(
                f'Expected a single codebook, got {indices.size(0)} codebooks for indices with shape {indices.shape}.'
            )

        indices = rearrange(indices, 'D B T -> B D T')
        # convert a single index to nonnegative index per-dimension
        codes_nonnegative = (indices // self.dim_base_index) % self.num_levels
        # convert nonnegative codes to codes (centered around zero)
        dequantized = self.nonnegative_to_codes(codes_nonnegative)

        if input_len is not None:
            # apply masking
            dequantized = mask_sequence_tensor(dequantized, input_len)
        return dequantized


class GroupFiniteScalarQuantizer(NeuralModule):
    """
    Grouped Finite Scalar Quantizer

    Args:
        num_quantizers: Number of codebooks to use.
        num_levels: Codebook leels
    """

    def __init__(
        self,
        num_groups: int,
        num_levels: List[int]
    ):
        super().__init__()
        self.num_groups = num_groups
        self.fsq_list = nn.ModuleList([FiniteScalarQuantizer(num_levels=num_levels) for _ in range(num_groups)])

    @typecheck(
        input_types={
            "inputs": NeuralType(('B', 'D', 'T'), EncodedRepresentation()),
            "input_len": NeuralType(tuple('B'), LengthsType()),
        },
        output_types={"preprocessed": NeuralType(('B', 'D', 'T'), EncodedRepresentation())},
    )
    def preprocess_input(self, inputs, input_len: Tensor):
        return inputs

    @property
    def input_types(self):
        return {
            "inputs": NeuralType(('B', 'D', 'T'), EncodedRepresentation()),
            "input_len": NeuralType(tuple('B'), LengthsType()),
        }

    @property
    def output_types(self):
        return {
            "dequantized": NeuralType(('B', 'D', 'T'), EncodedRepresentation()),
            "indices": NeuralType(('D', 'B', 'T'), Index()),
            "commit_loss": NeuralType((), LossType()),
            "codebook_loss": NeuralType((), LossType())
        }

    @typecheck()
    def forward(self, inputs: Tensor, input_len: Tensor) -> Tuple[Tensor, Tensor, float, float]:
        input_groups = inputs.chunk(self.num_groups, dim=1)
        index_list = []
        dequantized_list = []
        for in_group, fsq in zip(input_groups, self.fsq_list):
            dequantized_i, indices_i = fsq(inputs=in_group, input_len=input_len)
            index_list.append(indices_i)
            dequantized_list.append(dequantized_i)

        # [C, B, T]
        indices = torch.cat(index_list, dim=0)
        # [B, D, T]
        dequantized = torch.cat(dequantized_list, dim=1)
        commit_loss, codebook_loss = 0.0, 0.0
        return dequantized, indices, commit_loss, codebook_loss

    @typecheck(
        input_types={
            "inputs": NeuralType(('B', 'D', 'T'), EncodedRepresentation()),
            "input_len": NeuralType(tuple('B'), LengthsType()),
        },
        output_types={"indices": NeuralType(('D', 'B', 'T'), Index())},
    )
    def encode(self, inputs: Tensor, input_len: Tensor) -> Tensor:
        """Input is split into groups, each group is encoded separately, then the results are concatenated.
        """
        input_groups = inputs.chunk(self.num_groups, dim=1)
        indices = []
        for in_group, fsq in zip(input_groups, self.fsq_list):
            indices_group = fsq.encode(inputs=in_group, input_len=input_len)
            indices.append(indices_group)

        # concatenate along the codebook dimension
        indices = torch.cat(indices, dim=0)

        return indices

    @typecheck(
        input_types={
            "indices": NeuralType(('D', 'B', 'T'), Index()),
            "input_len": NeuralType(tuple('B'), LengthsType()),
        },
        output_types={"dequantized": NeuralType(('B', 'D', 'T'), EncodedRepresentation()),},
    )
    def decode(self, indices: Tensor, input_len: Tensor) -> Tensor:
        """Input indices are split into groups, each group is decoded separately, then the results are concatenated.
        """
        index_groups = indices.chunk(self.num_groups, dim=0)
        dequantized = []

        for index_group, fsq in zip(index_groups, self.fsq_list):
            dequantized_group = fsq.decode(indices=index_group, input_len=input_len)
            dequantized.append(dequantized_group)

        # concatenate along the feature dimension
        dequantized = torch.cat(dequantized, dim=1)

        return dequantized


def l2_norm(inputs):
    return F.normalize(inputs, dim=-1)


def compute_l2_distances(input1: Tensor, input2: Tensor) -> Tensor:
    """
    Compute pairwise L2 distance between two input tensors

    Args:
        input1: [B, D] first tensor.
        input2: [N, D] second tensor.

    Returns:
        [B, N] tensor of distances.
    """
    input2 = rearrange(input2, "N D -> D N")
    distances = input1.pow(2).sum(1, keepdim=True) - (2 * input1 @ input2) + input2.pow(2).sum(0, keepdim=True)
    return distances


def compute_cosine_distances(input1: Tensor, input2: Tensor) -> Tensor:
    """
    Compute pairwise cosine distance between two input tensors

    Args:
        input1: [B, D] first tensor.
        input2: [N, D] second tensor.

    Returns:
        [B, N] tensor of cosine distances.
    """
    # Cosine distance is proportional to the l2 distance between normalized vectors.
    input1_norm = F.normalize(input1, dim=1)
    input2_norm = F.normalize(input2, dim=1)
    cosine_dist = compute_l2_distances(input1_norm, input2_norm)
    return cosine_dist


def sample_vectors(samples: Tensor, num_sample: int) -> Tensor:
    """
    Randomly sample from the input batch.

    Args:
        samples: [B, D] tensor with features to sample.
        num_sample: Number of samples to draw.
            If the value is less than or equal to B, then the samples will be unique.
            If the value is greater than B, then samples will be drawn with replacement.

    Returns:
        Tensor with num_sample values randomly sampled from the input batch.
    """
    device = samples.device
    total_samples = samples.shape[0]

    if total_samples >= num_sample:
        indices = torch.randperm(total_samples, device=device)[:num_sample]
    else:
        indices = torch.randint(low=0, high=total_samples, size=(num_sample,), device=device)

    return samples[indices]


@experimental
class CosineCodebook(NeuralModule):
    """
    Single codebook.

    Args:
        codebook_size: Number of codes to use.
        codebook_dim: Dimension of each code.
    """

    def __init__(
        self,
        codebook_size: int,
        codebook_dim: int,
        code_decay: Optional[float] = None,
        code_ema_threshold: Optional[float] = None,
        norm_type: str = None,
    ):
        super().__init__()

        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim
        self.code_decay = code_decay
        self.code_ema_threshold = code_ema_threshold
        self.commit_loss_fn = MaskedCosineLoss()
        self.codebook_loss_fn = MaskedCosineLoss()

        if norm_type is None:
            self.norm = None
        elif norm_type == "l2":
            self.norm = l2_norm
        elif norm_type == "tanh":
            self.norm = F.tanh
        elif norm_type == "layer_norm":
            self.norm = torch.nn.LayerNorm(self.codebook_dim, elementwise_affine=False)
        else:
            raise ValueError(f"Unknown norm type {norm_type}")

        if norm_type == "tanh":
            weight = torch.FloatTensor(codebook_size, codebook_dim).uniform_(-1.0, 1.0)
            self.codebook = nn.Embedding(codebook_size, codebook_dim, _weight=weight)
        else:
            self.codebook = nn.Embedding(codebook_size, codebook_dim)

        if self.code_ema_threshold:
            ema_init = (1 + self.code_ema_threshold) * torch.ones(codebook_size) + torch.rand(codebook_size)
            self.register_buffer("ema", ema_init)

    def _compute_codebook_losses(self, inputs, codes, input_len):
        commit_loss = self.commit_loss_fn(
            predicted=inputs,
            target=codes.detach(),
            target_len=input_len,
        )

        codebook_loss = self.codebook_loss_fn(
            predicted=codes,
            target=inputs.detach(),
            target_len=input_len,
        )
        return commit_loss, codebook_loss

    def _expire_codes(self, inputs: Tensor) -> None:
        is_expired = self.ema < self.code_ema_threshold
        if not torch.any(is_expired):
            return

        samples = sample_vectors(samples=inputs, num_sample=self.codebook_size)
        modified_codes = torch.where(rearrange(is_expired, "N -> N ()"), samples, self.codebook.weight)
        self.codebook.weight.data.copy_(modified_codes)
        self.ema.data.add_(is_expired)
        #broadcast_tensors(self.buffers())

    def _update_codes(self, inputs: Tensor, indices: Tensor) -> None:
        code_onehot = F.one_hot(indices, self.codebook_size).type(inputs.dtype)
        code_onehot = rearrange(code_onehot, "B N -> N B")
        # [N]
        code_counts = code_onehot.sum(1)
        self.ema.data.mul_(self.code_decay).add_(code_counts, alpha=(1 - self.code_decay))

    def _quantize(self, inputs: Tensor) -> Tensor:
        codes = self.norm(self.codebook.weight)
        # [B, N]
        dist = compute_cosine_distances(inputs, codes)
        # [B]
        indices = dist.min(dim=1).indices
        return indices

    def _dequantize(self, indices: Tensor) -> Tensor:
        # [B, T, D]
        quantized = self.codebook(indices)
        if self.norm:
            quantized = self.norm(quantized)
        quantized = rearrange(quantized, "B T D -> B D T")
        return quantized

    @typecheck(
        input_types={
            "inputs": NeuralType(('B', 'D', 'T'), EncodedRepresentation())
        },
        output_types={"preprocessed": NeuralType(('D', 'D', 'T'), EncodedRepresentation())},
    )
    def preprocess_input(self, inputs):
        preprocessed = inputs
        if self.norm:
            preprocessed = rearrange(preprocessed, "B D T -> B T D")
            preprocessed = self.norm(preprocessed)
            preprocessed = rearrange(preprocessed, "B T D -> B D T")
        return preprocessed

    @property
    def input_types(self):
        return {
            "inputs": NeuralType(('B', 'D', 'T'), EncodedRepresentation()),
            "input_len": NeuralType(tuple('B'), LengthsType()),
        }

    @property
    def output_types(self):
        return {
            "dequantized": NeuralType(('B', 'D', 'T'), EncodedRepresentation()),
            "indices": NeuralType(('B', 'T'), Index()),
            "commit_loss": NeuralType((), LossType()),
            "codebook_loss": NeuralType((), LossType()),
        }

    @typecheck()
    def forward(self, inputs, input_len):
        input_flat = rearrange(inputs, "B D T -> (B T) D")
        # [(B T)]
        indices_flat = self._quantize(inputs=input_flat)
        # [B, T]
        indices = indices_flat.view([inputs.shape[0], inputs.shape[2]])
        # [B, T, D]
        dequantized = self._dequantize(indices=indices)
        dequantized = mask_sequence_tensor(dequantized, input_len)
        indices = mask_sequence_tensor(indices, input_len)

        if self.training:
            commit_loss, codebook_loss = self._compute_codebook_losses(
                inputs=inputs, codes=dequantized, input_len=input_len
            )
            dequantized = inputs + (dequantized - inputs).detach()
            if self.code_ema_threshold:
                self._expire_codes(inputs=input_flat)
                self._update_codes(inputs=input_flat, indices=indices_flat)
        else:
            commit_loss = 0.0
            codebook_loss = 0.0

        return dequantized, indices, commit_loss, codebook_loss

    @typecheck(
        input_types={
            "inputs": NeuralType(('B', 'D', 'T'), EncodedRepresentation()),
            "input_len": NeuralType(tuple('B'), LengthsType()),
        },
        output_types={"indices": NeuralType(('B', 'T'), Index())},
    )
    def encode(self, inputs, input_len):
        input_flat = rearrange(inputs, "B D T -> (B T) D")
        # [(B T)]
        indices_flat = self._quantize(inputs=input_flat)
        # [B, T]
        indices = indices_flat.view([inputs.shape[0], inputs.shape[2]])
        indices = mask_sequence_tensor(indices, input_len)
        return indices

    @typecheck(
        input_types={"indices": NeuralType(('B', 'T'), Index()), "input_len": NeuralType(tuple('B'), LengthsType()),},
        output_types={"quantized": NeuralType(('B', 'D', 'T'), EncodedRepresentation())},
    )
    def decode(self, indices, input_len):
        # [B, D, T]
        quantized = self._dequantize(indices=indices)
        quantized = mask_sequence_tensor(quantized, input_len)
        return quantized


class GroupCosineCodebook(NeuralModule):
    """Split the input vector into groups and apply codebook on each group separately.

    Args:
        num_codebooks: total number of codebooks
        codebook_dim: embedding dimension, will be split into num_codebooks
        **kwargs: parameters of ResidualVectorQuantizer

    References:
        Yang et al, HiFi-Codec: Group-residual Vector quantization for High Fidelity Audio Codec, 2023 (http://arxiv.org/abs/2305.02765).
    """

    def __init__(self, num_codebooks: int, codebook_dim: int, **kwargs):
        super().__init__()

        self.num_codebooks = num_codebooks
        self.codebook_dim = codebook_dim

        # Initialize RVQ for each group
        self.codebooks = torch.nn.ModuleList(
            [
                CosineCodebook(codebook_dim=self.codebook_dim, **kwargs)
                for _ in range(self.num_codebooks)
            ]
        )

        logging.debug('Initialized %s with', self.__class__.__name__)
        logging.debug('\tnum_codebooks:           %d', self.num_codebooks)
        logging.debug('\tcodebook_dim:            %d', self.codebook_dim)

    @typecheck(
        input_types={
            "inputs": NeuralType(('B', 'D', 'T'), EncodedRepresentation()),
            "input_len": NeuralType(tuple('B'), LengthsType()),
        },
        output_types={"preprocessed": NeuralType(('B', 'D', 'T'), EncodedRepresentation())},
    )
    def preprocess_input(self, inputs, input_len: Tensor):
        # C lists [B, W, T]
        inputs_grouped = inputs.chunk(self.num_codebooks, dim=1)

        preprocessed_list = []
        for in_group, codebook in zip(inputs_grouped, self.codebooks):
            preprocessed_i = codebook.preprocess_input(inputs=in_group)
            preprocessed_list.append(preprocessed_i)

        # [B, D, T]
        preprocessed = torch.cat(preprocessed_list, dim=1)
        preprocessed = mask_sequence_tensor(preprocessed, input_len)
        return preprocessed

    @property
    def input_types(self):
        return {
            "inputs": NeuralType(('B', 'D', 'T'), EncodedRepresentation()),
            "input_len": NeuralType(tuple('B'), LengthsType()),
        }

    @property
    def output_types(self):
        return {
            "dequantized": NeuralType(('B', 'D', 'T'), EncodedRepresentation()),
            "indices": NeuralType(('C', 'B', 'T'), Index()),
            "commit_loss": NeuralType((), LossType()),
            "codebook_loss": NeuralType((), LossType()),
        }

    @typecheck()
    def forward(self, inputs, input_len):
        """Quantize each group separately, then concatenate the results.
        """
        inputs_grouped = inputs.chunk(self.num_codebooks, dim=1)

        dequantized, indices = [], []
        commit_loss = 0.0
        codebook_loss = 0.0

        for in_group, codebook in zip(inputs_grouped, self.codebooks):
            dequantized_group, indices_group, commit_loss_group, codebook_loss_group = codebook(
                inputs=in_group, input_len=input_len
            )
            dequantized.append(dequantized_group)
            indices.append(indices_group)
            commit_loss += commit_loss_group
            codebook_loss += codebook_loss_group

        commit_loss /= len(self.codebooks)
        codebook_loss /= len(self.codebooks)

        # concatenate along the feature dimension
        dequantized = torch.cat(dequantized, dim=1)

        # stack along the codebook dimension
        indices = torch.stack(indices, dim=0)

        return dequantized, indices, commit_loss, codebook_loss

    @typecheck(
        input_types={
            "inputs": NeuralType(('B', 'D', 'T'), EncodedRepresentation()),
            "input_len": NeuralType(tuple('B'), LengthsType()),
        },
        output_types={"indices": NeuralType(('C', 'B', 'T'), Index())},
    )
    def encode(self, inputs: Tensor, input_len: Tensor) -> Tensor:
        """Input is split into groups, each group is encoded separately, then the results are concatenated.
        """
        inputs_grouped = inputs.chunk(self.num_codebooks, dim=1)
        indices = []

        for in_group, codebook in zip(inputs_grouped, self.codebooks):
            indices_group = codebook.encode(inputs=in_group, input_len=input_len)
            indices.append(indices_group)

        # stack along the codebook dimension
        indices = torch.stack(indices, dim=0)

        return indices

    @typecheck(
        input_types={
            "indices": NeuralType(('C', 'B', 'T'), Index()),
            "input_len": NeuralType(tuple('B'), LengthsType()),
        },
        output_types={"dequantized": NeuralType(('B', 'D', 'T'), EncodedRepresentation()),},
    )
    def decode(self, indices: Tensor, input_len: Tensor) -> Tensor:
        """Input indices are split into groups, each group is decoded separately, then the results are concatenated.
        """
        dequantized = []

        for i, codebook in enumerate(self.codebooks):
            indices_group = indices[i]
            dequantized_group = codebook.decode(indices=indices_group, input_len=input_len)
            dequantized.append(dequantized_group)

        # concatenate along the feature dimension
        dequantized = torch.cat(dequantized, dim=1)

        return dequantized