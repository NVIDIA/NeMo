# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn as nn


def create_norm(norm_type: str, dim: int, eps: float = 1e-6):
    """
    Creates the specified normalization layer based on the norm_type.
    Adopted from TorchTriton: https://github.com/pytorch/torchtitan/blob/main/torchtitan/models/norms.py

    Args:
        norm_type (str): The type of normalization layer to create.
            Supported types: 1. rmsnorm 2. fused_rmsnorm 3. layernorm 4. np_layernorm
        dim (int): The dimension of the normalization layer.
        eps (float, optional): The epsilon value for numerical stability. Defaults to 1e-6.

    Returns:
        The created normalization layer.

    Raises:
        NotImplementedError: If an unknown norm_type is provided.
    """
    norm_type = norm_type.lower()  # Normalize to lowercase

    if norm_type == "layernorm":
        return nn.LayerNorm(dim, eps=eps, bias=False)
    elif norm_type == "np_layernorm":
        return nn.LayerNorm(dim, eps=eps, elementwise_affine=False, bias=False)
    elif norm_type == "rmsnorm":
        return RMSNorm(dim, eps=eps, compile=False)
    elif norm_type == "compiled_rmsnorm":
        return RMSNorm(dim, eps=eps, compile=True)
    elif norm_type == "fused_rmsnorm":
        raise NotImplementedError("Fused RMSNorm is not supported yet.")
    else:
        raise NotImplementedError(f"Unknown norm_type: '{norm_type}'")


class RMSNorm(nn.Module):
    """
    Initialize the RMSNorm normalization layer.
    Reference implementation: https://github.com/pytorch/torchtitan/blob/main/torchtitan/models/norms.py

    Args:
        dim (int): The dimension of the input tensor.
        eps (float, optional): A small value added to the denominator for numerical stability. Default is 1e-6.
        compile (bool, optional): Whether to compile the forward function. Default is False.

    Attributes:
        eps (float): A small value added to the denominator for numerical stability.
        weight (nn.Parameter): Learnable scaling parameter.

    """

    def __init__(self, dim: int, eps: float = 1e-6, compile: bool = False):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        self.rmsnorm_fn = torch.compile(self.compute_rmsnorm, fullgraph=True) if compile else self.compute_rmsnorm

    @staticmethod
    def compute_rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float):
        def _norm(x, eps):
            # Computes the root-mean-square norm of the input tensor.
            return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)

        output = _norm(x.float(), eps).type_as(x)
        return output * weight

    def forward(self, x: torch.Tensor):
        return self.rmsnorm_fn(x, self.weight, self.eps)

    def reset_parameters(self):
        torch.nn.init.ones_(self.weight)
