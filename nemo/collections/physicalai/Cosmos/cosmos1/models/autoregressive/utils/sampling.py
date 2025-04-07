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

from typing import Optional, Tuple

import torch

from cosmos1.models.autoregressive.networks.transformer import Transformer


def sample_top_p(logits, temperature, top_p, return_probs: bool = False):
    """
    Perform top-p (nucleus) sampling on a probability distribution.

    Args:
        logits (torch.Tensor): Logits of the probability distribution.
        temperature (float): Temperature for sampling.
        top_p (float): Probability threshold for top-p sampling.

    Returns:
        torch.Tensor: Sampled token indices.

    Note:
        Top-p sampling selects the smallest set of tokens whose cumulative probability mass
        exceeds the threshold p. The distribution is renormalized based on the selected tokens.
    """
    probs = torch.softmax(logits[:, -1, :] / temperature, dim=-1)
    # Sort the probabilities in descending order and get their indices.
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    # Compute the cumulative sum of the sorted probabilities.
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    # Create a mask where the cumulative probability exceeds the threshold p.
    mask = probs_sum - probs_sort > top_p
    # Set the probabilities that exceed the threshold to 0.
    probs_sort[mask] = 0.0
    # Renormalize the remaining probabilities so they sum to 1.
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    # Sample from the renormalized probability distribution.
    # next_token = torch.multinomial(probs_sort, num_samples=1)
    next_token = multinomial_sample_one_no_sync(probs_sort, dtype=torch.int64)
    # Gather the indices of the sampled tokens.
    next_token = torch.gather(probs_idx, -1, next_token)
    if return_probs:
        # Initialize a tensor for unsorted probabilities
        probs_unsorted = torch.zeros_like(probs_sort)
        # Scatter the sorted probabilities back to their original order
        probs_unsorted.scatter_(-1, probs_idx, probs_sort)
    else:
        probs_unsorted = None
    return next_token, probs_unsorted


def multinomial_sample_one_no_sync(probs_sort, dtype=torch.int):
    """
    Multinomial sampling without a cuda synchronization.
    Source: https://github.com/pytorch-labs/gpt-fast/blob/main/generate.py
    """
    q = torch.empty_like(probs_sort).exponential_(1)
    return torch.argmax(probs_sort / q, dim=-1, keepdim=True).to(dtype=dtype)


def logits_to_probs(
    logits,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
):
    logits = logits / max(temperature, 1e-5)

    if top_k is not None:
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        pivot = v.select(-1, -1).unsqueeze(-1)
        logits = torch.where(logits < pivot, -float("Inf"), logits)
    probs = torch.nn.functional.softmax(logits, dim=-1)
    return probs


def sample_top_k(logits, temperature: float = 1.0, top_k: Optional[int] = None):
    """
    Sample from the logits using top-k sampling.
    Source: https://github.com/pytorch-labs/gpt-fast/blob/main/generate.py
    """
    # logits: [batch_size, seq_len, vocab_size]
    if temperature == 0.0:
        idx_next = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        probs = None
    else:
        probs = logits_to_probs(logits[:, -1, :], temperature, top_k)
        idx_next = multinomial_sample_one_no_sync(probs)
    return idx_next, probs


def prefill(
    model: Transformer,
    input_pos: torch.Tensor,
    tokens: torch.Tensor = None,
    token_embeddings: torch.Tensor = None,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    **kwargs,
) -> torch.Tensor:
    logits = model(tokens=tokens, token_embeddings=token_embeddings, input_pos=input_pos, **kwargs)
    # Only top-p or top-k can be provided
    assert (
        top_p is None or top_k is None
    ), "Only one of top-p or top-k can be provided, got top-p={top_p} and top-k={top_k}"
    if top_p is not None:
        return sample_top_p(logits, temperature=temperature, top_p=top_p)[0]
    else:
        return sample_top_k(logits, temperature=temperature, top_k=top_k)[0]


def decode_one_token(
    model: Transformer,
    tokens: torch.Tensor,
    input_pos: torch.Tensor,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    **kwargs,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Decode a single token from the autoregressive model.
    """
    logits = model(tokens=tokens, input_pos=input_pos, **kwargs)
    if top_p is not None:
        return sample_top_p(logits, temperature=temperature, top_p=top_p)
    else:
        return sample_top_k(logits, temperature=temperature, top_k=top_k)


def decode_n_tokens(
    model: Transformer,
    cur_token: torch.Tensor,
    input_pos: torch.Tensor,
    num_new_tokens: int,
    stop_tokens: torch.Tensor = None,
    temperature: float = 1.0,
    top_p: Optional[float] = None,
    top_k: Optional[int] = None,
    return_probs: bool = False,
    decode_one_token_function=decode_one_token,
    **kwargs,
):
    """
    Decode n tokens from the autoregressive model.
    Adapted from https://github.com/pytorch-labs/gpt-fast/blob/main/generate.py
    """
    new_tokens, new_probs = [], []
    batch_size = cur_token.shape[0]
    assert (
        top_p is None or top_k is None
    ), "Only one of top-p or top-k can be provided, got top-p={top_p} and top-k={top_k}"
    if stop_tokens is not None:
        # Indicator for whether the EOS token (stop token) has been reached for each sample in the batch
        eos_reached = torch.tensor([False] * batch_size, device="cuda")
    for t in range(num_new_tokens):
        with torch.backends.cuda.sdp_kernel(
            enable_flash=False, enable_mem_efficient=False, enable_math=True
        ):  # Actually better for Inductor to codegen attention here
            next_token, next_prob = decode_one_token_function(
                model,
                tokens=cur_token,
                input_pos=input_pos,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                **kwargs,
            )
            input_pos += 1
            if stop_tokens is not None and len(stop_tokens) > 0:
                eos_reached = eos_reached | (torch.isin(next_token, stop_tokens))
                if eos_reached.all():
                    break
            new_tokens.append(next_token.clone())
            if return_probs:
                new_probs.append(next_prob.clone())
            cur_token = next_token.clone()

    if return_probs:
        return new_tokens, new_probs
    else:
        return new_tokens
