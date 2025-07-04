# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for `StreamingBatchedAudioBuffer` and accompanying helper
classes defined in
`nemo.collections.asr.parts.utils.streaming_utils`.
"""

from __future__ import annotations

import math

import pytest
import torch

from nemo.collections.asr.parts.utils.streaming_utils import ContextSize, ContextSizeBatch, StreamingBatchedAudioBuffer

# -----------------------------------------------------------------------------
# Helper constants / fixtures
# -----------------------------------------------------------------------------

DEVICES: list[torch.device] = [torch.device("cpu")]
if torch.cuda.is_available():
    DEVICES.append(torch.device("cuda:0"))

# -----------------------------------------------------------------------------
# Tests for ContextSize and ContextSizeBatch
# -----------------------------------------------------------------------------


@pytest.mark.unit
def test_context_size_total_and_subsample():
    ctx = ContextSize(left=4, chunk=2, right=1)
    assert ctx.total() == 7

    half_ctx = ctx.subsample(factor=2)
    assert isinstance(half_ctx, ContextSize)
    assert half_ctx.left == 2 and half_ctx.chunk == 1 and half_ctx.right == 0
    assert half_ctx.total() == math.floor(7 / 2)


@pytest.mark.unit
@pytest.mark.parametrize("device", DEVICES)
def test_context_size_batch_total_and_subsample(device: torch.device):
    left = torch.tensor([4, 4], dtype=torch.long, device=device)
    chunk = torch.tensor([2, 2], dtype=torch.long, device=device)
    right = torch.tensor([2, 2], dtype=torch.long, device=device)
    batch_ctx = ContextSizeBatch(left=left, chunk=chunk, right=right)

    # total() should equal element-wise sum
    expected_total = left + chunk + right
    assert torch.equal(batch_ctx.total(), expected_total)

    # After subsampling by 2 each component should be halved (floor division)
    half_ctx = batch_ctx.subsample(2)
    assert torch.equal(half_ctx.left, left // 2)
    assert torch.equal(half_ctx.chunk, chunk // 2)
    assert torch.equal(half_ctx.right, right // 2)


# -----------------------------------------------------------------------------
# Tests for StreamingBatchedAudioBuffer
# -----------------------------------------------------------------------------


def _create_audio_batch(batch_size: int, length: int, device: torch.device, dtype: torch.dtype = torch.float32):
    """Create a dummy audio batch of shape (batch_size, length)."""
    # Use a simple ramp signal to ease debugging.
    vals = torch.arange(batch_size * length, device=device, dtype=dtype)
    return vals.view(batch_size, length)


@pytest.mark.unit
@pytest.mark.parametrize("device", DEVICES)
def test_streaming_batched_audio_buffer(device: torch.device):
    batch_size = 2
    expected_ctx = ContextSize(left=4, chunk=2, right=1)  # total = 7
    buffer = StreamingBatchedAudioBuffer(
        batch_size=batch_size,
        context_samples=expected_ctx,
        dtype=torch.float32,
        device=device,
    )

    # ------------------------------------------------------------------
    # First add : chunk + right (filling initial buffer)
    # ------------------------------------------------------------------
    first_len = expected_ctx.chunk + expected_ctx.right  # 3
    audio_batch = _create_audio_batch(batch_size, first_len, device)
    audio_lens = torch.full(
        [
            batch_size,
        ],
        first_len,
        dtype=torch.long,
        device=device,
    )
    buffer.add_audio_batch_(
        audio_batch=audio_batch,
        audio_lengths=audio_lens,
        is_last_chunk=False,
        is_last_chunk_batch=torch.zeros(batch_size, dtype=torch.bool, device=device),
    )

    # Validate context sizes
    assert buffer.context_size.left == 0
    assert buffer.context_size.chunk == expected_ctx.chunk
    assert buffer.context_size.right == expected_ctx.right
    assert buffer.samples.shape[1] == first_len  # No truncation yet

    # ------------------------------------------------------------------
    # Second add : only chunk length
    # ------------------------------------------------------------------
    chunk_len = expected_ctx.chunk  # 2
    audio_batch = _create_audio_batch(batch_size, chunk_len, device)
    audio_lens.fill_(chunk_len)
    buffer.add_audio_batch_(
        audio_batch=audio_batch,
        audio_lengths=audio_lens,
        is_last_chunk=False,
        is_last_chunk_batch=torch.zeros(batch_size, dtype=torch.bool, device=device),
    )

    # After second add, left should have grown by previous chunk (2)
    assert buffer.context_size.left == 2
    assert buffer.context_size.chunk == expected_ctx.chunk
    assert buffer.context_size.right == expected_ctx.right
    assert buffer.samples.shape[1] == 5  # 2 (left) + 2 (chunk) + 1 (right)

    # ------------------------------------------------------------------
    # Third add : another chunk, buffer should now reach full capacity (7)
    # ------------------------------------------------------------------
    buffer.add_audio_batch_(
        audio_batch=audio_batch,
        audio_lengths=audio_lens,
        is_last_chunk=False,
        is_last_chunk_batch=torch.zeros(batch_size, dtype=torch.bool, device=device),
    )

    assert buffer.samples.shape[1] == expected_ctx.total()
    assert buffer.context_size.total() == expected_ctx.total()

    # ------------------------------------------------------------------
    # Fourth add : buffer overflows by 2 samples; implementation should
    # drop the excess from the left context.
    # ------------------------------------------------------------------
    buffer.add_audio_batch_(
        audio_batch=audio_batch,
        audio_lengths=audio_lens,
        is_last_chunk=False,
        is_last_chunk_batch=torch.zeros(batch_size, dtype=torch.bool, device=device),
    )

    # Buffer length remains constant (total context size)
    assert buffer.samples.shape[1] == expected_ctx.total()
    assert buffer.context_size.total() == expected_ctx.total()

    # Left context should have been clipped by 2 samples (from 6 to 4)
    assert buffer.context_size.left == expected_ctx.left  # 4

    # ------------------------------------------------------------------
    # Final add : mark last chunk with shorter length; right context
    # should go to 0 afterwards.
    # ------------------------------------------------------------------
    last_len = 1
    audio_batch = _create_audio_batch(batch_size, last_len, device)
    audio_lens.fill_(last_len)
    buffer.add_audio_batch_(
        audio_batch=audio_batch,
        audio_lengths=audio_lens,
        is_last_chunk=True,
        is_last_chunk_batch=torch.ones(batch_size, dtype=torch.bool, device=device),
    )

    # After last chunk, right context must be zero and total size preserved
    assert buffer.context_size.right == 0
    assert buffer.context_size.total() == expected_ctx.total()
    assert buffer.samples.shape[1] == expected_ctx.total()


@pytest.mark.unit
@pytest.mark.parametrize("device", DEVICES)
def test_streaming_batched_audio_buffer_raises_on_too_long_chunk(device: torch.device):
    """`add_audio_batch_` should raise if provided chunk is larger than chunk + right."""

    expected_ctx = ContextSize(left=0, chunk=2, right=1)
    buffer = StreamingBatchedAudioBuffer(
        batch_size=1,
        context_samples=expected_ctx,
        dtype=torch.float32,
        device=device,
    )

    # Attempt to add a chunk that is too long (4 > 3)
    too_long_chunk_size = expected_ctx.chunk + expected_ctx.right + 1
    audio = _create_audio_batch(1, too_long_chunk_size, device)
    audio_lens = torch.tensor([too_long_chunk_size], dtype=torch.long, device=device)

    with pytest.raises(ValueError):
        buffer.add_audio_batch_(
            audio_batch=audio,
            audio_lengths=audio_lens,
            is_last_chunk=False,
            is_last_chunk_batch=torch.tensor([False], dtype=torch.bool, device=device),
        )
