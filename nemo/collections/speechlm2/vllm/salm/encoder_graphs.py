# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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
"""CUDA graphs for the ParallelExpertEncoder perception forward in the vLLM plugin.

vLLM encodes the audio of the requests it admits in one engine step, typically a few dozen short clips. At that
batch size the perception forward is bound by kernel launches and Python: ~1.8k kernels per call, with the GPU idle
more than half the time. This module captures the forward (everything after the mel preprocessor) into CUDA
graphs over a grid of (batch, frames) buckets and replays them.

Audio up to ``max_windows`` online-inference windows is graphed (the windowed path's control flow depends only on
shapes); longer audio falls back to the eager path. Rows are padded to the bucket with copies of the first row and
frames with zeros, which the encoder masks by length like the padding of any shorter row in a batch.
"""
from __future__ import annotations

import dataclasses
import logging
import math

import torch
from torch import nn

from nemo.collections.common.parts.optional_cuda_graphs import capture_cuda_graph

logger = logging.getLogger(__name__)

# Config keys (checkpoint ``config.json`` or vLLM ``--hf-overrides``) that set the graph buckets, mapped to
# EncoderGraphRunner arguments.
ENCODER_GRAPH_CONFIG_KEYS = {
    "encoder_cuda_graph_batch_sizes": "batch_sizes",
    "encoder_cuda_graph_frame_bucket": "frame_bucket",
    "encoder_cuda_graph_max_windows": "max_windows",
}
# Capture compiles the encoder's attention once per shape class it meets (batch size 1, sequence length, default
# dtype of each branch). Every bucket is captured in one call, which can need more variants than dynamo's default
# recompile limit; past the limit, compiled functions run uncompiled code that cannot be captured.
_CAPTURE_RECOMPILE_LIMIT = 64
# Up to this many rows, one graph replay costs about the same as one of a smaller batch, so splitting the rows over
# several graphs only adds replays.
_MIN_SPLIT_ROWS = 8


class EncoderGraphRunner:
    """Replays the perception forward from CUDA graphs captured per (batch, frames) bucket.

    Args:
        perception: Perception module whose encoder is a ParallelExpertEncoder (``chunk_feat_len`` and
            ``online_inference()``).
        batch_sizes: Graph batch sizes; a call's rows are split over them, largest first.
        frame_bucket: Graph length granularity in feature frames. Buckets never cross a window boundary.
        max_windows: Longest graphed audio, in online-inference windows.
        compile_layers: ``torch.compile`` every transformer block of the encoder first. This fuses the
            memory-bound LayerNorm, RoPE and residual operations that dominate the encoder's GPU time at these
            batch sizes, and changes numerics at bf16 rounding level.

    Set ``ctc_timestamp_inputs`` before the first call to also capture the encoder's CTC timestamp inputs, so
    that calls can return them.
    """

    def __init__(
        self,
        perception: nn.Module,
        batch_sizes: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64, 128),
        frame_bucket: int = 256,
        max_windows: int = 2,
        compile_layers: bool = False,
    ):
        if not batch_sizes or any(int(b) < 1 for b in batch_sizes):
            raise ValueError(f"batch_sizes must be positive integers, got {batch_sizes!r}")
        if frame_bucket < 1 or max_windows < 1:
            raise ValueError(f"frame_bucket and max_windows must be positive, got {frame_bucket}, {max_windows}")
        self.perception = perception
        self.window = int(perception.encoder.chunk_feat_len)
        self.max_frames = self.window * max_windows
        self.frame_bucket = frame_bucket
        self.batch_sizes = sorted({int(b) for b in batch_sizes})
        # Graphs over more frames hold fewer rows, so no graph needs more memory than the largest batch of
        # single windows.
        self.max_rows_x_frames = self.batch_sizes[-1] * self.window
        self.graphs: dict[tuple[int, int], tuple] = {}
        self.pool = None
        self.enabled = True
        self.ctc_timestamp_inputs = False
        # Audio longer than the graphed windows runs the eager online path, whose steady-state streaming
        # diarization steps still replay from CUDA graphs.
        perception.encoder.online_cuda_graphs = True
        if compile_layers:
            for name, module in perception.encoder.named_modules():
                if isinstance(module, nn.ModuleList) and name.endswith("layers"):
                    for layer in module:
                        layer.compile(dynamic=True)

    def _sizes_for(self, frames: int) -> list[int]:
        return [b for b in self.batch_sizes if b * frames <= self.max_rows_x_frames] or self.batch_sizes[:1]

    def _split(self, rows: int, frames: int) -> list[int]:
        """Cover ``rows`` with graph batch sizes: the largest that fits, padding only the last piece.

        Fewer than ``_MIN_SPLIT_ROWS`` remaining rows go into one graph rather than several smaller ones, since a
        replay costs about the same at any batch size up to that.
        """
        sizes = self._sizes_for(frames)
        pieces = []
        while rows > 0:
            fit = [b for b in sizes if b <= rows]
            if fit and (rows >= _MIN_SPLIT_ROWS or fit[-1] == rows):
                pieces.append(fit[-1])
            else:
                pieces.append(next((b for b in sizes if b >= rows), sizes[-1]))
            rows -= pieces[-1]
        return pieces

    def _frame_bucket(self, frames: int) -> int:
        # Never pad across a window boundary: the padded input then runs the same windows as eager inference.
        return min(
            math.ceil(frames / self.window) * self.window, math.ceil(frames / self.frame_bucket) * self.frame_bucket
        )

    def _forward(self, x: torch.Tensor, x_len: torch.Tensor):
        kwargs = {"return_ctc_timestamp_inputs": True} if self.ctc_timestamp_inputs else {}
        with self.perception.encoder.online_inference():
            out = self.perception(processed_signal=x, processed_signal_length=x_len, **kwargs)
        return out if self.ctc_timestamp_inputs else (*out, None)

    def _capture_all(self, n_feat: int, dtype: torch.dtype) -> None:
        """Capture every bucket at the first call, largest first so smaller graphs reuse the pool's blocks."""
        lengths = {self._frame_bucket(t) for t in range(1, self.max_frames + 1)}  # every bucket a call can reach
        for frames in sorted(lengths, reverse=True):
            for rows in reversed(self._sizes_for(frames)):
                self._capture(rows, frames, n_feat, dtype)

    def _capture(self, rows: int, frames: int, n_feat: int, dtype: torch.dtype) -> None:
        device = next(self.perception.parameters()).device
        x = torch.zeros(rows, n_feat, frames, device=device, dtype=dtype)
        x_len = torch.full((rows,), frames, device=device, dtype=torch.long)
        if self.pool is None:
            self.pool = torch.cuda.graph_pool_handle()
        graph, (emb, emb_len, states) = capture_cuda_graph(lambda: self._forward(x, x_len), pool=self.pool)
        self.graphs[rows, frames] = (graph, x, x_len, emb, emb_len, states)
        logger.info("Captured encoder CUDA graph for %d rows x %d frames", rows, frames)

    @torch.no_grad()
    def __call__(
        self, audio_signal: torch.Tensor, audio_lengths: torch.Tensor, return_ctc_timestamp_inputs: bool = False
    ):
        """Encode a batch of waveforms.

        Returns per-row embeddings, and with ``return_ctc_timestamp_inputs`` also the batch's
        ``CTCTimestampInputs``, like the perception forward. Returns None (encode eagerly) when the audio is too long
        to graph, capture failed, or the inputs are requested from graphs captured without them.
        """
        if not self.enabled or (return_ctc_timestamp_inputs and not self.ctc_timestamp_inputs):
            return None
        processed, processed_len = self.perception.maybe_preprocess_audio(audio_signal, audio_lengths, None, None)
        n_feat, frames = processed.shape[1], processed.shape[2]
        if frames > self.max_frames:
            return None
        if not self.graphs:
            try:
                with _compile_limit_for_capture():
                    self._capture_all(n_feat, processed.dtype)
            except Exception as error:  # noqa: BLE001 - a failed capture must not break inference
                logger.warning("Encoder CUDA graph capture failed; encoding eagerly from now on: %s", error)
                self.graphs.clear()
                self.pool = None
                self.enabled = False
                return None
        bucket = self._frame_bucket(frames)
        out, pieces, start = [], [], 0
        for size in self._split(processed.shape[0], bucket):
            rows = processed[start : start + size]
            n = rows.shape[0]
            graph, x, x_len, emb, emb_len, states = self.graphs[size, bucket]
            x[:n, :, :frames].copy_(rows)
            x[:n, :, frames:].zero_()
            x[n:].copy_(x[:1].expand_as(x[n:]))
            x_len[:n].copy_(processed_len[start : start + n])
            x_len[n:].copy_(x_len[:1].expand_as(x_len[n:]))
            start += n
            graph.replay()
            # One host sync per piece for the lengths; the clones leave the static output buffer free for reuse.
            out.extend(emb[i, :length].clone() for i, length in enumerate(emb_len[:n].tolist()))
            if return_ctc_timestamp_inputs:
                pieces.append(_rows(states, n))
        if not return_ctc_timestamp_inputs:
            return out
        tensors = {name: torch.cat([getattr(p, name) for p in pieces]) for name in _tensor_fields(pieces[0])}
        return out, dataclasses.replace(pieces[0], **tensors)


def _tensor_fields(states) -> list[str]:
    return [f.name for f in dataclasses.fields(states) if isinstance(getattr(states, f.name), torch.Tensor)]


def _rows(states, n: int):
    """Copies of the first ``n`` rows of every tensor in captured ``CTCTimestampInputs``."""
    return dataclasses.replace(states, **{name: getattr(states, name)[:n].clone() for name in _tensor_fields(states)})


def encoder_graph_options(config) -> dict:
    """Return the EncoderGraphRunner arguments set in a model config (see ``ENCODER_GRAPH_CONFIG_KEYS``).

    Args:
        config: HF model config, after ``--hf-overrides``.

    Returns:
        Keyword arguments for EncoderGraphRunner; unset keys are left out, so the runner's defaults apply.
    """
    options = {}
    for key, argument in ENCODER_GRAPH_CONFIG_KEYS.items():
        value = getattr(config, key, None)
        if value is not None:
            options[argument] = tuple(int(b) for b in value) if argument == "batch_sizes" else int(value)
    return options


def _compile_limit_for_capture():
    """Raise dynamo's per-function recompile limit to ``_CAPTURE_RECOMPILE_LIMIT`` while the graphs are captured."""
    config = torch._dynamo.config
    name = "recompile_limit" if hasattr(config, "recompile_limit") else "cache_size_limit"
    return config.patch(**{name: max(getattr(config, name), _CAPTURE_RECOMPILE_LIMIT)})
