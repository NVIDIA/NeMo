# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import itertools
import weakref
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from nemo.collections.speechlm2.parts.ctc_timestamp_utils import (
    MultiSpeakerSOTWordTimestampAligner,
    _one_pass_positions,
    _straight_band_search,
)


@pytest.mark.unit
@pytest.mark.parametrize("text", ["piece—just", "piece — just", "—piece——just—"])
def test_em_dash_separates_alignment_words(text):
    transcript = f"<spk:1> <sty:stt:flat> {text} <ev:laughter> <sty:end:flat>"
    words = MultiSpeakerSOTWordTimestampAligner.parse_sot_words(transcript)
    assert [w["word"] for w in words] == [
        "<sty:stt:flat>",
        "piece",
        "just",
        "<ev:laughter>",
        "<sty:end:flat>",
    ]
    assert all(w["speaker_tag"] == 1 for w in words)
    assert [w["word_index"] for w in words] == list(range(5))
    assert [w["word"] for w in MultiSpeakerSOTWordTimestampAligner.parse_sot_words("don't well-known")] == [
        "don't",
        "well-known",
    ]


@pytest.mark.unit
@pytest.mark.parametrize("core_length", [0, 3, 7])
@pytest.mark.parametrize("speaker_weight", [0.0, 0.25])
@pytest.mark.parametrize("window_budget", [16 * 1024 * 1024, 768])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_deferred_windows_release_full_vocabulary_and_preserve_alignment(
    core_length, speaker_weight, window_budget, dtype
):
    class Decoder(torch.nn.Module):
        num_classes_with_blank = 128

        def __init__(self):
            super().__init__()
            self.outputs = []
            self.check_lifetime = False
            self.shapes = []

        def forward(self, states, encoded_lengths):
            if self.check_lifetime:
                assert all(ref() is None for ref in self.outputs), "Previous full-vocabulary window is still live"
            self.shapes.append(states.shape)
            # Depend on context as well as the frame, so incorrect window slicing changes the oracle.
            logits = states.transpose(1, 2) + states.mean(dim=-1).unsqueeze(1)
            output = logits.log_softmax(dim=-1)
            self.outputs.append(weakref.ref(output))
            return output

    class Tokenizer:
        def text_to_ids(self, text):
            return [{"a": 2, "b": 7}[word] for word in text.split()]

    torch.manual_seed(23)
    decoder = Decoder()
    aligner = MultiSpeakerSOTWordTimestampAligner(
        ctc_decoder=decoder,
        tokenizer=Tokenizer(),
        blank_id=126,  # Also exercise a blank other than the last vocabulary column.
        online_inference_length=core_length,
        chunk_left_context=1,
        chunk_right_context=2,
        speaker_logprob_weight=speaker_weight,
    )
    inputs = SimpleNamespace(
        asr_encoded=torch.randn(3, 128, 7).to(dtype),
        asr_encoded_lengths=torch.tensor([7, 5, 2]),
        sortformer_sigmoids=torch.rand(3, 7, 2),
        sortformer_lengths=torch.tensor([7, 5, 2]),
        diarization_labels=None,
        diarization_lengths=None,
        diarization_frame_seconds=0.01,
    )
    transcripts = ["<spk:0> a a <spk:1> b", "b a", ""]
    durations = [0.7, 0.5, 0.2]
    # Former full-vocabulary window/cat implementation, independent of the new iterator.
    chunks = []
    for start in range(0, 7, core_length or 7):
        end = min(start + (core_length or 7), 7)
        left, right = max(start - 1, 0), min(end + 2, 7)
        lengths = (inputs.asr_encoded_lengths - left).clamp(min=0, max=right - left)
        output = decoder(inputs.asr_encoded[:, :, left:right], lengths)
        chunks.append(output[:, start - left : end - left])
    dense = torch.cat(chunks, dim=1)
    del chunks, output
    expected = aligner.prepare_alignment(
        dense,
        inputs.sortformer_sigmoids,
        transcripts,
        ctc_lengths=inputs.asr_encoded_lengths,
        sortformer_lengths=inputs.sortformer_lengths,
        audio_durations=durations,
    )
    del dense
    decoder.outputs.clear()
    decoder.shapes.clear()
    decoder.check_lifetime = True
    aligner._CTC_WINDOW_MAX_ELEMENTS = window_budget
    actual = aligner.prepare_from_inputs(inputs, transcripts, durations)
    assert all(ref() is None for ref in decoder.outputs)
    assert all(batch * frames * classes <= window_budget or batch == 1 for batch, classes, frames in decoder.shapes)
    for record, reference in zip(actual["records"], expected["records"]):
        assert record.keys() == reference.keys()
        for key in record:
            if isinstance(record[key], torch.Tensor):
                torch.testing.assert_close(record[key], reference[key], rtol=0, atol=0)
            else:
                assert record[key] == reference[key]
        assert record["ctc_log_probs"].device.type == "cpu"
        assert record["ctc_log_probs"].shape[1] <= 3
    assert aligner.align_prepared(actual) == aligner.align_prepared(expected)


@pytest.mark.unit
def test_word_timestamp_alignment_defaults_to_ctc_only():
    assert MultiSpeakerSOTWordTimestampAligner().speaker_logprob_weight == 0.0


def _dense_reference(
    log_probs,
    labels,
    state_lengths,
    blank_id,
    state_speaker_columns,
    speaker_probs,
    speaker_logprob_weight,
    epsilon=1.0e-6,
):
    """Former dense implementation retained only as an exactness oracle."""
    num_streams, max_states = labels.shape
    num_frames = log_probs.shape[0]
    state_mask = torch.arange(max_states).unsqueeze(0) < state_lengths.unsqueeze(1)
    emissions = log_probs[:, labels].permute(1, 0, 2).contiguous()
    emissions.masked_fill_(~state_mask.unsqueeze(1), -float("inf"))
    token_states = state_mask & (state_speaker_columns >= 0)
    if speaker_probs is not None and speaker_logprob_weight > 0 and token_states.any():
        columns = state_speaker_columns.clamp_min(0)
        activity = speaker_probs[:, columns].permute(1, 0, 2)
        emissions += torch.where(
            token_states.unsqueeze(1),
            speaker_logprob_weight * torch.log(activity.clamp_min(epsilon)),
            torch.zeros_like(activity),
        )

    previous = torch.full((num_streams, max_states), -float("inf"))
    previous[:, 0] = emissions[:, 0, 0]
    previous[:, 1] = emissions[:, 0, 1]
    backpointers = torch.full((num_frames, num_streams, max_states), -1, dtype=torch.long)
    states = torch.arange(max_states).unsqueeze(0).expand(num_streams, -1)
    for frame in range(1, num_frames):
        best = previous
        previous_states = states
        advance = torch.full_like(previous, -float("inf"))
        advance[:, 1:] = previous[:, :-1]
        take = advance > best
        best = torch.where(take, advance, best)
        previous_states = torch.where(take, states - 1, previous_states)
        skip = torch.full_like(previous, -float("inf"))
        can_skip = state_mask[:, 2:] & (labels[:, 2:] != blank_id) & (labels[:, 2:] != labels[:, :-2])
        skip[:, 2:] = torch.where(can_skip, previous[:, :-2], skip[:, 2:])
        take = skip > best
        best = torch.where(take, skip, best)
        previous_states = torch.where(take, states - 2, previous_states)
        previous = best + emissions[:, frame]
        previous.masked_fill_(~state_mask, -float("inf"))
        backpointers[frame] = previous_states

    last_blank = state_lengths - 1
    last_token = state_lengths - 2
    blank_scores = previous.gather(1, last_blank[:, None]).squeeze(1)
    token_scores = previous.gather(1, last_token[:, None]).squeeze(1)
    final_states = torch.where(token_scores > blank_scores, last_token, last_blank)
    final_scores = torch.maximum(token_scores, blank_scores)
    paths = torch.empty((num_streams, num_frames), dtype=torch.long)
    current = final_states
    stream_indices = torch.arange(num_streams)
    for frame in range(num_frames - 1, -1, -1):
        paths[:, frame] = current
        if frame:
            current = backpointers[frame, stream_indices, current]
    return list(paths), [float(score) for score in final_scores]


@pytest.mark.unit
@pytest.mark.parametrize("seed", [0, 7, 19])
@pytest.mark.parametrize("speaker_weight", [0.0, 0.25])
def test_compact_dp_matches_dense_reference_exactly(seed, speaker_weight):
    torch.manual_seed(seed)
    blank_id = 5
    labels = torch.tensor([[5, 0, 5, 1, 5, 1, 5], [5, 2, 5, 3, 5, 5, 5]])
    state_lengths = torch.tensor([7, 5])
    columns = torch.tensor([[-1, 0, -1, 0, -1, 0, -1], [-1, 1, -1, 1, -1, -1, -1]])
    log_probs = torch.log_softmax(torch.randn(15, blank_id + 1), dim=-1)
    speaker_probs = torch.sigmoid(torch.randn(15, 2))
    args = (
        log_probs,
        labels,
        state_lengths,
        blank_id,
        columns,
        speaker_probs,
        speaker_weight,
    )

    expected_paths, expected_scores = _dense_reference(*args)
    actual_paths, actual_scores = MultiSpeakerSOTWordTimestampAligner()._ctc_forced_align_batched(*args)

    assert all(torch.equal(actual, expected) for actual, expected in zip(actual_paths, expected_paths))
    assert actual_scores == pytest.approx(expected_scores, abs=1.0e-6)


@pytest.mark.unit
def test_compact_dp_storage_for_ten_minute_shape():
    storage = MultiSpeakerSOTWordTimestampAligner.estimate_dp_storage_bytes(
        num_frames=7500,
        num_streams=4,
        max_states=3001,
    )

    assert storage["dense_emissions"] == 360_120_000
    assert storage["dense_backpointers"] == 720_240_000
    assert storage["compact_backpointers"] == 22_526_996
    assert storage["compact_backpointers"] * 31 < storage["dense_backpointers"]


@pytest.mark.unit
def test_padded_record_batch_matches_independent_dp_and_backtraces():
    torch.manual_seed(23)
    blank_id = 4
    labels = torch.tensor([[4, 0, 4, 1, 4], [4, 2, 4, 3, 4]])
    state_lengths = torch.tensor([5, 5])
    columns = torch.tensor([[-1, 0, -1, 0, -1], [-1, 1, -1, 1, -1]])
    log_probs = torch.log_softmax(torch.randn(2, 9, blank_id + 1), dim=-1)
    speaker_probs = torch.sigmoid(torch.randn(2, 9, 2))
    frame_lengths = torch.tensor([9, 7])
    aligner = MultiSpeakerSOTWordTimestampAligner()

    batched_paths, batched_scores = aligner._ctc_forced_align_batched(
        log_probs,
        labels,
        state_lengths,
        blank_id,
        columns,
        speaker_probs,
        0.25,
        frame_lengths=frame_lengths,
    )

    for index, frame_length in enumerate(frame_lengths.tolist()):
        paths, scores = aligner._ctc_forced_align_batched(
            log_probs[index, :frame_length],
            labels[index : index + 1],
            state_lengths[index : index + 1],
            blank_id,
            columns[index : index + 1],
            speaker_probs[index, :frame_length],
            0.25,
        )
        assert torch.equal(batched_paths[index], paths[0])
        assert batched_scores[index] == pytest.approx(scores[0], abs=1.0e-6)


@pytest.mark.unit
def test_diarization_timestamps_are_contiguous_10ms_activity_segments():
    labels = torch.tensor(
        [
            [True, True, False, False, True, False],
            [False, True, True, True, False, False],
            [False, False, False, False, False, False],
        ]
    )

    segments = MultiSpeakerSOTWordTimestampAligner.diarization_segments(
        labels,
        frame_seconds=0.01,
        time_offset=1.0,
        audio_duration=0.055,
    )

    assert segments == [
        {"speaker": 0, "start": 1.0, "end": 1.02},
        {"speaker": 1, "start": 1.01, "end": 1.04},
        {"speaker": 0, "start": 1.04, "end": 1.05},
    ]


@pytest.mark.unit
def test_public_word_timestamps_only_expose_word_speaker_start_and_end():
    rows = [
        {
            "word": "hello",
            "speaker_tag": 2,
            "word_index": 7,
            "turn_index": 3,
            "start": 0.16,
            "end": 0.32,
            "start_frame": 2,
            "end_frame": 3,
            "ctc_confidence": 0.9,
            "sortformer_column": 1,
            "speaker_confidence": 0.8,
            "speaker_activity_start": 0.16,
            "speaker_activity_end": 0.32,
        }
    ]

    timestamps = MultiSpeakerSOTWordTimestampAligner._public_word_timestamps(rows)

    assert timestamps == {2: [{"word": "hello", "speaker": 2, "start": 0.16, "end": 0.32}]}


@pytest.mark.parametrize("seed", range(5))
def test_search_matches_exhaustive_legal_paths(seed):
    lp = np.random.default_rng(seed).normal(size=(6, 3)).astype(np.float32)
    labels = np.array([0, 1, 0, 2, 0])
    words = np.array([-1, 0, 0, 0, -1])
    cap = 3
    best = -np.inf
    for p in itertools.product(range(5), repeat=6):
        if p[0] not in (0, 1) or p[-1] not in (3, 4):
            continue
        legal = True
        for a, b in zip(p, p[1:]):
            if b - a not in (0, 1, 2) or (b - a == 2 and (labels[b] == 0 or labels[b] == labels[a])):
                legal = False
                break
        if not legal:
            continue
        for w in set(words):
            if w < 0:
                continue
            hit = np.flatnonzero(words[list(p)] == w)
            if len(hit) and hit[-1] - hit[0] + 1 > cap:
                legal = False
        for state in (0, 4):
            if p.count(state) > cap:
                legal = False
        if legal:
            best = max(best, float(sum(lp[t, labels[s]] for t, s in enumerate(p))))
    path, score, _ = _straight_band_search(lp, labels, words, np.arange(6), 1.0, cap)
    assert score == pytest.approx(best, abs=1e-5)


def test_collapse_and_straight_band():
    ctc = np.array([[0.0, -1], [0, -1], [-1, 0], [0, -1], [0, -1], [-1, 0]])
    assert _one_pass_positions(ctc, np.array([0, 1, 1, 1, 1, 1]), max_blank_frames=1).tolist() == [1, 2, 3, 5]
    lp = np.zeros((30, 2), np.float32)
    path, score, starts = _straight_band_search(
        lp, np.array([0, 1, 0, 1, 0]), np.array([-1, 0, -1, 1, -1]), np.arange(30), 0.1, 12
    )
    assert np.array_equal(starts, np.clip(np.rint(np.linspace(0, 4, 30)).astype(int) - 1, 0, 2))


def test_original_gap_counts_against_word_cap():
    lp = np.zeros((25, 3), np.float32)
    p, score, _ = _straight_band_search(
        lp, np.array([0, 1, 0, 2, 0]), np.array([-1, 0, 0, 0, -1]), np.array([0, 24]), 1.0, 12
    )
    assert not np.isfinite(score)


@pytest.mark.unit
def test_one_pass_does_not_invoke_preliminary_alignment_or_numba(monkeypatch):
    from nemo.collections.speechlm2.parts import ctc_timestamp_utils as utils

    def forbidden(*args, **kwargs):
        raise AssertionError('One-pass backend must not call full/compiled DP')

    monkeypatch.setattr(utils, '_get_compiled_ctc_functions', forbidden, raising=False)
    monkeypatch.setattr(MultiSpeakerSOTWordTimestampAligner, '_align_stream_batch', forbidden)
    tokenizer = SimpleNamespace(text_to_ids=lambda text: [1 if w == 'a' else 2 for w in text.split()])
    aligner = MultiSpeakerSOTWordTimestampAligner(
        tokenizer=tokenizer, blank_id=3, alignment_backend='one_pass', duration_cap=12
    )
    logits = torch.full((1, 8, 4), -10.0)
    logits[:, :, 3] = 0
    logits[0, 2, 1] = 10
    logits[0, 5, 2] = 10
    prepared = aligner.prepare_alignment(logits.log_softmax(-1), None, ['a b'], audio_durations=[0.64])
    for _ in range(2):
        result = aligner.align_prepared(prepared)[0]
        assert len(result['speaker_word_timestamps'][0]) == 2
        assert result['alignment_diagnostics']['preliminary_ctc_path_scores'] == {}
        assert len(aligner.dp_diagnostics) == 1
        assert aligner.dp_diagnostics[0]['attempts'] == 1
        assert not aligner.dp_diagnostics[0]['fallback']


@pytest.mark.unit
def test_one_pass_speaker_mapping_uses_tokens_not_column_order():
    from nemo.collections.speechlm2.parts.ctc_timestamp_utils import _one_pass_speaker_mapping

    ctc = np.array([[0, 4, 0], [4, 0, 0], [0, 0, 4]], dtype=np.float32)
    streams = [{'speaker_tag': 0, 'labels': [0, 1, 0]}, {'speaker_tag': 1, 'labels': [0, 2, 0]}]
    speaker = np.array([[0, 1], [0, 0], [1, 0]], dtype=np.float32)
    assert _one_pass_speaker_mapping(ctc, streams, speaker, 0.5) == {0: 1, 1: 0}


@pytest.mark.unit
@pytest.mark.parametrize('options', [{'band_ratio': 0}, {'duration_cap': True}, {'duration_cap': 1.5}])
def test_one_pass_rejects_invalid_configuration(options):
    with pytest.raises(ValueError):
        MultiSpeakerSOTWordTimestampAligner(alignment_backend='one_pass', **options)


@pytest.mark.unit
def test_one_pass_empty_activity_has_no_implicit_fallback():
    ctc = np.zeros((10, 3), dtype=np.float32)
    positions = _one_pass_positions(ctc, np.zeros(10))
    path, score, _ = _straight_band_search(ctc, np.array([0, 1, 0]), np.array([-1, 0, -1]), positions, 0.05, 12)
    assert len(path) == 0
    assert score == -np.inf
