# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
TTS-shaped end-to-end regression test for the indexed + lazy + resumable multi-config Lhotse dataloader.

On synthetic local data this reproduces the shape of the EasyMagpie multiturn pretraining dataloader
(``examples/tts/conf/magpietts/easy_magpietts_lhotse_multiturn.yaml`` plus the production SLURM overrides):
three sub-samplers (``tts_data`` / ``duplex_data`` / ``davidai_user_assistent``) fused with
``randomized_round_robin``, production-style blend YAMLs (outer group file -> nested leaf file, dict-form
``shar_path`` with brace patterns, weights, ``tokenizer_names`` tags, the davidai shards listed once as
``lhotse_shar`` and once as ``s2s_duplex_reverse_role``), gzip cuts shards indexed into an ``indexes_root``
mirror by ``scripts/dataloading/build_indexes.py``, ``StatefulDataLoader`` checkpoints and the opt-in
``shar_lazy_payloads`` mode.

What is proven:

1. every source is indexed and graph-restorable, and the dataloader never touches the streaming Shar reader
   nor builds an index at runtime (the prebuilt mirror is complete);
2. resuming from a ``StatefulDataLoader`` ``state_dict`` reproduces each sub-sampler's batch sequence exactly
   (per DataLoader worker), with filters, truncation and the reverse-role transform in the graph;
3. lazy Shar pointers resolve to the same payloads as streaming reads for the fields the TTS dataset loads;
4. the sampler checkpoint survives ``torch.save`` / ``torch.load`` (the Lightning checkpoint path), and
   indexed sources partition disjointly across DP ranks.

Fused batch order after a resume is recorded as a test property rather than asserted: Lhotse's
``RoundRobinSampler`` re-creates its numpy RNG in ``__iter__`` and does not checkpoint it, so the order in
which the (individually exact) sub-samplers are interleaved may legitimately differ from an uninterrupted run.
"""

import hashlib
import inspect
import io
import os
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch
import yaml
from click.testing import CliRunner
from lhotse import AudioSource, CutSet, MonoCut, Recording, SupervisionSegment
from lhotse.array import TemporalArray
from lhotse.dataset.dataloading import LHOTSE_USE_WORKER_PARTITION
from lhotse.features.io import MemoryRawWriter
from lhotse.indexing import gzip_index_file_path, index_file_path
from lhotse.lazy import get_graph_origin
from lhotse.shar.writers import SharWriter
from omegaconf import DictConfig, OmegaConf
from scripts.dataloading import build_indexes

from nemo.collections.common.data.lhotse import get_lhotse_dataloader_from_config
from nemo.collections.common.data.lhotse.cutset import read_cutset_from_config, s2s_duplex_reverse_role_for_one_cut
from nemo.collections.common.data.lhotse.dataloader import make_structured_with_schema_warnings

pytest.importorskip("indexed_gzip")
pytest.importorskip("torchdata.stateful_dataloader")

SAMPLE_RATE = 16000
SHARD_SIZE = 4
CODES_FPS = 25  # production codes are 25 fps (codes_25fpsSpectralCodecBWE)
NUM_CODEBOOKS = 8
SINGLE_CUTS = 24
MULTITURN_CUTS = 12
TOKENIZER = "nemotron_nano_30b"
TRUNCATE_DURATION = 18.0
# "single" cuts rejected by the TTS filters (max_cer=0.03, min_context_speaker_similarity=0.6).
BAD_CER_IDX = frozenset({3, 10, 17})
BAD_SIMILARITY_IDX = frozenset({5, 12})
# "davidai" cuts whose agent supervision carries a speaker_id listed in the excluded-speakers YAML.
EXCLUDED_SPEAKER_IDS = {
    2: "0f1e2d3c-4b5a-5968-8776-000000000002",
    9: "0f1e2d3c-4b5a-5968-8776-000000000009",
}
DECOY_SPEAKER_IDS = ["0f1e2d3c-4b5a-5968-8776-00000000ffff", "0f1e2d3c-4b5a-5968-8776-00000000fffe"]
SUB_CONFIG_NAMES = ("tts_data", "duplex_data", "davidai_user_assistent")
SOURCE_OF_SUB_CONFIG = {"tts_data": "single", "duplex_data": "gretel", "davidai_user_assistent": "davidai"}
K_PREFIX_BATCHES = 6
M_RESUME_BATCHES = 8
MAX_EXTRA_UNINTERRUPTED_BATCHES = 128
LOAD_BATCHES = 3  # per dataloading process: how many batches have their payloads loaded and digested
LOAD_ALL = 10**9


def _lazy_reader_available() -> bool:
    try:
        from lhotse.shar.readers.indexed import LazyIndexedSharIterator
    except ImportError:
        return False
    return "lazy" in inspect.signature(LazyIndexedSharIterator).parameters


LAZY_PARAMS = [
    pytest.param(False, id="eager"),
    pytest.param(
        True,
        id="lazy",
        marks=pytest.mark.skipif(
            not _lazy_reader_available(), reason="Installed Lhotse's LazyIndexedSharIterator does not accept lazy="
        ),
    ),
]


@dataclass
class _Blend:
    """Synthetic production-shaped data: Shar dirs, blend YAMLs, the index mirror and streaming references."""

    mirror: Path
    shar_roots: dict[str, Path]  # "single" | "gretel" | "davidai" -> Shar dir
    outer_yamls: dict[str, Path]  # "single" | "gretel" | "davidai" -> outer group YAML
    excluded_yaml: Path
    reference: dict[str, dict[str, MonoCut]]  # source -> cut id -> cut read by the streaming Shar reader

    @property
    def surviving_single_ids(self) -> set[str]:
        rejected = BAD_CER_IDX | BAD_SIMILARITY_IDX
        return {f"single-{idx:04d}" for idx in range(SINGLE_CUTS) if idx not in rejected}

    @property
    def surviving_davidai_indices(self) -> set[int]:
        return {idx for idx in range(MULTITURN_CUTS) if idx not in EXCLUDED_SPEAKER_IDS}


class _MagpieProbeDataset(torch.utils.data.Dataset):
    """
    Stand-in for ``MagpieTTSDatasetLhotseMultiturn``: returns the metadata needed to identify every sampled
    cut and, for the first ``load_batches`` batches of each dataloading process, SHA-1 digests of the payloads
    the TTS dataset loads (``load_audio()``, ``target_codes.load()``, ``context_codes.load()``,
    ``context_audio.load_audio()``, ``load_custom("target_audio")``).
    """

    def __init__(self, *, lazy: bool, load_batches: int = LOAD_BATCHES) -> None:
        self.lazy = lazy
        self.load_batches = load_batches
        self._batches_seen = 0

    def __getitem__(self, cuts: CutSet) -> dict:
        worker_info = torch.utils.data.get_worker_info()
        load_payloads = self._batches_seen < self.load_batches
        self._batches_seen += 1
        items = []
        for cut in cuts:
            items.append(
                {
                    "id": cut.id,
                    "recording_id": cut.recording.id,
                    "task": str(getattr(cut, "task", "tts")),
                    "lang": getattr(cut, "lang", None),
                    "tokenizer_names": list(getattr(cut, "tokenizer_names", None) or []),
                    "start": float(cut.start),
                    "duration": float(cut.duration),
                    "has_graph_origin": get_graph_origin(cut) is not None,
                    "digests": _payload_digests(cut, lazy=self.lazy) if load_payloads else None,
                }
            )
        return {"worker_id": -1 if worker_info is None else worker_info.id, "items": items}


@pytest.fixture(scope="session")
def blend(tmp_path_factory) -> _Blend:
    root = tmp_path_factory.mktemp("magpie_multiturn")
    rng = np.random.default_rng(2026)
    shar_roots = {
        "single": _write_shar(
            root / "data" / "single" / "shar",
            [_single_cut(idx, rng) for idx in range(SINGLE_CUTS)],
            fields={"recording": "wav", "context_audio": "wav", "target_codes": "numpy", "context_codes": "numpy"},
        ),
        "gretel": _write_shar(
            root / "data" / "gretel" / "shar",
            [_gretel_cut(idx, rng) for idx in range(MULTITURN_CUTS)],
            fields={"recording": "wav", "target_audio": "wav", "context_audio": "wav", "user_context_audio": "wav"},
        ),
        "davidai": _write_shar(
            root / "data" / "davidai" / "shar",
            [_davidai_cut(idx, rng) for idx in range(MULTITURN_CUTS)],
            fields={"recording": "wav", "target_audio": "wav"},
        ),
    }
    # Streaming references are read before any test forbids the streaming reader.
    reference = {name: _streaming_reference(path) for name, path in shar_roots.items()}

    blends = root / "blends"
    blends.mkdir()
    excluded_yaml = blends / "test_multiturn_speakers.yaml"
    excluded_yaml.write_text(yaml.safe_dump(list(EXCLUDED_SPEAKER_IDS.values()) + DECOY_SPEAKER_IDS))
    outer_yamls = _write_blend_yamls(blends, shar_roots)

    mirror = root / "indexes_root"
    result = CliRunner().invoke(
        build_indexes.main,
        ["--force", "--workers", "1", "--indexes-root", str(mirror), *(str(p) for p in outer_yamls.values())],
    )
    assert result.exit_code == 0, result.output
    return _Blend(
        mirror=mirror,
        shar_roots=shar_roots,
        outer_yamls=outer_yamls,
        excluded_yaml=excluded_yaml,
        reference=reference,
    )


def test_index_mirror_covers_every_shard_without_touching_the_data(blend):
    expected_tar_indexes = expected_cuts_indexes = 0
    for root in blend.shar_roots.values():
        cuts_shards = sorted(root.glob("cuts.*.jsonl.gz"))
        tars = sorted(root.glob("*.tar"))
        assert cuts_shards and tars
        for path in cuts_shards:
            assert Path(index_file_path(path, blend.mirror)).is_file(), path
            assert Path(gzip_index_file_path(path, blend.mirror)).is_file(), path
        for path in tars:
            assert Path(index_file_path(path, blend.mirror)).is_file(), path
        # Production data is read-only (AIStore): nothing may be written next to the shards.
        assert not list(root.glob("*.idx")) and not list(root.glob("*.gzidx"))
        expected_cuts_indexes += len(cuts_shards)
        expected_tar_indexes += len(tars)
    # Exactly one sidecar per shard: the davidai shards are declared twice (lhotse_shar + reverse role).
    assert len(list(blend.mirror.rglob("*.idx"))) == expected_cuts_indexes + expected_tar_indexes == 54
    assert len(list(blend.mirror.rglob("*.gzidx"))) == expected_cuts_indexes == 12


@pytest.mark.parametrize("lazy", LAZY_PARAMS)
@pytest.mark.parametrize("num_workers", [0, 2])
def test_multi_config_resume_is_exact_per_sub_sampler(blend, monkeypatch, record_property, num_workers, lazy):
    _forbid_streaming_reader_and_runtime_indexing(monkeypatch)
    config = _multi_config(blend, num_workers=num_workers, lazy=lazy)

    # (1) Every source is indexed and graph-restorable, which is what the O(1) bucketing checkpoint needs.
    for name in SUB_CONFIG_NAMES:
        cuts, is_tarred = read_cutset_from_config(_materialized_sub_config(config, name))
        assert is_tarred, name
        assert cuts.is_indexed, name
        assert cuts.has_constant_time_access, name

    loaders = []
    try:
        with _dp_env(rank=0, world_size=1):
            uninterrupted = _make_loader(config, _MagpieProbeDataset(lazy=lazy))
            loaders.append(uninterrupted)
            stream = iter(uninterrupted)
            prefix = [next(stream) for _ in range(K_PREFIX_BATCHES)]
            checkpoint = uninterrupted.state_dict()

            # (3) The checkpoint survives torch.save / torch.load exactly (Lightning loads with weights_only=False).
            restored_checkpoint = _torch_roundtrip(checkpoint)
            _assert_same_state(restored_checkpoint, checkpoint)
            sampler_state = uninterrupted.dataset.sampler.state_dict()
            _assert_same_state(_torch_roundtrip(sampler_state), sampler_state)

            # (4) Graph-origin tokens survive filters / truncate / reverse-role: every sub-sampler that already
            # yielded a batch carries the O(1) DynamicBucketer restore state (Lhotse raises otherwise).
            streams_in_prefix = {_stream_key(batch) for batch in prefix}
            sub_states = [sub for state in _sampler_states(checkpoint) for sub in state["samplers"]]
            with_bucketer_state = [sub for sub in sub_states if "bucketer_state" in sub and "rng_state" in sub]
            assert len(with_bucketer_state) == len(streams_in_prefix), (len(with_bucketer_state), streams_in_prefix)
            assert all(
                any(len(tokens) for tokens in sub["bucketer_state"]["bucket_tokens"]) for sub in with_bucketer_state
            )

            resumed_loader = _make_loader(config, _MagpieProbeDataset(lazy=lazy))
            loaders.append(resumed_loader)
            resumed_loader.load_state_dict(restored_checkpoint)
            resumed_stream = iter(resumed_loader)
            resumed = [next(resumed_stream) for _ in range(M_RESUME_BATCHES)]
            expected = [next(stream) for _ in range(M_RESUME_BATCHES)]

            # (2) Compare the fused order first; it is a recorded observation, not an assertion (see module doc).
            fused_identical = [_batch_signature(b) for b in resumed] == [_batch_signature(b) for b in expected]
            record_property("fused_batch_order_identical_after_resume", fused_identical)

            # Per (worker, sub-sampler) stream the resumed sequence must be a prefix of the uninterrupted one.
            # Extend the uninterrupted run until every stream has at least as many batches as the resumed run.
            resumed_streams = _group_by_stream(resumed)
            uninterrupted_batches = list(expected)
            for _ in range(MAX_EXTRA_UNINTERRUPTED_BATCHES):
                uninterrupted_streams = _group_by_stream(uninterrupted_batches)
                if all(len(uninterrupted_streams.get(k, ())) >= len(v) for k, v in resumed_streams.items()):
                    break
                uninterrupted_batches.append(next(stream))
            else:
                pytest.fail(f"Uninterrupted run did not cover the resumed streams: {resumed_streams.keys()}")
    finally:
        _shutdown(loaders)

    uninterrupted_streams = _group_by_stream(uninterrupted_batches)
    matched_recording_ids = set()
    for key, resumed_sequence in resumed_streams.items():
        matched = uninterrupted_streams[key][: len(resumed_sequence)]
        assert matched == resumed_sequence, f"stream {key} diverged after resume"
        matched_recording_ids.update(sig[0] for batch in matched for sig in batch)
    record_property("resumed_batches_per_stream", {str(k): len(v) for k, v in resumed_streams.items()})
    # (2b) A cut seen both before the checkpoint and after the resume is one the uninterrupted run repeats too.
    prefix_ids = {item["recording_id"] for batch in prefix for item in batch["items"]}
    resumed_ids = {item["recording_id"] for batch in resumed for item in batch["items"]}
    assert prefix_ids & resumed_ids <= matched_recording_ids

    all_items = [item for batch in (*prefix, *uninterrupted_batches, *resumed) for item in batch["items"]]
    _assert_filters_and_tags(all_items)
    assert len(all_items) >= 3 * (K_PREFIX_BATCHES + M_RESUME_BATCHES)
    assert {_source_of(item) for item in all_items} >= {"single"}
    _assert_payload_digests(blend, all_items, min_items=3)


@pytest.mark.parametrize("lazy", LAZY_PARAMS)
@pytest.mark.parametrize("name", ["duplex_data", "davidai_user_assistent"])
def test_multiturn_sources_full_epoch_filters_truncation_and_reverse_role(
    blend, monkeypatch, record_property, name, lazy
):
    _forbid_streaming_reader_and_runtime_indexing(monkeypatch)
    config = _finite_sub_config(blend, name, lazy=lazy)
    loaders = []
    try:
        with _dp_env(rank=0, world_size=1):
            loader = _make_loader(config, _MagpieProbeDataset(lazy=lazy, load_batches=LOAD_ALL))
            loaders.append(loader)
            items = [item for batch in loader for item in batch["items"]]
    finally:
        _shutdown(loaders)

    _assert_filters_and_tags(items)
    _assert_payload_digests(blend, items, min_items=len(items))
    recording_ids = [item["recording_id"] for item in items]
    assert len(recording_ids) == len(set(recording_ids)), "a finite epoch must yield every surviving cut once"
    if name == "duplex_data":
        assert set(recording_ids) == {f"gretel-rec-{idx:04d}" for idx in range(MULTITURN_CUTS)}
        assert max(item["duration"] for item in items) > 14.0, "long duplex cuts must not be dropped by bucketing"
    else:
        survivors = blend.surviving_davidai_indices
        assert set(recording_ids) == {f"davidai-rec-{idx:04d}" for idx in survivors} | {
            f"davidai-tgt-{idx:04d}" for idx in survivors
        }
        truncated = [item for item in items if item["duration"] == TRUNCATE_DURATION]
        assert truncated and all(item["start"] == 0.0 for item in truncated)
        # Truncation mints new (random) cut ids; recording ids remain the stable identity of a cut.
        assert all(
            item["id"] != item["recording_id"].replace("-rec-", "-").replace("-tgt-", "-") for item in truncated
        )
        reverse_role_tasks = {item["task"] for item in items if "-tgt-" in item["recording_id"]}
        record_property("reverse_role_task_tag", sorted(reverse_role_tasks))


@pytest.mark.parametrize("lazy", LAZY_PARAMS)
def test_tts_data_partitions_disjointly_across_two_ranks(blend, monkeypatch, lazy):
    _forbid_streaming_reader_and_runtime_indexing(monkeypatch)
    config = _finite_sub_config(blend, "tts_data", lazy=lazy)
    world_size = 2
    per_rank: list[list[dict]] = []
    for rank in range(world_size):
        loaders = []
        try:
            with _dp_env(rank=rank, world_size=world_size):
                loader = _make_loader(
                    config,
                    _MagpieProbeDataset(lazy=lazy, load_batches=LOAD_ALL),
                    global_rank=rank,
                    world_size=world_size,
                )
                loaders.append(loader)
                per_rank.append([item for batch in loader for item in batch["items"]])
        finally:
            _shutdown(loaders)

    ids_per_rank = [[item["id"] for item in items] for items in per_rank]
    for ids in ids_per_rank:
        assert ids, "every rank must receive a share of the data"
        assert len(ids) == len(set(ids)), "a finite epoch must yield every cut once per rank"
    assert set(ids_per_rank[0]).isdisjoint(ids_per_rank[1])
    assert set(ids_per_rank[0]) | set(ids_per_rank[1]) == blend.surviving_single_ids
    for items in per_rank:
        _assert_filters_and_tags(items)
        _assert_payload_digests(blend, items, min_items=len(items))


# ----------------------------------------------------------------------------------------------------------------------
# Synthetic data
# ----------------------------------------------------------------------------------------------------------------------


def _wav_recording(recording_id: str, duration: float, rng: np.random.Generator) -> Recording:
    num_samples = int(round(duration * SAMPLE_RATE))
    samples = rng.uniform(-0.5, 0.5, num_samples).astype(np.float32)
    buffer = io.BytesIO()
    sf.write(buffer, samples, SAMPLE_RATE, format="WAV", subtype="FLOAT")
    return Recording(
        id=recording_id,
        sources=[AudioSource(type="memory", channels=[0], source=buffer.getvalue())],
        sampling_rate=SAMPLE_RATE,
        num_samples=num_samples,
        duration=num_samples / SAMPLE_RATE,
    )


def _codes(key: str, duration: float, rng: np.random.Generator) -> TemporalArray:
    frames = max(1, int(round(duration * CODES_FPS)))
    codes = rng.integers(0, 1024, size=(NUM_CODEBOOKS, frames), dtype=np.int32)
    return MemoryRawWriter().store_array(key=key, value=codes, frame_shift=1 / CODES_FPS, temporal_dim=1, start=0.0)


def _turns(
    recording: Recording, speakers: list[str], speaker_ids: list[str] | None = None
) -> list[SupervisionSegment]:
    turn = round(recording.duration / len(speakers), 2)
    supervisions = []
    for turn_idx, speaker in enumerate(speakers):
        custom = {"speaker_id": speaker_ids[turn_idx]} if speaker_ids is not None else None
        supervisions.append(
            SupervisionSegment(
                id=f"{recording.id}-turn{turn_idx}",
                recording_id=recording.id,
                start=turn_idx * turn,
                duration=turn if turn_idx < len(speakers) - 1 else recording.duration - turn_idx * turn,
                text=f"{recording.id} turn {turn_idx}",
                language="en",
                speaker=speaker,
                custom=custom,
            )
        )
    return supervisions


def _single_cut(idx: int, rng: np.random.Generator) -> MonoCut:
    """Singleturn TTS cut: recording + context_audio Recordings, target_codes/context_codes TemporalArrays."""
    duration = round(1.0 + 5.0 * idx / (SINGLE_CUTS - 1), 2)
    recording = _wav_recording(f"single-rec-{idx:04d}", duration, rng)
    # SharWriter stores custom recordings through ``cut.load_custom`` (the cut window), so they span the cut.
    context_audio = _wav_recording(f"single-ctx-{idx:04d}", duration, rng)
    supervision = SupervisionSegment(
        id=f"{recording.id}-sup",
        recording_id=recording.id,
        start=0.0,
        duration=recording.duration,
        text=f"single utterance {idx}",
        language="en",
        speaker="agent",
        custom={
            "cer": 0.08 if idx in BAD_CER_IDX else 0.01,
            "context_speaker_similarity": 0.4 if idx in BAD_SIMILARITY_IDX else 0.9,
        },
    )
    return MonoCut(
        id=f"single-{idx:04d}",
        start=0.0,
        duration=recording.duration,
        channel=0,
        recording=recording,
        supervisions=[supervision],
        custom={
            "context_audio": context_audio,
            "target_codes": _codes("target_codes", recording.duration, rng),
            "context_codes": _codes("context_codes", context_audio.duration, rng),
            "tokenizer_names": [TOKENIZER],
        },
    )


def _gretel_cut(idx: int, rng: np.random.Generator) -> MonoCut:
    """Multiturn duplex cut: user recording, agent target_audio, two context Recordings, user/agent turns."""
    duration = round(4.0 + 15.5 * idx / (MULTITURN_CUTS - 1), 2)  # 4.0 .. 19.5 s
    recording = _wav_recording(f"gretel-rec-{idx:04d}", duration, rng)
    return MonoCut(
        id=f"gretel-{idx:04d}",
        start=0.0,
        duration=recording.duration,
        channel=0,
        recording=recording,
        supervisions=_turns(recording, ["user", "agent"]),
        custom={
            "target_audio": _wav_recording(f"gretel-tgt-{idx:04d}", duration, rng),
            "context_audio": _wav_recording(f"gretel-ctx-{idx:04d}", duration, rng),
            "user_context_audio": _wav_recording(f"gretel-uctx-{idx:04d}", duration, rng),
        },
    )


def _davidai_cut(idx: int, rng: np.random.Generator) -> MonoCut:
    """Multiturn duplex cut with supervision.speaker_id; two cuts carry an excluded agent speaker_id."""
    duration = round(4.0 + 26.0 * idx / (MULTITURN_CUTS - 1), 2)  # 4.0 .. 30.0 s
    recording = _wav_recording(f"davidai-rec-{idx:04d}", duration, rng)
    agent_speaker_id = EXCLUDED_SPEAKER_IDS.get(idx, f"davidai-agent-{idx:04d}")
    return MonoCut(
        id=f"davidai-{idx:04d}",
        start=0.0,
        duration=recording.duration,
        channel=0,
        recording=recording,
        supervisions=_turns(recording, ["user", "agent"], speaker_ids=[f"davidai-user-{idx:04d}", agent_speaker_id]),
        custom={"target_audio": _wav_recording(f"davidai-tgt-{idx:04d}", duration, rng)},
    )


def _write_shar(root: Path, cuts: list[MonoCut], *, fields: dict[str, str]) -> Path:
    root.mkdir(parents=True)
    # Production shards: gzip cuts JSONL, no sidecars next to the data (they live in the indexes_root mirror).
    with SharWriter(root, fields=fields, shard_size=SHARD_SIZE, compress_jsonl=True, create_index=False) as writer:
        for cut in cuts:
            writer.write(cut)
    return root


def _streaming_reference(root: Path) -> dict[str, MonoCut]:
    return {cut.id: cut for cut in CutSet.from_shar(in_dir=root, indexed=False)}


def _brace_fields(root: Path, fields: tuple[str, ...], first_shard: int, last_shard: int) -> dict[str, str]:
    shards = f"{{{first_shard:06d}..{last_shard:06d}}}"
    spec = {"cuts": f"{root}/cuts.{shards}.jsonl.gz"}
    spec.update({field: f"{root}/{field}.{shards}.tar" for field in fields})
    return spec


def _write_blend_yamls(blends: Path, shar_roots: dict[str, Path]) -> dict[str, Path]:
    """Production layout: ``train_all_*.yaml`` outer group files pointing at nested leaf files."""
    tokenizer_tags = {"tokenizer_names": [TOKENIZER]}
    leaves = {
        # Two leaves over the same dataset (shards 0-2 and 3-5) so the group multiplexes weighted indexed sources.
        "single": [
            {
                "type": "lhotse_shar",
                "shar_path": _brace_fields(
                    shar_roots["single"], ("recording", "context_audio", "target_codes", "context_codes"), 0, 2
                ),
                "weight": 2.0,
                "tags": tokenizer_tags,
            },
            {
                "type": "lhotse_shar",
                "shar_path": _brace_fields(
                    shar_roots["single"], ("recording", "context_audio", "target_codes", "context_codes"), 3, 5
                ),
                "weight": 1.0,
                "tags": tokenizer_tags,
            },
        ],
        "gretel": [
            {
                "type": "lhotse_shar",
                "shar_path": _brace_fields(
                    shar_roots["gretel"], ("recording", "target_audio", "context_audio", "user_context_audio"), 0, 2
                ),
                "weight": 525.41,
                "tags": tokenizer_tags,
            }
        ],
        # The same shards twice: once as-is and once with user/agent roles and audio streams swapped.
        "davidai": [
            {
                "type": kind,
                "shar_path": _brace_fields(shar_roots["davidai"], ("recording", "target_audio"), 0, 2),
                "weight": 15115.61,
                "tags": tokenizer_tags,
            }
            for kind in ("lhotse_shar", "s2s_duplex_reverse_role")
        ],
    }
    outer = {
        "single": ("train_all_singleturn.yaml", {"task": "tts", "lang": "en"}, 124147.25),
        "gretel": ("train_all_multiturn_gretel.yaml", {"task": "duplex", "lang": "en"}, 2347.24),
        "davidai": ("train_all_multiturn_davidai.yaml", {"task": "duplex", "lang": "en"}, 20992.81),
    }
    outer_yamls = {}
    for name, (outer_name, tags, weight) in outer.items():
        leaf_path = blends / f"train_en_{name}.yaml"
        leaf_path.write_text(yaml.safe_dump(leaves[name], sort_keys=False))
        outer_path = blends / outer_name
        outer_path.write_text(
            yaml.safe_dump(
                [{"type": "group", "weight": weight, "tags": tags, "input_cfg": str(leaf_path)}], sort_keys=False
            )
        )
        outer_yamls[name] = outer_path
    return outer_yamls


# ----------------------------------------------------------------------------------------------------------------------
# Dataloader configs (mirroring easy_magpietts_lhotse_multiturn.yaml + the production SLURM overrides)
# ----------------------------------------------------------------------------------------------------------------------


def _multi_config(blend: _Blend, *, num_workers: int, lazy: bool) -> DictConfig:
    # NOTE: with ``bucket_batch_size`` the last entry of ``bucket_duration_bins`` is the upper bound accepted by
    # ``BucketingFilter`` (``max_duration`` does not extend it), so the bins must reach the longest kept cut.
    return OmegaConf.create(
        {
            "multi_config": True,
            "shuffle": True,
            "seed": 42,
            "shard_seed": 42,
            "num_workers": num_workers,
            "pin_memory": False,
            "sampler_fusion": "randomized_round_robin",
            "sampler_weights": {"tts_data": 0.7, "duplex_data": 0.1, "davidai_user_assistent": 0.2},
            "indexed": True,
            "indexes_root": str(blend.mirror),
            "use_stateful_dataloader": True,
            "shar_lazy_payloads": lazy,
            "force_finite": False,
            "tts_data": {
                "input_cfg": str(blend.outer_yamls["single"]),
                "use_bucketing": True,
                "bucket_duration_bins": [2.0, 3.5, 5.0],
                "batch_duration": 12,
                "quadratic_duration": None,
                "bucket_buffer_size": 16,
                "shuffle_buffer_size": 16,
                "concurrent_bucketing": False,
                "max_cer": 0.03,
                "min_context_speaker_similarity": 0.6,
                "min_duration": 0.2,
                "max_duration": 30.0,
                "drop_last": True,
                "reweight_temperature": 0.7,
            },
            "duplex_data": {
                "input_cfg": str(blend.outer_yamls["gretel"]),
                "use_bucketing": True,
                "bucket_duration_bins": [8.0, 14.0, 20.0],
                "bucket_batch_size": [3, 2, 1],
                "bucket_buffer_size": 8,
                "shuffle_buffer_size": 8,
                "concurrent_bucketing": False,
                "max_duration": 300,
            },
            "davidai_user_assistent": {
                "input_cfg": str(blend.outer_yamls["davidai"]),
                "use_bucketing": True,
                "bucket_duration_bins": [8.0, 14.0, TRUNCATE_DURATION],
                "bucket_batch_size": [3, 2, 1],
                "bucket_buffer_size": 8,
                "shuffle_buffer_size": 8,
                "concurrent_bucketing": False,
                "truncate_duration": TRUNCATE_DURATION,
                "truncate_offset_type": "start",
                "max_duration": TRUNCATE_DURATION,
                "excluded_speaker_ids": str(blend.excluded_yaml),
                "speaker_filter_fields": ["speaker_id"],
            },
        }
    )


def _materialized_sub_config(config: DictConfig, name: str) -> DictConfig:
    """Replicate ``get_lhotse_dataloader_from_multi_config.gather_shared_opts`` for one sub-config."""
    sub = make_structured_with_schema_warnings(config[name])
    for key in (
        "seed",
        "shard_seed",
        "num_workers",
        "pin_memory",
        "shuffle",
        "force_finite",
        "use_stateful_dataloader",
        "indexed",
        "indexes_root",
        "shar_lazy_payloads",
    ):
        sub[key] = config[key]
    return sub


def _finite_sub_config(blend: _Blend, name: str, *, lazy: bool) -> DictConfig:
    """Single-config variant of one sub-sampler that yields every surviving cut exactly once."""
    sub = _materialized_sub_config(_multi_config(blend, num_workers=0, lazy=lazy), name)
    sub.force_finite = True
    sub.drop_last = False
    return sub


# ----------------------------------------------------------------------------------------------------------------------
# Loader helpers
# ----------------------------------------------------------------------------------------------------------------------


@contextmanager
def _dp_env(rank: int, world_size: int):
    """
    Mimic the env ``lhotse.dataset.dataloading.worker_init_fn`` sets (NeMo calls it eagerly in the main process
    for ``num_workers=0``), and restore it afterwards so tests do not leak RANK/WORLD_SIZE to each other.
    """
    keys = ("RANK", "WORLD_SIZE", LHOTSE_USE_WORKER_PARTITION, "LHOTSE_PROCESS_SEED")
    saved = {key: os.environ.get(key) for key in keys}
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ[LHOTSE_USE_WORKER_PARTITION] = "1"
    try:
        yield
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _make_loader(config: DictConfig, dataset: _MagpieProbeDataset, *, global_rank: int = 0, world_size: int = 1):
    return get_lhotse_dataloader_from_config(
        config=config, global_rank=global_rank, world_size=world_size, dataset=dataset
    )


def _shutdown(loaders) -> None:
    for loader in loaders:
        iterator = getattr(loader, "_iterator", None)
        if iterator is not None and getattr(loader, "num_workers", 0):
            iterator._shutdown_workers()


def _forbid_streaming_reader_and_runtime_indexing(monkeypatch) -> None:
    def reject_streaming(*args, **kwargs):
        raise AssertionError("The streaming LazySharIterator must not be used by an indexed dataloader")

    def reject_autobuild(*args, **kwargs):
        raise AssertionError("Runtime must use the prebuilt indexes_root sidecars instead of building indexes")

    monkeypatch.setattr("lhotse.shar.readers.lazy.LazySharIterator.__init__", reject_streaming)
    monkeypatch.setattr("lhotse.shar.readers.indexed.create_jsonl_index", reject_autobuild)
    monkeypatch.setattr("lhotse.shar.readers.indexed.create_tar_index", reject_autobuild)


def _torch_roundtrip(state):
    buffer = io.BytesIO()
    torch.save(state, buffer)
    buffer.seek(0)
    # Lightning's ``_load`` passes ``weights_only=False``; the sampler state carries RNG tuples and tokens.
    return torch.load(buffer, weights_only=False)


def _assert_same_state(actual, expected, path: str = "state") -> None:
    assert type(actual) is type(expected), (path, type(actual), type(expected))
    if isinstance(actual, dict):
        assert list(actual.keys()) == list(expected.keys()), path
        for key in actual:
            _assert_same_state(actual[key], expected[key], f"{path}.{key}")
    elif isinstance(actual, (list, tuple)):
        assert len(actual) == len(expected), path
        for idx, (a, b) in enumerate(zip(actual, expected)):
            _assert_same_state(a, b, f"{path}[{idx}]")
    elif isinstance(actual, np.ndarray):
        assert actual.dtype == expected.dtype and np.array_equal(actual, expected), path
    else:
        assert actual == expected, path


def _sampler_states(state) -> list[dict]:
    """Every ``IterableDatasetWrapper`` sampler state inside a (single- or multi-worker) loader checkpoint."""
    found = []

    def walk(obj):
        if isinstance(obj, dict):
            if isinstance(obj.get("sampler_state"), dict):
                found.append(obj["sampler_state"])
            for value in obj.values():
                walk(value)
        elif isinstance(obj, (list, tuple)):
            for value in obj:
                walk(value)

    walk(state)
    return found


# ----------------------------------------------------------------------------------------------------------------------
# Batch identity, payload digests and assertions
# ----------------------------------------------------------------------------------------------------------------------


def _source_of(item: dict) -> str:
    return item["recording_id"].split("-", 1)[0]


def _batch_signature(batch: dict) -> tuple:
    # Truncated cuts get fresh uuid ids on every iteration, so identify cuts by recording id + task + window.
    return tuple(
        (item["recording_id"], item["task"], round(item["start"], 6), round(item["duration"], 6))
        for item in batch["items"]
    )


def _stream_key(batch: dict) -> tuple[int, str]:
    sources = {_source_of(item) for item in batch["items"]}
    assert len(sources) == 1, f"a mini-batch must come from exactly one sub-sampler, got {sources}"
    return batch["worker_id"], sources.pop()


def _group_by_stream(batches: list[dict]) -> dict[tuple[int, str], list[tuple]]:
    streams: dict[tuple[int, str], list[tuple]] = {}
    for batch in batches:
        streams.setdefault(_stream_key(batch), []).append(_batch_signature(batch))
    return streams


def _digest(array: np.ndarray) -> str:
    array = np.ascontiguousarray(array)
    return hashlib.sha1(f"{array.dtype}|{array.shape}|".encode() + array.tobytes()).hexdigest()


def _payload_digests(cut: MonoCut, *, lazy: bool) -> dict[str, str]:
    """Load what the TTS multiturn dataset loads, checking the storage types the mode is expected to produce."""
    audio_type = "shar_ptr" if lazy else "memory"
    array_type = "shar_ptr_array" if lazy else "memory_npy"
    assert cut.recording.sources[0].type == audio_type, (cut.id, cut.recording.sources[0].type)
    digests = {"recording": _digest(cut.load_audio())}
    for field in ("target_codes", "context_codes"):
        if cut.has_custom(field):
            manifest = getattr(cut, field)
            assert manifest.array.storage_type == array_type, (cut.id, field, manifest.array.storage_type)
            digests[field] = _digest(manifest.load())
    if cut.has_custom("context_audio"):
        assert cut.context_audio.sources[0].type == audio_type, cut.id
        digests["context_audio"] = _digest(cut.context_audio.load_audio())
    if cut.has_custom("target_audio"):
        assert cut.target_audio.sources[0].type == audio_type, cut.id
        digests["target_audio"] = _digest(cut.load_custom("target_audio"))
    return digests


def _expected_digests(blend: _Blend, item: dict) -> dict[str, str]:
    source, kind, idx = item["recording_id"].split("-")
    reference = blend.reference[source][f"{source}-{idx}"]
    if kind == "tgt":
        # A reverse-role cut: its recording is the original target_audio and vice versa.
        reference = s2s_duplex_reverse_role_for_one_cut(
            reference, agent_roles=("agent",), user_roles=("user",), target_agent_name="agent", target_user_name="user"
        )
    if item["duration"] < reference.duration - 1e-9:
        reference = reference.truncate(offset=item["start"], duration=item["duration"], preserve_id=True)
    return _payload_digests(reference, lazy=False)


def _assert_payload_digests(blend: _Blend, items: list[dict], *, min_items: int) -> None:
    loaded = [item for item in items if item["digests"] is not None]
    assert len(loaded) >= min_items, (len(loaded), min_items)
    for item in loaded:
        assert item["digests"] == _expected_digests(blend, item), item["recording_id"]
    fields = {field for item in loaded for field in item["digests"]}
    sources = {_source_of(item) for item in loaded}
    if "single" in sources:
        assert {"recording", "context_audio", "target_codes", "context_codes"} <= fields
    if sources & {"gretel", "davidai"}:
        assert "target_audio" in fields


def _assert_filters_and_tags(items: list[dict]) -> None:
    for item in items:
        assert item["has_graph_origin"], item["recording_id"]
        assert item["lang"] == "en", item
        assert item["tokenizer_names"] == [TOKENIZER], item
        source = _source_of(item)
        idx = int(item["recording_id"].rsplit("-", 1)[1])
        if source == "single":
            assert idx not in BAD_CER_IDX and idx not in BAD_SIMILARITY_IDX, item["recording_id"]
            assert item["task"] == "tts", item
            assert 0.2 <= item["duration"] <= 30.0
        elif source == "davidai":
            assert idx not in EXCLUDED_SPEAKER_IDS, item["recording_id"]
            assert item["duration"] <= TRUNCATE_DURATION + 1e-6, item
            assert item["start"] == 0.0, item
            # Group-level ``tags.task`` is attached after the reverse-role parser; production marks both duplex.
            assert item["task"] in ("duplex", "s2s_duplex_reverse_role"), item
            if "-rec-" in item["recording_id"]:
                assert item["task"] == "duplex", item
        else:
            assert source == "gretel", item
            assert item["task"] == "duplex", item
            assert item["duration"] <= 300.0
