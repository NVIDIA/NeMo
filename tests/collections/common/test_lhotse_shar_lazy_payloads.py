# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Opt-in Lhotse Shar lazy-payload mode (``shar_lazy_payloads``) through NeMo's dataloader."""

import inspect
import io
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch
from lhotse import AudioSource, CutSet, MonoCut, Recording, SupervisionSegment
from lhotse.array import TemporalArray
from lhotse.features.io import MemoryRawWriter
from lhotse.shar.writers import SharWriter
from omegaconf import OmegaConf

from nemo.collections.common.data.lhotse import get_lhotse_dataloader_from_config
from nemo.collections.common.data.lhotse.cutset import read_cutset_from_config
from nemo.collections.common.data.lhotse.dataloader import (
    LhotseDataLoadingConfig,
    make_structured_with_schema_warnings,
)

pytest.importorskip("indexed_gzip")


def _lazy_reader_available() -> bool:
    try:
        from lhotse.shar.readers.indexed import LazyIndexedSharIterator
    except ImportError:
        return False
    return "lazy" in inspect.signature(LazyIndexedSharIterator).parameters


pytestmark = pytest.mark.skipif(
    not _lazy_reader_available(), reason="Installed Lhotse's LazyIndexedSharIterator does not accept lazy="
)

SAMPLING_RATE = 16000
NUM_CUTS = 6
SHARD_SIZE = 3  # two shards: 000000 and 000001
NUM_CODEBOOKS = 4
NUM_FRAMES = 100  # 1s at frame_shift=0.01


class _Identity(torch.utils.data.Dataset):
    def __getitem__(self, cuts: CutSet) -> CutSet:
        return cuts


@pytest.fixture
def tts_shar(tmp_path) -> Path:
    """TTS-style Shar: ``recording`` + custom ``context_audio`` Recording + custom ``target_codes`` TemporalArray."""
    root = tmp_path / "tts-shar"
    root.mkdir()
    rng = np.random.default_rng(0)
    with SharWriter(
        root,
        fields={"recording": "wav", "context_audio": "wav", "target_codes": "numpy"},
        shard_size=SHARD_SIZE,
        compress_jsonl=True,
        create_index=True,
    ) as writer:
        for idx in range(NUM_CUTS):
            cut = _make_cut(
                idx,
                rng,
                custom={"context_audio": _wav_recording(f"context-{idx:04d}", rng), "target_codes": _codes(rng)},
                speakers=["agent"],
            )
            writer.write(cut)
    assert sorted(p.name for p in root.glob("cuts.*")) == [
        "cuts.000000.jsonl.gz",
        "cuts.000000.jsonl.gz.gzidx",
        "cuts.000000.jsonl.gz.idx",
        "cuts.000001.jsonl.gz",
        "cuts.000001.jsonl.gz.gzidx",
        "cuts.000001.jsonl.gz.idx",
    ]
    return root


@pytest.fixture
def duplex_shar(tmp_path) -> Path:
    """Duplex S2S Shar: ``recording`` (user) + custom ``target_audio`` Recording (agent), user/agent supervisions."""
    root = tmp_path / "duplex-shar"
    root.mkdir()
    rng = np.random.default_rng(1)
    with SharWriter(
        root,
        fields={"recording": "wav", "target_audio": "wav"},
        shard_size=SHARD_SIZE,
        compress_jsonl=True,
        create_index=True,
    ) as writer:
        for idx in range(NUM_CUTS):
            cut = _make_cut(
                idx, rng, custom={"target_audio": _wav_recording(f"target-{idx:04d}", rng)}, speakers=["user", "agent"]
            )
            writer.write(cut)
    return root


def _lazy_config(shar_path, **overrides) -> OmegaConf:
    config = {
        "shar_path": shar_path,
        "indexed": True,
        "shar_lazy_payloads": True,
        "force_finite": True,
        "shard_seed": 0,
    }
    config.update(overrides)
    return OmegaConf.create(config)


def _streaming_reference(root: Path) -> dict[str, MonoCut]:
    return {cut.id: cut for cut in CutSet.from_shar(in_dir=root, indexed=False)}


def _brace_fields(root: Path) -> dict[str, str]:
    return {
        field: f"{root}/{field}.{{000000..000001}}.{ext}"
        for field, ext in (
            ("cuts", "jsonl.gz"),
            ("recording", "tar"),
            ("context_audio", "tar"),
            ("target_codes", "tar"),
        )
    }


def _assert_lazy_pointer(source: str, field: str) -> None:
    from lhotse.shar.lazy_pointer import decode_pointer, is_shar_pointer

    assert is_shar_pointer(source), source
    tar_path, offset, end_offset = decode_pointer(source)
    assert Path(tar_path).name.startswith(f"{field}.") and tar_path.endswith(".tar"), tar_path
    assert 0 <= offset < end_offset


def _assert_tts_cut_is_lazy(cut: MonoCut) -> None:
    assert cut.recording.sources[0].type == "shar_ptr"
    _assert_lazy_pointer(cut.recording.sources[0].source, "recording")
    assert cut.has_custom("context_audio") and cut.context_audio.sources[0].type == "shar_ptr"
    _assert_lazy_pointer(cut.context_audio.sources[0].source, "context_audio")
    assert cut.has_custom("target_codes") and cut.target_codes.array.storage_type == "shar_ptr_array"
    _assert_lazy_pointer(cut.target_codes.array.storage_key, "target_codes")


def _assert_tts_payloads_match(actual: MonoCut, expected: MonoCut) -> None:
    np.testing.assert_array_equal(actual.load_audio(), expected.load_audio())
    np.testing.assert_array_equal(actual.target_codes.load(), expected.target_codes.load())
    np.testing.assert_array_equal(actual.load_custom("target_codes"), expected.load_custom("target_codes"))
    np.testing.assert_array_equal(actual.context_audio.load_audio(), expected.context_audio.load_audio())
    np.testing.assert_array_equal(actual.load_custom("context_audio"), expected.load_custom("context_audio"))
    assert actual.target_codes.load().shape == (NUM_CODEBOOKS, NUM_FRAMES)
    assert [s.speaker for s in actual.supervisions] == [s.speaker for s in expected.supervisions]


@pytest.mark.parametrize("declaration", ["directory", "brace_fields", "weighted_list"])
def test_lazy_payloads_match_streaming_reads(tts_shar, declaration):
    expected = _streaming_reference(tts_shar)
    shar_path = {
        "directory": str(tts_shar),
        "brace_fields": _brace_fields(tts_shar),
        "weighted_list": [[str(tts_shar), 1.0]],
    }[declaration]

    cuts, is_tarred = read_cutset_from_config(_lazy_config(shar_path))

    assert is_tarred
    assert cuts.is_indexed
    actual = {cut.id: cut for cut in cuts}
    assert actual.keys() == expected.keys()
    for cut_id, cut in actual.items():
        _assert_tts_cut_is_lazy(cut)
        _assert_tts_payloads_match(cut, expected[cut_id])


def test_default_keeps_eager_inlined_payloads(tts_shar):
    cuts, _ = read_cutset_from_config(
        OmegaConf.create({"shar_path": str(tts_shar), "indexed": True, "force_finite": True, "shard_seed": 0})
    )
    assert cuts.is_indexed
    seen = 0
    for cut in cuts:
        seen += 1
        assert cut.recording.sources[0].type == "memory"
        assert cut.context_audio.sources[0].type == "memory"
        assert cut.target_codes.array.storage_type == "memory_npy"
        assert isinstance(cut.recording.sources[0].source, bytes)
    assert seen == NUM_CUTS


def test_lazy_payloads_metadata_only_reads_cuts_only(tts_shar):
    cuts, is_tarred = read_cutset_from_config(_lazy_config(str(tts_shar), metadata_only=True))
    assert is_tarred
    ids = [cut.id for cut in cuts]
    assert sorted(ids) == sorted(_streaming_reference(tts_shar))
    # No tar field is opened: the Shar placeholders stay unfilled (neither inlined nor lazy pointers).
    cut = next(iter(cuts))
    assert cut.recording.sources[0].type == "shar"
    assert cut.target_codes.array.storage_type == "shar"


def test_top_level_flag_propagates_through_input_cfg_groups(tts_shar):
    config = OmegaConf.create(
        {
            "input_cfg": [
                {
                    "type": "group",
                    "weight": 1.0,
                    "input_cfg": [{"type": "lhotse_shar", "shar_path": str(tts_shar), "weight": 1.0}],
                }
            ],
            "indexed": True,
            "shar_lazy_payloads": True,
            "force_finite": True,
            "shard_seed": 0,
        }
    )
    expected = _streaming_reference(tts_shar)

    cuts, is_tarred = read_cutset_from_config(config)

    assert is_tarred
    actual = {cut.id: cut for cut in cuts}
    assert actual.keys() == expected.keys()
    for cut_id, cut in actual.items():
        _assert_tts_cut_is_lazy(cut)
        _assert_tts_payloads_match(cut, expected[cut_id])


@pytest.mark.parametrize("num_workers", [0, 2])
def test_multi_config_dataloader_shares_top_level_lazy_payloads_flag(tts_shar, num_workers):
    pytest.importorskip("torchdata.stateful_dataloader")
    sub_config = {"shar_path": str(tts_shar), "batch_size": 2, "sample_rate": SAMPLING_RATE}
    config = OmegaConf.create(
        {
            "multi_config": True,
            "indexed": True,
            "shar_lazy_payloads": True,
            "use_stateful_dataloader": True,
            "force_finite": True,
            "seed": 0,
            "shard_seed": 0,
            "shuffle": True,
            "num_workers": num_workers,
            "first": dict(sub_config),
            "second": dict(sub_config),
        }
    )
    expected = _streaming_reference(tts_shar)

    loader = get_lhotse_dataloader_from_config(config=config, global_rank=0, world_size=1, dataset=_Identity())
    try:
        batch = next(iter(loader))
        assert len(batch) == 2
        for cut in batch:
            # Lazy pointers are plain strings, so they survive pickling into worker processes
            # and the payload can still be fetched in the main process.
            _assert_tts_cut_is_lazy(cut)
            np.testing.assert_array_equal(cut.load_audio(), expected[cut.id].load_audio())
            np.testing.assert_array_equal(cut.target_codes.load(), expected[cut.id].target_codes.load())
    finally:
        iterator = getattr(loader, "_iterator", None)
        if num_workers and iterator is not None:
            iterator._shutdown_workers()


def test_s2s_duplex_reverse_role_swaps_lazy_audio_streams(duplex_shar):
    expected = _streaming_reference(duplex_shar)
    config = OmegaConf.create(
        {
            "input_cfg": [{"type": "s2s_duplex_reverse_role", "shar_path": str(duplex_shar), "weight": 1.0}],
            "indexed": True,
            "shar_lazy_payloads": True,
            "force_finite": True,
            "shard_seed": 0,
        }
    )

    cuts, is_tarred = read_cutset_from_config(config)

    assert is_tarred
    actual = {cut.id: cut for cut in cuts}
    assert actual.keys() == expected.keys()
    for cut_id, cut in actual.items():
        reference = expected[cut_id]
        assert cut.task == "s2s_duplex_reverse_role"
        assert [s.speaker for s in reference.supervisions] == ["user", "agent"]
        assert [s.speaker for s in cut.supervisions] == ["agent", "user"]
        assert cut.recording.sources[0].type == "shar_ptr"
        assert cut.target_audio.sources[0].type == "shar_ptr"
        _assert_lazy_pointer(cut.recording.sources[0].source, "target_audio")
        _assert_lazy_pointer(cut.target_audio.sources[0].source, "recording")
        np.testing.assert_array_equal(cut.load_audio(), reference.target_audio.load_audio())
        np.testing.assert_array_equal(cut.target_audio.load_audio(), reference.load_audio())
        np.testing.assert_array_equal(cut.load_custom("target_audio"), reference.load_audio())
        assert not np.array_equal(cut.load_audio(), reference.load_audio())


@pytest.mark.parametrize("indexed", [None, False])
def test_lazy_payloads_require_indexed(tts_shar, indexed):
    config = {"shar_path": str(tts_shar), "shar_lazy_payloads": True, "force_finite": True, "shard_seed": 0}
    if indexed is not None:
        config["indexed"] = indexed
    with pytest.raises(ValueError, match="shar_lazy_payloads requires indexed: true"):
        read_cutset_from_config(OmegaConf.create(config))


def test_lazy_payloads_reject_slice_length(tts_shar):
    with pytest.raises(ValueError, match="slice_length"):
        read_cutset_from_config(_lazy_config(str(tts_shar), slice_length=2))


def test_lazy_payloads_require_lhotse_with_lazy_reader(tts_shar, monkeypatch):
    class LegacyIndexedSharIterator:
        def __init__(
            self,
            fields=None,
            in_dir=None,
            *,
            shuffle=False,
            seed=42,
            split_for_dataloading=False,
            index_path=None,
            indexes_root=None,
        ):
            raise AssertionError("A reader without lazy= must not be constructed")

    monkeypatch.setattr("lhotse.shar.readers.indexed.LazyIndexedSharIterator", LegacyIndexedSharIterator)
    with pytest.raises(ImportError, match="lhotse#1598"):
        read_cutset_from_config(_lazy_config(str(tts_shar)))


def test_schema_keeps_shar_lazy_payloads():
    assert OmegaConf.structured(LhotseDataLoadingConfig).shar_lazy_payloads is False
    config = make_structured_with_schema_warnings(
        OmegaConf.create({"shar_path": "unused", "shar_lazy_payloads": True})
    )
    assert config.shar_lazy_payloads is True


def _wav_recording(recording_id: str, rng: np.random.Generator) -> Recording:
    samples = rng.uniform(-0.5, 0.5, SAMPLING_RATE).astype(np.float32)
    buffer = io.BytesIO()
    sf.write(buffer, samples, SAMPLING_RATE, format="WAV", subtype="FLOAT")
    return Recording(
        id=recording_id,
        sources=[AudioSource(type="memory", channels=[0], source=buffer.getvalue())],
        sampling_rate=SAMPLING_RATE,
        num_samples=len(samples),
        duration=len(samples) / SAMPLING_RATE,
    )


def _codes(rng: np.random.Generator) -> TemporalArray:
    codes = rng.integers(0, 1024, size=(NUM_CODEBOOKS, NUM_FRAMES), dtype=np.int32)
    return MemoryRawWriter().store_array(key="codes", value=codes, frame_shift=0.01, temporal_dim=1, start=0.0)


def _make_cut(idx: int, rng: np.random.Generator, *, custom: dict, speakers: list[str]) -> MonoCut:
    recording = _wav_recording(f"recording-{idx:04d}", rng)
    turn = recording.duration / len(speakers)
    supervisions = [
        SupervisionSegment(
            id=f"sup-{idx:04d}-{turn_idx}",
            recording_id=recording.id,
            start=turn_idx * turn,
            duration=turn,
            text=f"utterance {idx} turn {turn_idx}",
            speaker=speaker,
        )
        for turn_idx, speaker in enumerate(speakers)
    ]
    return MonoCut(
        id=f"cut-{idx:04d}",
        start=0.0,
        duration=recording.duration,
        channel=0,
        recording=recording,
        supervisions=supervisions,
        custom=custom,
    )
