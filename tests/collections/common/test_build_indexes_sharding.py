# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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

"""Sharded (``--shard``) and parallel-validated (``--validate-workers``) runs of ``build_indexes.py``."""

import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner
from lhotse import CutSet
from lhotse.indexing import index_file_path
from lhotse.shar.writers import SharWriter
from lhotse.testing.dummies import DummyManifest
from scripts.dataloading import build_indexes

NUM_SHARDS = 4
NUM_JOBS = 2 * NUM_SHARDS  # one cuts JSONL and one recording tar per shard


@pytest.fixture
def shar_config(tmp_path) -> tuple[Path, Path]:
    """An uncompressed, unindexed 4-shard Shar dataset declared through brace patterns."""
    root = tmp_path / "shar"
    root.mkdir()
    cuts = DummyManifest(CutSet, begin_id=0, end_id=NUM_JOBS, with_data=True)
    for cut in cuts:
        cut.features = None
        cut.custom = None
        cut.supervisions[0].custom = None
    with SharWriter(root, fields={"recording": "wav"}, shard_size=2, compress_jsonl=False, create_index=False) as w:
        for cut in cuts:
            w.write(cut)
    last = f"{NUM_SHARDS - 1:06d}"
    config = tmp_path / "input_cfg.yaml"
    config.write_text(
        yaml.safe_dump(
            [
                {
                    "type": "lhotse_shar",
                    "shar_path": {
                        "cuts": f"{root}/cuts.{{000000..{last}}}.jsonl",
                        "recording": f"{root}/recording.{{000000..{last}}}.tar",
                    },
                    "indexed": True,
                }
            ]
        )
    )
    return root, config


def test_shards_partition_the_dry_run_job_list(shar_config, caplog):
    _, config = shar_config

    everything = _job_lines(_dry_run(config, caplog))
    first = _job_lines(_dry_run(config, caplog, "--shard", "0/2"))
    second_messages = _dry_run(config, caplog, "--shard", "1/2")
    second = _job_lines(second_messages)

    assert len(everything) == NUM_JOBS
    assert len(first) == len(second) == NUM_SHARDS
    assert set(first).isdisjoint(second)
    assert sorted(first + second) == sorted(everything)
    # Interleaved selection: each shard gets JSONLs and tars, not one kind only.
    assert any("[jsonl]" in line for line in first) and any("[wds_tar]" in line for line in first)
    assert f"Shard 1/2 selected {NUM_SHARDS} of {NUM_JOBS} jobs." in second_messages


@pytest.mark.parametrize("shard", ["2/2", "a/b", "0/0", "1", "-1/2", "0/2/3"])
def test_invalid_shard_is_a_usage_error(tmp_path, shard):
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text("{}\n")
    config = tmp_path / "input.yaml"
    config.write_text(yaml.safe_dump({"type": "nemo", "manifest_filepath": str(manifest)}))

    result = CliRunner().invoke(build_indexes.main, ["--dry-run", "--shard", shard, str(config)])

    assert result.exit_code == 2, result.output
    assert "Invalid value for '--shard'" in result.output
    assert not Path(f"{manifest}.idx").exists()


def test_sharded_builds_cover_every_sidecar_then_revalidate_in_parallel(shar_config, tmp_path, caplog, monkeypatch):
    root, config = shar_config
    mirror = tmp_path / "mirror"
    expected = {
        Path(index_file_path(str(path), str(mirror)))
        for pattern in ("cuts.*.jsonl", "recording.*.tar")
        for path in root.glob(pattern)
    }
    assert len(expected) == NUM_JOBS

    # (c) Two array tasks, each building its own interleaved half under the mirror.
    result = CliRunner().invoke(
        build_indexes.main, ["--workers", "2", "--indexes-root", str(mirror), "--shard", "0/2", str(config)]
    )
    assert result.exit_code == 0, result.output
    assert {path.name for path in mirror.rglob("*.idx")} == {
        "cuts.000000.jsonl.idx",
        "cuts.000002.jsonl.idx",
        "recording.000000.tar.idx",
        "recording.000002.tar.idx",
    }
    result = CliRunner().invoke(
        build_indexes.main, ["--workers", "2", "--indexes-root", str(mirror), "--shard", "1/2", str(config)]
    )
    assert result.exit_code == 0, result.output
    assert set(mirror.rglob("*.idx")) == expected
    assert not list(root.glob("*.idx")), "sidecars must only be written under --indexes-root"

    # (d) A follow-up run validates everything on the requested thread count and builds nothing.
    pool_sizes = []

    class RecordingPool(ThreadPoolExecutor):
        def __init__(self, max_workers=None, *args, **kwargs):
            pool_sizes.append(max_workers)
            super().__init__(max_workers, *args, **kwargs)

    def refuse_to_build(job, *, force):
        raise AssertionError(f"nothing should be rebuilt, got {job}")

    monkeypatch.setattr(build_indexes, "ThreadPoolExecutor", RecordingPool)
    monkeypatch.setattr(build_indexes, "_build_one", refuse_to_build)
    caplog.clear()
    with caplog.at_level(logging.INFO):
        result = CliRunner().invoke(
            build_indexes.main, ["--validate-workers", "3", "--indexes-root", str(mirror), str(config)]
        )
    assert result.exit_code == 0, result.output
    assert f"Discovered {NUM_JOBS} files ({NUM_JOBS} already indexed, 0 to build)." in caplog.text
    assert f"validated {NUM_JOBS}/{NUM_JOBS}  pending=0" in caplog.text
    assert pool_sizes == [3]


def test_validation_reports_progress_while_a_check_is_blocked(shar_config, monkeypatch, caplog):
    _, config = shar_config
    reported = threading.Event()
    validated = []

    class ProgressHandler(logging.Handler):
        def emit(self, record):
            if f"validated 0/{NUM_JOBS}" in record.getMessage():
                reported.set()

    def is_indexed(job):
        assert reported.wait(2), "No progress was logged while a validation was unfinished"
        validated.append(job)
        return False

    monkeypatch.setattr(build_indexes, "_PROGRESS_INTERVAL", 0.02, raising=False)
    monkeypatch.setattr(build_indexes, "_is_indexed", is_indexed)
    handler = ProgressHandler()
    logger = logging.getLogger()
    logger.addHandler(handler)
    try:
        with caplog.at_level(logging.INFO):
            result = CliRunner().invoke(build_indexes.main, ["--dry-run", "--validate-workers", "1", str(config)])
    finally:
        logger.removeHandler(handler)

    assert result.exit_code == 0, result.output
    assert len(validated) == NUM_JOBS
    assert f"validated {NUM_JOBS}/{NUM_JOBS}  pending=0" in caplog.text
    assert f"Discovered {NUM_JOBS} files (0 already indexed, {NUM_JOBS} to build)." in caplog.text


def test_find_unindexed_preserves_job_order(monkeypatch):
    jobs = [build_indexes.IndexJob(f"/data/{i}.jsonl", build_indexes.JSONL) for i in range(10)]

    def is_indexed(job):
        number = int(Path(job.path).stem)
        time.sleep(0.005 * (len(jobs) - number))  # complete in reverse submission order
        return number % 3 == 0

    monkeypatch.setattr(build_indexes, "_is_indexed", is_indexed)

    todo = build_indexes._find_unindexed(jobs, workers=4)

    assert todo == [job for job in jobs if int(Path(job.path).stem) % 3 != 0]
    assert build_indexes._find_unindexed([], workers=4) == []


def test_interrupted_validation_cancels_pending_checks(monkeypatch):
    jobs = [build_indexes.IndexJob(f"/data/{i}.jsonl", build_indexes.JSONL) for i in range(8)]
    calls = []

    def is_indexed(job):
        calls.append(job)
        if len(calls) == 1:
            raise KeyboardInterrupt
        time.sleep(0.5)
        return True

    monkeypatch.setattr(build_indexes, "_is_indexed", is_indexed)
    started = time.monotonic()

    with pytest.raises(KeyboardInterrupt):
        build_indexes._find_unindexed(jobs, workers=1)

    # Without cancel_futures the pool would run the seven remaining 0.5 s checks before re-raising.
    assert len(calls) <= 2
    assert time.monotonic() - started < 2.0


def _dry_run(config: Path, caplog, *extra_args: str) -> list[str]:
    """Run ``build_indexes.py --dry-run`` with *extra_args* and return the logged messages."""
    caplog.clear()
    with caplog.at_level(logging.INFO):
        result = CliRunner().invoke(build_indexes.main, ["--dry-run", *extra_args, str(config)])
    assert result.exit_code == 0, result.output
    return list(caplog.messages)


def _job_lines(messages: list[str]) -> list[str]:
    """The ``  [kind] source -> sidecar`` lines that a dry run prints for every job it would build."""
    return [message for message in messages if message.startswith("  [")]
