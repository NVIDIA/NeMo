from collections import Counter
from io import BytesIO
from itertools import islice
from pathlib import Path
from typing import Dict, List, Tuple

import lhotse
import numpy as np
import pytest
import torch
from lhotse import CutSet, MonoCut, NumpyFilesWriter, Recording, compute_num_samples
from lhotse.audio import AudioLoadingError
from lhotse.augmentation import ReverbWithImpulseResponse
from lhotse.cut import Cut, MixedCut, PaddingCut
from lhotse.dataset import RoundRobinSampler, ZipSampler
from lhotse.shar import JsonlShardWriter
from lhotse.testing.dummies import dummy_recording
from lhotse.testing.random import deterministic_rng
from omegaconf import OmegaConf

from nemo.collections.common.data.lhotse import get_lhotse_dataloader_from_config
from nemo.collections.common.data.lhotse.text_adapters import SourceTargetTextExample, TextExample
from nemo.collections.common.tokenizers.sentencepiece_tokenizer import SentencePieceTokenizer, create_spt_model


@pytest.fixture(scope="session")
def cutset_path(tmp_path_factory) -> Path:
    """10 utterances of length 1s as a Lhotse CutSet."""
    from lhotse import CutSet
    from lhotse.testing.dummies import DummyManifest

    cuts = DummyManifest(CutSet, begin_id=0, end_id=10, with_data=True)
    for c in cuts:
        c.features = None
        c.custom = None
        c.supervisions[0].custom = None

    tmp_path = tmp_path_factory.mktemp("data")
    p = tmp_path / "cuts.jsonl.gz"
    pa = tmp_path / "audio"
    cuts.save_audios(pa).to_file(p)
    return p


class Identity(torch.utils.data.Dataset):
    def __getitem__(self, item):
        return item


def test_dataloader_augconfig_examples_rir(cutset_path: Path):
    config = OmegaConf.create(
        {
            "cuts_path": str(cutset_path),
            "augment_examples": [
                {
                    "_target_": "nemo.collections.common.data.lhotse.augment.ReverbRIR",
                    "prob": 1.0,
                }
            ],
            "batch_size": 1,
            "seed": 0,
            "shard_seed": 0,
        }
    )
    dl = get_lhotse_dataloader_from_config(
        config=config,
        global_rank=0,
        world_size=1,
        dataset=Identity(),
    )
    batch = next(iter(dl))
    assert isinstance(batch, CutSet)
    assert len(batch) == 1
    cut = batch[0]
    assert isinstance(cut, MonoCut)
    assert isinstance(cut.recording.transforms, list) and len(cut.recording.transforms) == 1
    tfnm = cut.recording.transforms[0]
    if isinstance(tfnm, dict):  # lhotse<=1.23.0
        assert tfnm["name"] == "ReverbWithImpulseResponse"
    else:  # lhotse>=1.24.0
        assert isinstance(tfnm, ReverbWithImpulseResponse)


def test_dataloader_augconfig_batch_rir(cutset_path: Path):
    config = OmegaConf.create(
        {
            "cuts_path": str(cutset_path),
            "augment_batch": [
                {
                    "_target_": "nemo.collections.common.data.lhotse.augment.ReverbRIR",
                    "prob": 1.0,
                }
            ],
            "batch_size": 1,
            "seed": 0,
            "shard_seed": 0,
        }
    )
    dl = get_lhotse_dataloader_from_config(
        config=config,
        global_rank=0,
        world_size=1,
        dataset=Identity(),
    )
    batch = next(iter(dl))
    assert isinstance(batch, CutSet)
    assert len(batch) == 1
    cut = batch[0]
    assert isinstance(cut, MonoCut)
    assert isinstance(cut.recording.transforms, list) and len(cut.recording.transforms) == 1
    tfnm = cut.recording.transforms[0]
    if isinstance(tfnm, dict):  # lhotse<=1.23.0
        assert tfnm["name"] == "ReverbWithImpulseResponse"
    else:  # lhotse>=1.24.0
        assert isinstance(tfnm, ReverbWithImpulseResponse)


def test_lhotse_augconfig_perturb_speed(cutset_path: Path):
    config = OmegaConf.create(
        {
            "cuts_path": str(cutset_path),
            "augment_examples": [
                {
                    "_target_": "nemo.collections.common.data.lhotse.augment.PerturbSpeed",
                    "factors": [0.5, 1.5],
                }
            ],
            "batch_size": 16,
            "seed": 0,
            "shard_seed": 0,
        }
    )
    dl = get_lhotse_dataloader_from_config(
        config=config,
        global_rank=0,
        world_size=1,
        dataset=Identity(),
    )
    batch = next(iter(dl))
    assert isinstance(batch, CutSet)
    assert len(batch) == 16
    uniq_durs = set(c.duration for c in batch)
    assert uniq_durs == {2.0, 1.0, 0.6666875}  # 50% speed: 1s=>2s; 150% speed => 1s=>0.6666..s


def test_lhotse_augconfig_composite(cutset_path: Path):
    config = OmegaConf.create(
        {
            "cuts_path": str(cutset_path),
            "augment_examples": [
                {
                    "_target_": "nemo.collections.common.data.lhotse.augment.PerturbSpeed",
                },
                {
                    "_target_": "nemo.collections.common.data.lhotse.augment.ReverbRIR",
                    "prob": 1.0,
                },
                {
                    "_target_": "nemo.collections.common.data.lhotse.augment.MinPadding",
                    "duration": 10.0,
                },
                {
                    "_target_": "nemo.collections.common.data.lhotse.augment.Mix",
                    "noise_path": str(cutset_path),
                    "prob": 1.0,
                },
                {
                    "_target_": "nemo.collections.common.data.lhotse.augment.CutIntoWindows",
                    "duration": 3.0,
                    "hop": 2.0,
                },
                {
                    "_target_": "nemo.collections.common.data.lhotse.augment.Truncate",
                    "duration": 2.0,
                },
                {
                    "_target_": "nemo.collections.common.data.lhotse.augment.Mix",
                    "noise_path": str(cutset_path),
                    "prob": 1.0,
                },
            ],
            "batch_size": 16,
            "seed": 0,
            "shard_seed": 0,
        }
    )
    dl = get_lhotse_dataloader_from_config(
        config=config,
        global_rank=0,
        world_size=1,
        dataset=Identity(),
    )
    batch = next(iter(dl))
    assert isinstance(batch, CutSet)
    assert len(batch) == 16
    for cut in batch:
        assert 1 <= cut.duration <= 2
        assert isinstance(cut, MixedCut)
        assert len(cut.tracks) >= 3
        cut.load_audio()  # does not fail
