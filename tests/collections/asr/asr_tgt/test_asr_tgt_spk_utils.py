import json
import random

import pytest
from lhotse import CutSet
from lhotse.testing.dummies import DummyManifest

from nemo.collections.asr.parts.utils.asr_tgtspeaker_utils import (
    codec_augment,
    get_bounded_segment,
    get_separator_audio,
    mix_noise,
    rir_augment,
)


@pytest.mark.unit
def test_get_separator_audio():
    separator_audio = get_separator_audio(freq=500, sr=16000, duration=1, ratio=0.3)
    assert separator_audio.shape == (16000,)
    assert separator_audio[:4800].sum() == 0
    assert separator_audio[-4800:].sum() == 0

    # if ratio > 0.5, the separator audio should silence (all 0)
    separater_audio = get_separator_audio(freq=500, sr=16000, duration=1, ratio=random.uniform(0.5, 0.9))
    assert separater_audio.sum() == 0


@pytest.mark.unit
def test_get_bounded_segment():
    segment_start, segment_duration = get_bounded_segment(
        start_time=0, total_duration=10, min_duration=0.5, max_duration=3
    )
    assert segment_start >= 0
    assert segment_duration >= 0.5
    assert segment_duration <= 3
    assert segment_start + segment_duration <= 10

    # test edge cases
    segment_start, segment_duration = get_bounded_segment(
        start_time=0, total_duration=2, min_duration=0.5, max_duration=3
    )
    assert segment_start >= 0
    assert segment_duration >= 0.5
    assert segment_duration <= 3
    assert segment_start + segment_duration <= 2


@pytest.mark.unit
def test_augmentation():
    cuts = DummyManifest(CutSet, begin_id=0, end_id=2, with_data=True)

    # test mix_noise
    noise_manifest_path = '/home/TestData/an4_tsasr/simulated_train/tsasr_train_tiny.json'
    with open(noise_manifest_path, 'r') as f:
        noise_manifests = [json.loads(line) for line in f]
    noise_mixed_cuts = mix_noise(cuts, noise_manifests, snr=10, mix_prob=1)
    assert len(noise_mixed_cuts) == 2
    assert noise_mixed_cuts[0].duration == cuts[0].duration
    assert noise_mixed_cuts[1].duration == cuts[1].duration

    # test rir_augment
    rir_aug_cuts = rir_augment(cuts, prob=1)
    assert len(rir_aug_cuts) == 2
    assert rir_aug_cuts[0].duration == cuts[0].duration
    assert rir_aug_cuts[1].duration == cuts[1].duration

    # test codec_augment
    codec_aug_cuts = codec_augment(cuts, prob=1)
    assert len(codec_aug_cuts) == 2
    assert codec_aug_cuts[0].duration == cuts[0].duration
    assert codec_aug_cuts[1].duration == cuts[1].duration
