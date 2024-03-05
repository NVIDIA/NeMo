import numpy as np
import pytest
from lhotse.dataset import DynamicBucketingSampler, DynamicCutSampler

from nemo.collections.common.data.lhotse.text_adapters import TextExample


@pytest.fixture
def text_source():
    def get_text_source():
        while True:
            for item in ("hello world", "example text", "this is my text data"):
                # for this example, "bytes are all you need", could be BPE, etc.
                yield TextExample(item)

    return get_text_source()


def test_text_dynamic_cut_sampler_static_batch_size(text_source):
    sampler = DynamicCutSampler(text_source, max_cuts=16)
    batch = next(iter(sampler))
    assert len(batch) == 16
    assert isinstance(batch[0], TextExample)
    assert isinstance(batch[0].text, str)


def test_text_dynamic_cut_sampler_dynamic_batch_size(text_source):
    sampler = DynamicCutSampler(text_source, max_duration=256,)
    batch = next(iter(sampler))
    assert isinstance(batch[0], TextExample)
    assert isinstance(batch[0].text, str)
    assert len(batch) == 12


def test_text_dynamic_bucketing_sampler(text_source):
    sampler = DynamicBucketingSampler(text_source, max_duration=256, num_buckets=2, quadratic_duration=128)
    batch = next(iter(sampler))
    assert isinstance(batch[0], TextExample)
    assert isinstance(batch[0].text, str)
    assert len(batch) == 11
