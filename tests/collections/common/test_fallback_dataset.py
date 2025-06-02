import torch
from nemo.collections.common.data.fallback import FallbackDataset


class DummyDataset(torch.utils.data.Dataset):
    def __getitem__(self, item):
        return item


def test_fallback_dataset():
    dset = DummyDataset()
    assert dset[0] == 0
    assert dset[None] is None

    fallback = FallbackDataset(dset)
    assert fallback[0] == 0
    assert fallback[None] == 0
    assert fallback[1] == 1
    assert fallback[None] == 1

    fallback = FallbackDataset(dset)
    assert fallback[0] == 0
    assert fallback[None] == 0
    assert fallback[None] == 0

    fallback = FallbackDataset(dset)
    assert fallback[None] is None
    assert fallback[0] == 0
    assert fallback[None] == 0
