import pytorch_lightning as pl

from nemo.collections.vlm.neva.data.mock import MockDataModule
from nemo.collections.vlm.neva.data.lazy import NevaLazyDataModule
from nemo.collections.llm.utils import factory


@factory
def mock() -> pl.LightningDataModule:
    return MockDataModule(seq_length=4096, global_batch_size=16, micro_batch_size=2)


@factory
def squad() -> pl.LightningDataModule:
    return NevaLazyDataModule(seq_length=4096, global_batch_size=16, micro_batch_size=2)


__all__ = ["mock", "lazy"]
