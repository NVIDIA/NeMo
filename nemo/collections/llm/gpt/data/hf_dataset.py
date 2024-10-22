import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader


class HfDatasetDataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset,
        num_workers = 2,
        pin_memory = True,
        persistent_workers = True,
        micro_batch_size = 2,
        global_batch_size = 2,
        pad_token_id = 0,
    ) -> None:
        super().__init__()
        assert pad_token_id is not None

        self.dataset = dataset
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        self.micro_batch_size = micro_batch_size
        self.global_batch_size = global_batch_size
        self.pad_token_id = pad_token_id

    @staticmethod
    def collate_fn(batch, pad_token_id=0):
        def batchify(tensor):
            if tensor.ndim == 1:
                return tensor.unsqueeze_(0)
            return tensor

        def extract_key_from_dicts(batch, key):
            return list(map(lambda x: x[key], batch))

        def pad_within_micro(batch, pad_token_id):
            max_len = max(map(len, batch))
            return [
                item + [pad_token_id] * (max_len - len(item))
                for item in batch
            ]
        return {
            key: batchify(
                torch.LongTensor(
                    pad_within_micro(
                        extract_key_from_dicts(batch, key),
                        pad_token_id,
                    )
                )
            )
            for key in ['tokens', 'labels']
        }

    def train_dataloader(self, collate_fn=None):
        from nemo.lightning.data import add_megatron_sampler

        if collate_fn is None:
            collate_fn = lambda x: HfDatasetDataModule.collate_fn(x, pad_token_id=self.pad_token_id)

        return add_megatron_sampler(
            DataLoader(
                self.dataset,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                persistent_workers=self.persistent_workers,
                collate_fn=collate_fn,
            ),
            self.micro_batch_size,
            self.global_batch_size,
        )

