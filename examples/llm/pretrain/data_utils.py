import lightning.pytorch as pl
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from nemo.collections import llm


class SquadDataModuleWithPthDataloader(llm.SquadDataModule):
    """Creates a squad dataset with a PT dataloader"""

    def _create_dataloader(self, dataset, mode, **kwargs) -> DataLoader:
        return DataLoader(
            dataset,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            collate_fn=dataset.collate_fn,
            batch_size=self.micro_batch_size,
            **kwargs,
        )


def squad(tokenizer, mbs=1, gbs=2) -> pl.LightningDataModule:
    """Instantiates a SquadDataModuleWithPthDataloader and return it

    Args:
        tokenizer (AutoTokenizer): the tokenizer to use

    Returns:
        pl.LightningDataModule: the dataset to train with.
    """
    return SquadDataModuleWithPthDataloader(
        tokenizer=tokenizer,
        seq_length=512,
        micro_batch_size=mbs,
        global_batch_size=gbs,
        num_workers=0,
        dataset_kwargs={
            "sanity_check_dist_workers": False,
            "get_attention_mask_from_fusion": True,
        },
    )
