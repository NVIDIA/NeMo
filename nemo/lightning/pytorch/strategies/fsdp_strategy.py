from pytorch_lightning.strategies.fsdp import FSDPStrategy as PLFSDPStrategy
from torch.utils.data import DataLoader
from typing_extensions import override


class FSDPStrategy(PLFSDPStrategy):
    """NeMo plugin for Pytorch Lightning FSDP Strategy.

    This strategy simply uses Pytorch Lightning's FSDP strategy adding the necessary component needed to
    work with NeMo's data module.

    Args:
        **kwargs: check out arguments in
            https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.strategies.FSDPStrategy.html
    """

    @override
    def process_dataloader(self, dataloader: DataLoader) -> DataLoader:
        if self.data_sampler:
            return self.data_sampler.transform_dataloader(dataloader)

        return dataloader
