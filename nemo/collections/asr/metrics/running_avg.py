import torch
from torchmetrics import Metric


class RunningAverage(Metric):
    """Compute running average of a loss."""

    full_state_update: bool = True

    def __init__(self):
        super().__init__()
        self.add_state('sum', default=torch.tensor(0, dtype=torch.float), dist_reduce_fx='sum', persistent=True)
        self.add_state('count', default=torch.tensor(0, dtype=torch.int), dist_reduce_fx='sum', persistent=True)

    def compute(self):
        return self.sum / self.count

    def update(self, val, n=1) -> None:
        self.sum += val
        self.count += n
