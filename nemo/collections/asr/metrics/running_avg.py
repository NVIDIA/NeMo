import torch
from torchmetrics import Metric


class RunningAverage(Metric):
    """Compute running average of a loss."""

    full_state_update: bool = False

    def __init__(self, alpha=0.98):
        super().__init__()
        assert 0.0 < alpha <= 1.0, "Argument alpha should be a float between 0.0 and 1.0"
        self.alpha = alpha
        self.add_state('value', default=torch.tensor(0), dist_reduce_fx='mean')

    def compute(self):
        return self.value

    def update(self, loss) -> None:
        self.value = self.value * self.alpha + (1.0 - self.alpha) * loss
