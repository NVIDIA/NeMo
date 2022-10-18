from typing import Any

from pyannote.metrics.diarization import DiarizationErrorRate
from torchmetrics import Metric


class DER(Metric):
    """
    A wrapper around the pyannote object to use during training/evaluation with Pytorch Lightning.
    """

    def __init__(self, collar: float = 0.25, ignore_overlap: bool = True):
        super().__init__()
        self.metric = DiarizationErrorRate(collar=2 * collar, skip_overlap=ignore_overlap)

    def update(self, reference, hypothesis) -> None:

        ref_labels = reference
        _, hyp_labels = hypothesis
        self.metric(ref_labels, hyp_labels)

    def compute(self) -> Any:
        return abs(self.metric)
