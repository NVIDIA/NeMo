from typing import Any

import torch
from pyannote.core import Annotation, Segment
from pyannote.metrics.diarization import DiarizationErrorRate
from torchmetrics import Metric


class DER(Metric):
    """
    A wrapper around the pyannote object to use during training/evaluation with Pytorch Lightning.
    """

    full_state_update: bool = False

    def __init__(
        self,
        seconds_per_frame: int,
        min_seconds_for_segment: int,
        combine_segments_seconds: int,
        threshold: float,
        collar: float = 0.0,
        ignore_overlap: bool = False,
        post_process: bool = True,
    ):
        super().__init__()
        self.seconds_per_frame = seconds_per_frame
        self.min_seconds_for_segment = min_seconds_for_segment
        self.combine_segments_seconds = combine_segments_seconds
        self.collar = collar
        self.threshold = threshold
        self.ignore_overlap = ignore_overlap
        self.post_process = post_process
        self.add_state('missed_detection', torch.tensor(0, dtype=torch.float), dist_reduce_fx='sum')
        self.add_state('confusion', torch.tensor(0, dtype=torch.float), dist_reduce_fx='sum')
        self.add_state('false_alarm', torch.tensor(0, dtype=torch.float), dist_reduce_fx='sum')
        self.add_state('total', torch.tensor(0, dtype=torch.float), dist_reduce_fx='sum')

    def annotations(self, logits: torch.Tensor):
        logits = logits > self.threshold
        logits = logits.transpose(0, 1)
        annotation = Annotation()
        start, end = None, None
        for speaker_label, speaker_decisions in zip(['A', 'B'], logits):
            for idx, speaker_spoke in enumerate(speaker_decisions):
                if speaker_spoke:
                    if start is None:
                        start = self.seconds_per_frame * idx
                    end = self.seconds_per_frame * (idx + 1)  # keep track of the final frame
                if not speaker_spoke or idx == len(speaker_decisions) - 1:
                    if start is not None and end is not None:
                        annotation[Segment(start, end)] = speaker_label
                        start, end = None, None

        if self.post_process:
            # # combine smaller segments together
            # # with same uri and modality as original
            support = annotation.empty()
            for label in annotation.labels():
                # get timeline for current label
                timeline = annotation.label_timeline(label, copy=True)
                # fill the gaps shorter than combine_segments_seconds
                timeline = timeline.support(self.combine_segments_seconds)
                # reconstruct annotation with merged tracks
                for segment in timeline.support():
                    if segment.end - segment.start >= self.min_seconds_for_segment:
                        support[segment] = label
        return annotation

    def update(self, logits, annotations) -> None:
        hyp_labels = self.annotations(logits)
        metric = DiarizationErrorRate(collar=self.collar, skip_overlap=self.ignore_overlap)
        metric(annotations, hyp_labels, detailed=True)
        results = metric[:]
        self.missed_detection += results['missed detection']
        self.confusion += results['confusion']
        self.false_alarm += results['false alarm']
        self.total += results['total']

    def compute(self) -> Any:
        return (self.missed_detection + self.confusion + self.false_alarm) / self.total
