import numpy as np
import pytest
import torch

from nemo.collections.asr.parts.utils.vad_utils import align_labels_to_frames


class TestVADUtils:
    @pytest.mark.parametrize(["logits_len", "labels_len"], [(20, 10), (20, 11), (20, 9), (10, 21), (10, 19)])
    @pytest.mark.unit
    def test_align_label_logits(self, logits_len, labels_len):
        logits = np.arange(logits_len).tolist()
        labels = np.arange(labels_len).tolist()
        labels_new = align_labels_to_frames(probs=logits, labels=labels)

        assert len(labels_new) == len(logits)
