import torch

from nemo.collections.tts.models.inpainting import InpaintingMSELoss
from pytest import approx
import pytest


class TestInpainting:
    @pytest.mark.unit
    def test_inpainting_loss_simple(self):
        spec_reference = torch.tensor([
            [
                [1, 1, 1],
                [1, 1, 1],
                [1, 1, 1],
                [1, 1, 1],
                [1, 1, 1],
            ]
        ], dtype=torch.float32)

        spec_predicted = torch.tensor([
            [
                [1, 1, 1],
                [1, 1, 1],
                [1, 0, -1],
                [1, 0, 1],
                [1, 1, 1],
            ]
        ], dtype=torch.float32)

        spec_mask = torch.tensor([
            [
                [1, 1, 1],
                [1, 1, 1],
                [0, 0, 0],  # the two 0 rows are where we are calculating loss
                [0, 0, 0],
                [1, 1, 1],
            ]
        ], dtype=torch.float32)

        loss_fn = InpaintingMSELoss()

        output = loss_fn(
            spect_predicted=spec_predicted.transpose(1, 2),
            spect_tgt=spec_reference.transpose(1, 2),
            spect_mask=spec_mask.transpose(1, 2)
        )

        # loss should be:
        # (1 - 1)^2 + (1 - 0)^2 + (-1 - 1)^2 +
        # (1 - 1)^2 + (1 - 0)^2 + (1 - 1)^ 2
        # / 2
        #
        # = (0 + 1 + 4) + (0 + 1 + 0) / 2
        #
        # = 6 / 1 = 6
        assert output == approx(6)
