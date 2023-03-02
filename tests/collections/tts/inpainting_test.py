import torch

from nemo.collections.tts.models.inpainting import InpaintingMSELoss, ConvUnit, random_slice_if_longer_than
from pytest import approx
import pytest
import numpy as np


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
            spect_predicted=spec_predicted,
            spect_tgt=spec_reference,
            spect_mask=spec_mask
        )

        # loss should be average loss per pixel:
        # (1 - 1)^2 + (1 - 0)^2 + (-1 - 1)^2 +
        # (1 - 1)^2 + (1 - 0)^2 + (1 - 1)^ 2
        # / 6
        #
        # = (0 + 1 + 4) + (0 + 1 + 0) / 6
        #
        # = 6 / 6 = 1
        assert output == approx(1)

    @pytest.mark.unit
    def test_discriminator(self):
        num_buckets = 32
        mel_height = 80
        c = 16

        example_input = torch.tensor(
            np.random.normal(size=(1, num_buckets, mel_height)),
            dtype=torch.float32
        )

        module = ConvUnit((1, num_buckets, mel_height), c=c, s_t=2, s_f=2)

        output = module(example_input)
        assert output.shape == torch.Size(
            [c * 2, num_buckets // 2, mel_height // 2])

    @pytest.mark.unit
    def test_slicing(self):
        a = torch.randint(4, size=(3, 15, 5))
        lens = torch.tensor([9, 15, 13])
        max_len = 10
        mels, spec_len, slice_indices = random_slice_if_longer_than(
            a, lens, max_len)

        assert mels.shape == torch.Size([3, 10, 5])
        assert spec_len.equal(torch.tensor([9, 10, 10]))

        assert slice_indices[0].equal(torch.arange(10))
