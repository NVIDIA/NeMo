import pytest
import torch

from nemo.collections.tts.modules.fcd_metric import FrechetCodecDistance
from nemo.collections.tts.models import AudioCodecModel


class TestFrechetCodecDistance:
    @pytest.fixture
    def device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @pytest.fixture
    def codec(self, device, scope="session"):
        return AudioCodecModel.from_pretrained("nvidia/low-frame-rate-speech-codec-22khz").to(device)

    @pytest.fixture
    def metric(self, codec, device):        
        codec_feature_dim = codec.vector_quantizer.codebook_dim
        return FrechetCodecDistance(codec=codec, feature_dim=codec_feature_dim).to(device)

    @pytest.mark.unit
    def test_same_distribution(self, metric, device, codec):
        """Test that FCD is close to zero when comparing identical distributions."""
        B, C, T = 3, codec.num_codebooks, 20
        codes = torch.randint(low=0, high=codec.codebook_size, size=(B, C, T), device=device)
        codes_len = torch.randint(low=1, high=T, size=(B,), device=device)

        # Update with same codes for both real and fake
        metric.update(codes, codes_len, is_real=True)
        metric.update(codes, codes_len, is_real=False)

        eps = 0.01
        fcd = metric.compute()
        assert fcd < eps and fcd >= 0, f"FCD value is {fcd} but should be close to 0"
        metric.reset()

    @pytest.mark.unit
    def test_different_distribution(self, metric, device, codec):
        """Test that FCD is positive when comparing different distributions."""
        B, C, T = 3, codec.num_codebooks, 20

        # Generate two different sets of codes
        codes1 = torch.randint(low=0, high=codec.codebook_size, size=(B, C, T), device=device)
        codes2 = torch.randint(low=0, high=codec.codebook_size, size=(B, C, T), device=device)
        codes_len = torch.randint(low=1, high=T, size=(B,), device=device)

        metric.update(codes1, codes_len, is_real=True)
        metric.update(codes2, codes_len, is_real=False)

        fcd = metric.compute()
        assert fcd > 0, f"FCD value is {fcd} but should be positive for different distributions"
        metric.reset()

    def test_empty_distribution(self, metric):
        """Test that computing the FCD on empty distributions returns 0."""
        fcd = metric.compute()
        assert fcd == 0.0

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.unit
    def test_gpu_compatibility(self, metric, device, codec):
        """Test that the metric works correctly on GPU."""
        assert metric.device.type == "cuda"
        B, C, T = 3, codec.num_codebooks, 20
        codes = torch.randint(low=0, high=codec.codebook_size, size=(B, C, T), device=device)
        codes_len = torch.randint(low=1, high=T, size=(B,), device=device)

        metric.update(codes, codes_len, is_real=True)
        metric.update(codes, codes_len, is_real=False)

        fcd = metric.compute()

        eps = 0.01
        assert isinstance(fcd, torch.Tensor)
        assert fcd.device.type == "cuda"
        assert fcd < eps and fcd >= 0, f"FCD value is {fcd} but should be close to 0"

    @pytest.mark.unit
    def test_update_from_audio_file(self, metric):
        """Test the update_from_audio_file method."""

        # Test with both "real" and "fake" audio files (different files)
        metric.update_from_audio_file("tests/.data/tts/mini_ljspeech/wavs/LJ019-0373.wav", is_real=True)
        metric.update_from_audio_file("tests/.data/tts/mini_ljspeech/wavs/LJ050-0234.wav", is_real=False)

        fcd = metric.compute()
        assert isinstance(fcd, torch.Tensor)
        assert fcd > 0, f"FCD value is {fcd} but should be positive given that we tested different audio files"
