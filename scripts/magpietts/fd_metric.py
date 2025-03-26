# Adapted from https://github.com/pytorch/torcheval/blob/main/torcheval/metrics/image/fid.py

import warnings
from importlib.util import find_spec
from typing import Any, Iterable, Optional, TypeVar, Union

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torcheval.metrics.metric import Metric
from einops import rearrange
import soundfile as sf

from nemo.collections.tts.models import AudioCodecModel
from nemo.collections.tts.parts.utils.tts_dataset_utils import load_audio, _read_audio

# pyre-ignore-all-errors[16]: Undefined attribute of metric states.

class CodecEmbedder(nn.Module):
    def __init__(self, codec: AudioCodecModel):
        super().__init__()
        self.codec = codec

    def codes_to_emedding(self, x: Tensor, x_len: Tensor) -> Tensor:
        # x: (B, T, C)
        # x_len: (B,)
        return self.codec.dequantize(tokens=x, tokens_len=x_len)
    
    def encode_from_file(self, audio_path: str) -> Tensor:
        print(f"Encoding audio {audio_path}")
        audio_segment = _read_audio(audio_filepath=audio_path, sample_rate=self.codec.sample_rate, offset=0, duration=0)
        samples = samples = torch.tensor(audio_segment.samples, device=self.codec.device).unsqueeze(0)
        audio_len = torch.tensor(samples.shape[1], device=self.codec.device).unsqueeze(0)
        codes, codes_len = self.codec.encode(audio=samples, audio_len=audio_len)
        return codes, codes_len

class FrechetCodecDistance(Metric[torch.Tensor]):
    def __init__(
        self, 
        codec,
        feature_dim: int,
        device: Optional[torch.device] = None,
    ) -> None:
        """
        Computes the Frechet Inception Distance (FID) between two distributions of images (real and generated).

        The original paper: https://arxiv.org/pdf/1706.08500.pdf

        Args:
            model (nn.Module): Module used to compute feature activations.
                If None, a default InceptionV3 model will be used.
            feature_dim (int): The number of features in the model's output,
                the default number is 2048 for default InceptionV3.
            device (torch.device): The device where the computations will be performed.
                If None, the default device will be used.
        """
        super().__init__(device=device)

        # Set the model and put it in evaluation mode
        self.model = CodecEmbedder(codec)
        self.model = self.model.to(device)
        self.model.eval()
        self.model.requires_grad_(False)

        # Initialize state variables used to compute FID
        self._add_state("real_sum", torch.zeros(feature_dim, device=device))
        self._add_state(
            "real_cov_sum", torch.zeros((feature_dim, feature_dim), device=device)
        )
        self._add_state("fake_sum", torch.zeros(feature_dim, device=device))
        self._add_state(
            "fake_cov_sum", torch.zeros((feature_dim, feature_dim), device=device)
        )
        self._add_state("num_real_frames", torch.tensor(0, device=device).int())
        self._add_state("num_fake_frames", torch.tensor(0, device=device).int())

    @torch.inference_mode()
    def update_from_audio_file(self, audio_path: str, is_real: bool) -> Tensor:
        """
        Takes a path to an audio file, embeds it, and updates the FID metric.
        """
        codes, codes_len = self.model.encode_from_file(audio_path)
        self.update_from_codes(codes, codes_len, is_real)
    
    @torch.inference_mode()
    def update(self, codes: Tensor, codes_len: Tensor, is_real: bool):
        # alias for update_from_codes
        self.update_from_codes(codes, codes_len, is_real)
        
    @torch.inference_mode()
    def update_from_codes(self, codes: Tensor, codes_len: Tensor, is_real: bool):
        """
        Update the states with a batch of real and fake codes.
        Takes pre-computed codec codes, embeds them, and updates the FID metric.

        Args:
            codes (Tensor): A batch of codec frames of shape (B, C, T).
            is_real (Boolean): Denotes if images are real or not.
        """
        assert codes.ndim == 3
        codes = codes.to(self.device)

        # Dequantize the codes to a continuous representation
        embeddings = self.model.codes_to_emedding(codes, codes_len) # B, E, T where E is the codec's embedding dimension, usually 4*num_codebooks

        # keep only the valid frames
        valid_frames = []
        for i in range(codes.shape[0]):
            valid_frames.append(embeddings[i, :, :codes_len[i]].T) # T', E
        embeddings = torch.cat(valid_frames, dim=0) # total_valid_frames, E
        valid_frame_count = embeddings.shape[0]

        # Update the state variables used to compute FID
        if is_real:
            self.num_real_frames += valid_frame_count
            self.real_sum += torch.sum(embeddings, dim=0)
            self.real_cov_sum += torch.matmul(embeddings.T, embeddings)
        else:
            self.num_fake_frames += valid_frame_count
            self.fake_sum += torch.sum(embeddings, dim=0)
            self.fake_cov_sum += torch.matmul(embeddings.T, embeddings)

        return self

    @torch.inference_mode()
    def merge_state(
        self, metrics):
        """
        Merge the state of another FID instance into this instance.

        Args:
            metrics (Iterable[FID]): The other FID instance(s) whose state will be merged into this instance.
        """
        for metric in metrics:
            self.real_sum += metric.real_sum.to(self.device)
            self.real_cov_sum += metric.real_cov_sum.to(self.device)
            self.fake_sum += metric.fake_sum.to(self.device)
            self.fake_cov_sum += metric.fake_cov_sum.to(self.device)
            self.num_real_frames += metric.num_real_frames.to(self.device)
            self.num_fake_frames += metric.num_real_frames.to(self.device)

        return self

    @torch.inference_mode()
    def compute(self) -> Tensor:
        """
        Compute the FID.

        Returns:
            tensor: The FID.
        """

        # If the user has not already updated with at lease one
        # image from each distribution, then we raise an Error.
        if (self.num_real_frames == 0) or (self.num_fake_frames == 0):
            warnings.warn(
                "Computing FID requires at least 1 real image and 1 fake image,"
                f"but currently running with {self.num_real_frames} real images and {self.num_fake_frames} fake images."
                "Returning 0.0",
                RuntimeWarning,
            )

            return torch.tensor(0.0)

        # Compute the mean activations for each distribution
        real_mean = (self.real_sum / self.num_real_frames).unsqueeze(0)
        fake_mean = (self.fake_sum / self.num_fake_frames).unsqueeze(0)

        # Compute the covariance matrices for each distribution
        real_cov_num = self.real_cov_sum - self.num_real_frames * torch.matmul(
            real_mean.T, real_mean
        )
        real_cov = real_cov_num / (self.num_real_frames - 1)
        fake_cov_num = self.fake_cov_sum - self.num_fake_frames * torch.matmul(
            fake_mean.T, fake_mean
        )
        fake_cov = fake_cov_num / (self.num_fake_frames - 1)

        # Compute the Frechet Distance between the distributions
        fd = self._calculate_frechet_distance(
            real_mean.squeeze(), real_cov, fake_mean.squeeze(), fake_cov
        )
        # FD should be non-negative but due to numerical errors, it can be slightly negative
        # Have seen -0.0011 in the past
        assert fd >= -0.005
        return torch.max(torch.tensor(0.0), fd)

    def _calculate_frechet_distance(
        self,
        mu1: Tensor,
        sigma1: Tensor,
        mu2: Tensor,
        sigma2: Tensor,
    ) -> Tensor:
        """
        Calculate the Frechet Distance between two multivariate Gaussian distributions.

        Args:
            mu1 (Tensor): The mean of the first distribution.
            sigma1 (Tensor): The covariance matrix of the first distribution.
            mu2 (Tensor): The mean of the second distribution.
            sigma2 (Tensor): The covariance matrix of the second distribution.

        Returns:
            tensor: The Frechet Distance between the two distributions.
        """

        # Compute the squared distance between the means
        mean_diff = mu1 - mu2
        mean_diff_squared = mean_diff.square().sum(dim=-1)

        # Calculate the sum of the traces of both covariance matrices
        trace_sum = sigma1.trace() + sigma2.trace()

        # Compute the eigenvalues of the matrix product of the real and fake covariance matrices
        sigma_mm = torch.matmul(sigma1, sigma2)
        eigenvals = torch.linalg.eigvals(sigma_mm)

        # Take the square root of each eigenvalue and take its sum
        sqrt_eigenvals_sum = eigenvals.sqrt().real.sum(dim=-1)

        # Calculate the FID using the squared distance between the means,
        # the sum of the traces of the covariance matrices, and the sum of the square roots of the eigenvalues
        fid = mean_diff_squared + trace_sum - 2 * sqrt_eigenvals_sum

        return fid

    def to(
        self,
        device: Union[str, torch.device],
        *args: Any,
        **kwargs: Any,
    ):
        super().to(device=device)
        self.model.to(self.device)
        return self


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    codec_path = "/datap/misc/checkpoints/AudioCodec_21Hz_no_eliz.nemo"
    codec_model = AudioCodecModel.restore_from(codec_path, strict=False)
    codec_model.to(device)
    num_codebooks = 8
    codec_feature_dim = num_codebooks * 4
    metric = FrechetCodecDistance(codec=codec_model, feature_dim=codec_feature_dim, device=device)

    B = 3
    T = 20
    C = 8
    codes = torch.randint(low=0, high=2048, size=(B, C, T), device=device)
    codes_len = torch.randint(low=1, high=T, size=(B,), device=device)
    metric.update_from_codes(codes, codes_len, is_real=True)
    metric.update_from_codes(codes, codes_len, is_real=True)
    metric.update_from_audio_file("/datap/misc/Datasets/LibriTTS/dev-clean/1272/141231/1272_141231_000013_000002.wav", is_real=True)


    codes = torch.randint(low=0, high=2048, size=(B, C, T), device=device)
    codes_len = torch.randint(low=1, high=T, size=(B,), device=device)
    metric.update(codes, codes_len, is_real=False)
    print(metric.compute())
