import torch
import torch.nn as nn
from numba import cuda

from nemo.utils import logging


THREAD_BUFFER = 128


@cuda.jit()
def spec_augment_kernel(
    x: torch.Tensor,
    x_len: torch.Tensor,
    freq_starts: torch.Tensor,
    freq_widths: torch.Tensor,
    time_starts: torch.Tensor,
    time_widths: torch.Tensor,
    mask_value: float,
):
    """
    Numba CUDA kernel to perform SpecAugment in-place on the GPU.

    Args:
        x: Pytorch tensor of shape [B, F, T] with the acoustic features.
        x_len: Pytorch tensor of shape [B] with the lengths of the padded sequence.
        freq_starts: Pytorch tensor of shape [B] with the start indices of freq masks.
        freq_widths: Pytorch tensor of shape [B] with the width of freq masks.
        time_starts: Pytorch tensor of shape [B] with the start indices of time masks.
        time_widths: Pytorch tensor of shape [B] with the width of time masks.
        mask_value: Float value that will be used as mask value.
    """
    f = cuda.blockIdx.x  # indexes the Freq dim
    t = cuda.blockIdx.y  # indexes the Time dim
    tid = cuda.threadIdx.x  # index of the current mask
    threads_per_block = cuda.blockDim.x

    # Compute the number of masks over freq axis
    len_f = freq_starts.shape[0]
    # For `len_f` number of freq masks that must be applied
    for fidx in range(0, len_f, threads_per_block):
        # Resolve the index of the freq mask (case where more masks than THREAD_BUFFER)
        fm_idx = fidx * threads_per_block + tid

        # If resolved freq mask index < total number of freq masks
        if fm_idx < len_f:
            # Access the start index and width of this freq mask
            f_start = freq_starts[fm_idx]
            f_width = freq_widths[fm_idx]

            # If block idx `f` >= start and < (start + width) of this freq mask
            if f >= f_start and f < (f_start + f_width):
                # For all samples in the batch, apply the freq mask
                for b in range(x.shape[0]):
                    x[b, f, t] = mask_value

    # Compute the number of masks over time axis
    len_t = time_starts.shape[0]
    # For `len_t` number of freq masks that must be applied
    for tidx in range(0, len_t, threads_per_block):
        # Resolve the index of the freq mask (case where more masks than THREAD_BUFFER)
        tm_idx = tidx * threads_per_block + tid

        # If resolved time mask index < total number of time masks
        if tm_idx < len_t:
            # Access the start index and width of this time mask
            t_start = time_starts[tm_idx]
            t_width = time_widths[tm_idx]

            # If block idx `t` >= start and < (start + width) of this time mask
            if t >= t_start and t < (t_start + t_width):
                # For all samples in the batch, apply the time mask
                for b in range(x.shape[0]):
                    # Current block idx `t` < current seq length x_len[b]
                    # This ensure that we mask only upto the length of that sample
                    # Everything after that index is padded value so unnecessary to mask
                    if t < x_len[b]:
                        x[b, f, t] = mask_value


def launch_spec_augment_kernel(
    x: torch.Tensor,
    x_len: torch.Tensor,
    freq_starts: torch.Tensor,
    freq_lengths: torch.Tensor,
    time_starts: torch.Tensor,
    time_lengths: torch.Tensor,
    freq_masks: int,
    time_masks: int,
    mask_value: float,
):
    """
    Helper method to launch the SpecAugment kernel

    Args:
        x: Pytorch tensor of shape [B, F, T] with the acoustic features.
        x_len: Pytorch tensor of shape [B] with the lengths of the padded sequence.
        freq_starts: Pytorch tensor of shape [B] with the start indices of freq masks.
        freq_widths: Pytorch tensor of shape [B] with the width of freq masks.
        time_starts: Pytorch tensor of shape [B] with the start indices of time masks.
        time_widths: Pytorch tensor of shape [B] with the width of time masks.
        freq_masks: Int value that determines the number of time masks.
        time_masks: Int value that determines the number of freq masks.
        mask_value: Float value that will be used as mask value.

    Returns:
        The spec augmented tensor 'x'
    """
    # Setup CUDA stream
    sh = x.shape
    stream = cuda.external_stream(torch.cuda.current_stream(x.device).cuda_stream)

    if time_masks > 0 or freq_masks > 0:
        # Parallelize over freq and time axis, parallel threads over max(num_freq_masks, num_time_masks)
        # Sequential over batch size (adaptive in time).
        blocks_per_grid = [sh[1], sh[2]]
        threads_per_block = min(THREAD_BUFFER, max(freq_masks, time_masks))

        # Numba does not support fp16, force cast to fp32 temporarily at the expense of memory
        original_dtype = x.dtype
        cast_x = False
        if x.dtype == torch.float16:
            x = x.float()
            cast_x = True

        # Launch CUDA kernel
        spec_augment_kernel[blocks_per_grid, threads_per_block, stream, 0](
            x, x_len, freq_starts, freq_lengths, time_starts, time_lengths, mask_value
        )
        torch.cuda.synchronize()

        # Recast back to original dtype if earlier cast was performed
        if cast_x:
            x = x.to(dtype=original_dtype)

    return x


class SpecAugmentNumba(nn.Module):
    """
    Zeroes out(cuts) random continuous horisontal or
    vertical segments of the spectrogram as described in
    SpecAugment (https://arxiv.org/abs/1904.08779).

    Utilizes a Numba CUDA kernel to perform inplace edit of the input without loops.

    Args:
        freq_masks - how many frequency segments should be cut
        time_masks - how many time segments should be cut
        freq_width - maximum number of frequencies to be cut in one segment
        time_width - maximum number of time steps to be cut in one segment.
            Can be a positive integer or a float value in the range [0, 1].
            If positive integer value, defines maximum number of time steps
            to be cut in one segment.
            If a float value, defines maximum percentage of timesteps that
            are cut adaptively.
        rng: Ignored.
    """

    def __init__(
        self, freq_masks=0, time_masks=0, freq_width=10, time_width=0.1, rng=None, mask_value=0.0,
    ):
        super().__init__()

        self.freq_masks = freq_masks
        self.time_masks = time_masks

        self.freq_width = freq_width
        self.time_width = time_width

        self.mask_value = mask_value

        # Unused
        self.rng = rng
        if self.rng is not None:
            logging.warning("`rng` was supplied to SpecAugmentNumba, but it is not used.")

        if isinstance(time_width, int):
            self.adaptive_temporal_width = False
        else:
            if time_width > 1.0 or time_width < 0.0:
                raise ValueError('If `time_width` is a float value, must be in range [0, 1]')

            self.adaptive_temporal_width = True

    @torch.no_grad()
    def forward(self, x, x_len):
        sh = x.shape

        if self.adaptive_temporal_width:
            time_width = max(1, int(sh[2] * self.time_width))
        else:
            time_width = self.time_width

        # Construct the freq and time masks as well as start positions
        if self.freq_masks > 0:
            freq_starts = torch.randint(0, sh[1] - self.freq_width, size=[self.freq_masks], device=x.device)
            freq_lengths = torch.randint(0, self.freq_width, size=[self.freq_masks], device=x.device)
        else:
            freq_starts = torch.zeros([1], dtype=torch.int64, device=x.device)
            freq_lengths = torch.zeros([1], dtype=torch.int64, device=x.device)

        if self.time_masks > 0:
            time_starts = torch.randint(0, sh[2] - time_width, size=[self.time_masks], device=x.device)
            time_lengths = torch.randint(0, time_width, size=[self.time_masks], device=x.device)
        else:
            time_starts = torch.zeros([1], dtype=torch.int64, device=x.device)
            time_lengths = torch.zeros([1], dtype=torch.int64, device=x.device)

        x = launch_spec_augment_kernel(
            x,
            x_len,
            freq_starts=freq_starts,
            freq_lengths=freq_lengths,
            time_starts=time_starts,
            time_lengths=time_lengths,
            freq_masks=self.freq_masks,
            time_masks=self.time_masks,
            mask_value=self.mask_value,
        )

        return x
