import torch
import torch.nn as nn
from numba import cuda


THREAD_BUFFER = 128


@cuda.jit()
def spec_augment(
    x: torch.Tensor,
    x_len: torch.Tensor,
    freq_starts: torch.Tensor,
    freq_widths: torch.Tensor,
    time_starts: torch.Tensor,
    time_widths: torch.Tensor,
    mask_value: float,
):
    f = cuda.blockIdx.x
    t = cuda.blockIdx.y
    tid = cuda.threadIdx.x
    threads_per_block = cuda.blockDim.x

    len_f = freq_starts.shape[0]
    for fidx in range(0, len_f, threads_per_block):
        ft_idx = fidx * threads_per_block + tid

        if ft_idx < len_f:
            f_start = freq_starts[ft_idx]
            f_width = freq_widths[ft_idx]

            if f >= f_start and f < (f_start + f_width):
                for b in range(x.shape[0]):
                    x[b, f, t] = mask_value

    len_t = time_starts.shape[0]
    for tidx in range(0, len_t, threads_per_block):
        tt_idx = tidx * threads_per_block + tid

        if tt_idx < len_t:
            t_start = time_starts[tt_idx]
            t_width = time_widths[tt_idx]

            if t >= t_start and t < (t_start + t_width):
                for b in range(x.shape[0]):  # current t < current max len x_len[b]
                    if t < x_len[b]:
                        x[b, f, t] = mask_value


class SpecAugmentNumba(nn.Module):
    """
    Zeroes out(cuts) random continuous horisontal or
    vertical segments of the spectrogram as described in
    SpecAugment (https://arxiv.org/abs/1904.08779).
    params:
    freq_masks - how many frequency segments should be cut
    time_masks - how many time segments should be cut
    freq_width - maximum number of frequencies to be cut in one segment
    time_width - maximum number of time steps to be cut in one segment.
        Can be a positive integer or a float value in the range [0, 1].
        If positive integer value, defines maximum number of time steps
        to be cut in one segment.
        If a float value, defines maximum percentage of timesteps that
        are cut adaptively.
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

        freq_starts = torch.randint(0, sh[1] - self.freq_width, size=[self.freq_masks], device=x.device)
        freq_lengths = torch.randint(0, self.freq_width, size=[self.freq_masks], device=x.device)
        time_starts = torch.randint(0, sh[2] - time_width, size=[self.time_masks], device=x.device)
        time_lengths = torch.randint(0, time_width, size=[self.time_masks], device=x.device)

        stream = cuda.external_stream(torch.cuda.current_stream(x.device).cuda_stream)

        blocks_per_grid = [sh[1], sh[2]]
        threads_per_block = min(THREAD_BUFFER, max(self.freq_masks, self.time_masks))
        spec_augment[blocks_per_grid, threads_per_block, stream, 0](
            x, x_len, freq_starts, freq_lengths, time_starts, time_lengths, self.mask_value
        )

        torch.cuda.synchronize()

        return x


if __name__ == '__main__':

    shape = [1, 2, 50]
    x = torch.randn(*shape, device='cuda')
    x_len = torch.randint(shape[-1], size=[shape[0]], device=x.device)

    spec_aug = SpecAugmentNumba(
        freq_masks=2, time_masks=10, freq_width=1, time_width=0.05, mask_value=0.0
    )
    # Warmup
    _ = spec_aug(x, x_len)

    with torch.autograd.profiler.profile(use_cuda=True, profile_memory=True) as prof:
        with torch.autograd.profiler.record_function("spec_aug_cuda"):
            x_masked = spec_aug(x, x_len)

    print(prof)

    # print(x_masked[0].to('cpu'))
