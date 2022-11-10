import torch


class ContextWindow(torch.nn.Module):
    def __init__(
        self, left_frames=0, right_frames=0,
    ):
        super().__init__()
        self.left_frames = left_frames
        self.right_frames = right_frames
        self.context_len = self.left_frames + self.right_frames + 1
        self.kernel_len = 2 * max(self.left_frames, self.right_frames) + 1

        # Kernel definition
        self.kernel = torch.eye(self.context_len, self.kernel_len)

        if self.right_frames > self.left_frames:
            lag = self.right_frames - self.left_frames
            self.kernel = torch.roll(self.kernel, lag, 1)

        self.first_call = True

    @torch.no_grad()
    def forward(self, x):
        """Returns the tensor with the surrounding context.
        Arguments
        ---------
        x : tensor
            A batch of tensors.
        """

        x = x.transpose(1, 2)

        if self.first_call is True:
            self.first_call = False
            self.kernel = (
                self.kernel.repeat(x.shape[1], 1, 1).view(x.shape[1] * self.context_len, self.kernel_len,).unsqueeze(1)
            )

        # Managing multi-channel case
        or_shape = x.shape
        if len(or_shape) == 4:
            x = x.reshape(or_shape[0] * or_shape[2], or_shape[1], or_shape[3])

        # Compute context (using the estimated convolutional kernel)
        cw_x = torch.nn.functional.conv1d(
            x, self.kernel.to(x.device), groups=x.shape[1], padding=max(self.left_frames, self.right_frames),
        )

        # Retrieving the original dimensionality (for multi-channel case)
        if len(or_shape) == 4:
            cw_x = cw_x.reshape(or_shape[0], cw_x.shape[1], or_shape[2], cw_x.shape[-1])

        cw_x = cw_x.transpose(1, 2)

        return cw_x


def assign_frame_level_spk_vector(
    rttm_timestamps: list,
    round_digits: int,
    frame_per_sec: int,
    sample_rate: int,
    start_duration: float,
    end_duration: float,
    preprocessor,
    subsampling: int,
    speakers: list,
    min_spks: int = 2,
):
    """
    Create a multi-dimensional vector sequence containing speaker timestamp information in RTTM.
    The unit-length is the frame shift length of the acoustic feature. The feature-level annotations
    `fr_level_target` will later be converted to base-segment level diarization label.

    Args:
        rttm_timestamps (list):
            List containing start and end time for each speaker segment label.
            stt_list, end_list and speaker_list are contained.
        frame_per_sec (int):
            Number of feature frames per second. This quantity is determined by window_stride variable in preprocessing module.

    Returns:
        fr_level_target (torch.tensor):
            Tensor containing label for each feature level frame.
    """
    stt_list, end_list, speaker_list = rttm_timestamps
    segment_duration = end_duration - start_duration

    total_fr_len = preprocessor.featurizer.get_seq_len(torch.tensor(segment_duration * sample_rate, dtype=torch.float))
    total_fr_len = int(total_fr_len / subsampling) + 1
    spk_num = max(len(speakers), min_spks)
    speaker_mapping_dict = {rttm_key: x_int for x_int, rttm_key in enumerate(speakers)}
    fr_level_target = torch.zeros(total_fr_len, spk_num)

    # If RTTM is not provided, then there is no speaker mapping dict in target_spks.
    # Thus, return a zero-filled tensor as a placeholder.
    for count, (stt, end, spk_rttm_key) in enumerate(zip(stt_list, end_list, speaker_list)):
        stt, end = round(stt, round_digits), round(end, round_digits)
        # check if this sample is within the segment frame.
        if (start_duration <= stt) and (end <= end_duration):
            spk = speaker_mapping_dict[spk_rttm_key]
            # calculate the start/end relative to this segment of the file.
            relative_stt, relative_end = stt - start_duration, end - start_duration
            stt_fr, end_fr = int(relative_stt * frame_per_sec), int(relative_end * frame_per_sec)
            fr_level_target[stt_fr:end_fr, spk] = 1
        if stt >= end_duration:
            # we've reached the required size of the segment
            break
    return fr_level_target
