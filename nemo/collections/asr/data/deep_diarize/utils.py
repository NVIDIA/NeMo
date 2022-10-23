import torch


def assign_frame_level_spk_vector(
    rttm_timestamps: list,
    round_digits: int,
    frame_per_sec: int,
    sample_rate: int,
    start_duration: float,
    end_duration: float,
    preprocessor,
    subsampling: int,
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
    sorted_speakers = sorted(list(set(speaker_list)))
    segment_duration = end_duration - start_duration

    total_fr_len = preprocessor.featurizer.get_seq_len(torch.tensor(segment_duration * sample_rate, dtype=torch.float))
    total_fr_len = int(total_fr_len / subsampling) + 1
    spk_num = max(len(sorted_speakers), min_spks)
    speaker_mapping_dict = {rttm_key: x_int for x_int, rttm_key in enumerate(sorted_speakers)}
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
