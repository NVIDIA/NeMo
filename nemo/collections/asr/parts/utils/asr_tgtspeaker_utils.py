import random
from copy import deepcopy
from typing import List

import numpy as np
from cytoolz import groupby
from lhotse import AudioSource, Recording, SupervisionSegment, SupervisionSet
from lhotse.cut import Cut, CutSet, MixedCut, MixTrack, MonoCut
from lhotse.lazy import LazyJsonlIterator
from lhotse.utils import compute_num_samples, uuid4

from nemo.collections.asr.parts.utils.asr_multispeaker_utils import (
    find_segments_from_rttm,
    get_hidden_length_from_sample_length,
)


def mix_noise(
    cuts,
    noise_manifests,
    snr,
    mix_prob,
):
    """Mix noise to the cuts with a given probability and SNR.

    Args:
        cuts (CutSet): Currently only supports MonoCut in cutset.
        noise_manifests (list): The list of noise manifests.
        snr (float or list): The SNR value. If a list is provided, a random SNR will be sampled from the list.
        mix_prob (float): The probability of mixing noise.
    Returns:
        CutSet: The cutset with mixed noise.
    """
    mixed_cuts = []
    assert 0.0 <= mix_prob <= 1.0, "mix_prob must be between 0.0 and 1.0"
    for cut in cuts:
        if random.uniform(0.0, 1.0) > mix_prob or cut.duration == 0:
            mixed_cuts.append(cut)
            continue
        to_mix_manifest = random.choice(noise_manifests)
        to_mix_cut = json_to_cut(to_mix_manifest)
        to_mix_cut = to_mix_cut.resample(16000)
        snr = random.uniform(*snr) if isinstance(snr, (list, tuple)) else snr
        mixed = cut.mix(to_mix_cut, snr=snr)
        mixed = mixed.truncate(duration=cut.duration)
        mixed_cuts.append(mixed)
    return CutSet.from_cuts(mixed_cuts)


def rir_augment(cuts, prob):
    """
    Augment the cuts with RIR.

    Args:
        cuts (CutSet):  Currently only supports MonoCut in cutset.
        prob (float): The probability of augmenting the cuts with RIR.
    Returns:
        CutSet: The cutset with RIR augmented.
    TODO:
        - Add support for MixedCut in cutset.
        - Receive RIR manifest and sample RIR
    """
    mixed_cuts = []
    for cut in cuts:
        if random.uniform(0.0, 1.0) > prob:
            mixed_cuts.append(cut)
        else:
            mixed_cuts.append(cut.reverb_rir())
    return CutSet.from_cuts(mixed_cuts)


def codec_augment(cuts, prob):
    """
    Augment the cuts with narrowband codec.

    Args:
        cuts (CutSet):  Currently only supports MonoCut in cutset.
        prob (float): The probability of augmenting the cuts with narrowband codec.
    Returns:
        CutSet: The cutset with narrowband codec augmented.
    TODO:
        - Add support for MixedCut in cutset.
        - apply lhotse-style codec augmentation.
    """
    mixed_cuts = []
    for cut in cuts:
        if random.uniform(0.0, 1.0) > prob:
            mixed_cuts.append(cut)
        else:
            mixed_cuts.append(cut.narrowband('mulaw'))
    return CutSet.from_cuts(mixed_cuts)


def speaker_to_target_w_query(
    a_cut,
    query,
    separater_duration: int = 1,
    num_speakers: int = 4,
    num_sample_per_mel_frame: int = 160,
    num_mel_frame_per_asr_frame: int = 8,
    spk_tar_all_zero: bool = False,
    boundary_segments: bool = False,
):
    """
    Get rttm samples corresponding to one cut,
    generate speaker mask numpy.ndarray with shape (num_speaker, hidden_length).

    This function is needed for speaker diarization with ASR model trainings.

    Args:
        a_cut (MonoCut, MixedCut): Lhotse Cut instance which is MonoCut or MixedCut instance.
        query (MonoCut): query cut
        separater_duration (int): separater duration, 1 by default
        num_speakers (int): max number of speakers for all cuts ("mask" dim0), 4 by default
        num_sample_per_mel_frame (int):
            number of sample per mel frame, sample_rate / 1000 * window_stride, 160 by default (10ms window stride)
        num_mel_frame_per_asr_frame (int): encoder subsampling_factor, 8 by default
        spk_tar_all_zero (Tensor): set to True gives all zero "mask"
        boundary_segments (bool):
            set to True to include segments containing the boundary of the cut,
            False by default for multi-speaker ASR training

    Returns:
        mask (Tensor): speaker mask with shape (num_speaker, hidden_lenght)
    """
    # get cut-related segments from rttms
    if isinstance(a_cut, MixedCut):
        cut_list = [track.cut for track in a_cut.tracks if isinstance(track.cut, MonoCut)]
        offsets = [track.offset for track in a_cut.tracks if isinstance(track.cut, MonoCut)]
    elif isinstance(a_cut, MonoCut):
        cut_list = [a_cut]
        offsets = [0]
    else:
        raise ValueError(f"Unsupported cut type type{cut}: only MixedCut and MonoCut are supported")

    # initialize mask matrices (num_speaker, encoder_hidden_len)
    encoder_hidden_len = get_hidden_length_from_sample_length(
        a_cut.num_samples + query.num_samples + separater_duration * query.sampling_rate,
        num_sample_per_mel_frame,
        num_mel_frame_per_asr_frame,
    )

    separater_hidden_len = get_hidden_length_from_sample_length(
        separater_duration * query.sampling_rate, num_sample_per_mel_frame, num_mel_frame_per_asr_frame
    )

    query_hidden_len = get_hidden_length_from_sample_length(
        query.num_samples, num_sample_per_mel_frame, num_mel_frame_per_asr_frame
    )

    mask = np.zeros((num_speakers, encoder_hidden_len))

    if spk_tar_all_zero:
        return mask

    segments_total = []
    for i, cut in enumerate(cut_list):
        if hasattr(cut, 'rttm_filepath') and cut.rttm_filepath is not None:
            rttms = SupervisionSet.from_rttm(cut.rttm_filepath)
        elif hasattr(cut, 'speaker_id') and cut.speaker_id is not None:
            rttms = SupervisionSet.from_segments(
                [
                    SupervisionSegment(
                        id=uuid4(),
                        recording_id=cut.recording_id,
                        start=cut.start,
                        duration=cut.duration,
                        channel=1,
                        speaker=cut.speaker_id,
                        language=None,
                    )
                ]
            )
        else:
            raise ValueError(f"Cut {cut.id} does not have rttm_filepath or speaker_id")
        if boundary_segments:  # segments with seg_start < total_end and seg_end > total_start are included
            segments_iterator = find_segments_from_rttm(
                recording_id=cut.recording_id, rttms=rttms, start_after=cut.start, end_before=cut.end, tolerance=0.0
            )
        else:  # segments with seg_start > total_start and seg_end < total_end are included
            segments_iterator = rttms.find(
                recording_id=cut.recording_id, start_after=cut.start, end_before=cut.end, adjust_offset=True
            )

        for seg in segments_iterator:
            if seg.start < 0:
                seg.duration += seg.start
                seg.start = 0
            if seg.end > cut.duration:
                seg.duration -= seg.end - cut.duration
            seg.start += offsets[i]
            segments_total.append(seg)

    # apply arrival time sorting to the existing segments
    segments_total.sort(key=lambda rttm_sup: rttm_sup.start)
    seen = set()
    seen_add = seen.add
    if isinstance(a_cut, MixedCut):
        cut = a_cut
    # add query speaker as the first speaker
    speaker_lst = [cut.query_speaker_id] + [s.speaker for s in segments_total]  # add query
    speaker_ats = [s for s in speaker_lst if not (s in seen or seen_add(s))]

    speaker_to_idx_map = {spk: idx for idx, spk in enumerate(speaker_ats)}

    if hasattr(query, 'rttm_filepath') and query.rttm_filepath is not None:
        # if query cut has rttm, use it to generate query speaker mask
        query_rttms = SupervisionSet.from_rttm(query.rttm_filepath)
        query_segments_iterator = find_segments_from_rttm(
            recording_id=query.recording_id,
            rttms=query_rttms,
            start_after=query.start,
            end_before=query.end,
            tolerance=0.0,
        )
        query_segments_total = []
        for seg in query_segments_iterator:
            if seg.start < 0:
                seg.duration += seg.start
                seg.start = 0
            if seg.end > query.duration:
                seg.duration -= seg.end - query.duration
            query_segments_total.append(seg)
        for rttm_sup in query_segments_total:
            st = compute_num_samples(rttm_sup.start, query.sampling_rate) if rttm_sup.start > 0 else 0
            et = (
                compute_num_samples(rttm_sup.end, query.sampling_rate)
                if rttm_sup.end < query.duration
                else compute_num_samples(query.duration, query.sampling_rate)
            )
            st_encoder_loc = get_hidden_length_from_sample_length(
                st, num_sample_per_mel_frame, num_mel_frame_per_asr_frame
            )
            et_encoder_loc = get_hidden_length_from_sample_length(
                et, num_sample_per_mel_frame, num_mel_frame_per_asr_frame
            )
            mask[0, st_encoder_loc:et_encoder_loc] = 1
    else:
        # if query cut has no rttm, use query cut duration to generate all-one query speaker mask
        mask[0, :query_hidden_len] = 1

    for rttm_sup in segments_total:
        speaker_idx = speaker_to_idx_map[rttm_sup.speaker]
        # only consider the first <num_speakers> speakers
        if speaker_idx < 4:
            st = compute_num_samples(rttm_sup.start, cut.sampling_rate) if rttm_sup.start > 0 else 0
            et = (
                compute_num_samples(rttm_sup.end, cut.sampling_rate)
                if rttm_sup.end < cut.duration
                else compute_num_samples(cut.duration, cut.sampling_rate)
            )

            # map start time (st) and end time (et) to encoded hidden location
            st_encoder_loc = get_hidden_length_from_sample_length(
                st, num_sample_per_mel_frame, num_mel_frame_per_asr_frame
            )
            et_encoder_loc = get_hidden_length_from_sample_length(
                et, num_sample_per_mel_frame, num_mel_frame_per_asr_frame
            )

            mask[
                speaker_idx,
                query_hidden_len
                + separater_hidden_len
                + st_encoder_loc : query_hidden_len
                + separater_hidden_len
                + et_encoder_loc,
            ] = 1

    return mask


def get_separator_audio(freq, sr, duration, ratio):
    """
    Generate a separator audio with a given frequency, sampling rate, duration, and ratio.

    Args:
        freq (float): The frequency of the separator audio.
        sr (int): The sampling rate of the separator audio.
        duration (float): The duration of the separator audio.
        ratio (float): The ratio of the separator audio.
    Returns:
        y (np.ndarray): The separator audio.
    """
    # Generate time values
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)

    # Generate sine wave
    y = np.sin(2 * np.pi * freq * t) * 0.1

    y[: int(sr * duration * ratio)] = 0
    y[-int(sr * duration * ratio) :] = 0
    return y


def get_query_cut(cut):
    """
    Extract query from the cut and saved as a separate cut.

    Args:
        cut: An audio cut. The cut should contain keys "query_audio_filepath", "query_offet", "query_duration"

    Returns:
        query_cut: a cut containing query information
    """
    query_rec = Recording.from_file(cut.query_audio_filepath)
    if query_rec.sampling_rate != 16000:
        query_rec = query_rec.resample(sampling_rate=16000)
    query_sups = [
        SupervisionSegment(
            id=query_rec.id + '_query' + str(cut.query_offset) + '-' + str(cut.query_offset + cut.query_duration),
            recording_id=query_rec.id,
            start=0,
            duration=cut.query_duration,
            speaker=cut.query_speaker_id,
        )
    ]
    # additional information for query
    custom = {'rttm_filepath': cut.custom.get('query_rttm_filepath', None)}
    query_cut = MonoCut(
        id=query_rec.id + '_query' + str(cut.query_offset) + '-' + str(cut.query_offset + cut.query_duration),
        start=cut.query_offset,
        duration=cut.query_duration,
        channel=0,
        recording=query_rec,
        supervisions=query_sups,
    )
    query_cut.custom = custom
    return query_cut


def get_bounded_segment(start_time, total_duration, min_duration=1.0, max_duration=10.0):
    """
    Generate a segment within an audio clip with bounded duration.

    Args:
        start_time (float): Start time of the audio in seconds
        total_duration (float): Total duration of the audio in seconds
        min_duration (float): Minimum allowed segment duration in seconds
        max_duration (float): Maximum allowed segment duration in seconds

    Returns:
        tuple: (segment_start, segment_duration)
    """
    # Ensure max_duration doesn't exceed total_duration
    max_duration = min(max_duration, total_duration)

    # Ensure min_duration is not greater than max_duration
    min_duration = min(min_duration, max_duration)

    # Generate random duration within bounds
    segment_duration = np.round(random.uniform(min_duration, max_duration), decimals=3)

    # Calculate maximum possible start time
    max_start = total_duration - segment_duration

    # Generate random start time
    segment_start = np.round(random.uniform(start_time, start_time + max_start), decimals=3)

    return segment_start, segment_duration


def json_to_cut(json_dict):
    """
    Convert a json dictionary to a Cut instance.

    Args:
        json_dict (dict): A json dictionary.
    Returns:
        cut (Cut): A Cut instance.
    """
    audio_path = json_dict["audio_filepath"]
    duration = json_dict["duration"]
    offset = json_dict.get("offset", None)
    cut = _create_cut(
        audio_path=audio_path, offset=offset, duration=duration, sampling_rate=json_dict.get("sampling_rate", None)
    )
    # Note that start=0 and not start=offset because supervision's start if relative to the
    # start of the cut; and cut.start is already set to offset
    cut.supervisions.append(
        SupervisionSegment(
            id=cut.id,
            recording_id=cut.recording_id,
            start=0,
            duration=cut.duration,
            text=json_dict.get("text"),
            language=json_dict.get("language", "en"),
        )
    )
    cut.custom = json_dict

    return cut


def _create_cut(
    audio_path: str,
    offset: float,
    duration: float,
    sampling_rate: int | None = None,
) -> Cut:
    """
    Create a cut from an audio path, offset, duration, and sampling rate.

    Args:
        audio_path (str): The path to the audio file.
        offset (float): The offset of the cut.
        duration (float): The duration of the cut.
        sampling_rate (int | None): The sampling rate of the audio file.
    """
    recording = _create_recording(audio_path, duration, sampling_rate)
    cut = recording.to_cut()
    if offset is not None:
        cut = cut.truncate(offset=offset, duration=duration, preserve_id=True)
        cut.id = f"{cut.id}-{round(offset * 1e2):06d}-{round(duration * 1e2):06d}"
    return cut


def _create_recording(
    audio_path: str,
    duration: float,
    sampling_rate: int | None = None,
) -> Recording:
    """
    Create a recording from an audio path, duration, and sampling rate.

    Args:
        audio_path (str): The path to the audio file.
        duration (float): The duration of the audio file.
        sampling_rate (int | None): The sampling rate of the audio file.
    Returns:
        recording (Recording): A Recording instance.
    """
    if sampling_rate is not None:
        # TODO(pzelasko): It will only work with single-channel audio in the current shape.
        return Recording(
            id=audio_path,
            sources=[AudioSource(type="file", channels=[0], source=audio_path)],
            sampling_rate=sampling_rate,
            num_samples=compute_num_samples(duration, sampling_rate),
            duration=duration,
            channel_ids=[0],
        )
    return Recording.from_file(audio_path)


class TargetSpeakerSimulator:
    """
    This class is used to simulate target-speaker audio data,
    which can be used for target-speaker ASR and speaker diarization training.
    """

    def __init__(
        self,
        manifest_filepath,
        num_speakers,
        simulator_type,
        min_delay=0.5,
        max_delay_after_each_mono: float = 0,
        non_query_sample: bool = False,
        query_duration: List[float] = None,
    ):
        """
        Args:
            manifest_filepath (str): The path to the manifest file.
            num_speakers (int): The number of speakers in the simulated audio.
            simulator_type (str): The type of simulator to use.
                - 'lsmix': LibriSpeechMix-style training sample (mix single speaker audio).
            min_delay (float): The minimum delay between speakers
                to avoid the same starting time for multiple speakers.
            max_delay_after_each_mono (float):
                The maximum delay of another mono cut after each mono cut.
                Default is 0, means audio mixtures guaranteed to overlap.
            non_query_sample (bool):
                Whether to sample a sample where query speaker not in target audio.
                Default is False.
            query_duration (list): The duration of the query sample in s. Default is [3, 10].
            TODO: add mono_duration (list):
                Select random start and duration for each single speaker audio according to mono_duration [min max].
                Emprically, need to set min_duration > 0 if max_after_each_mono > 0!!!
        """
        if query_duration is None:
            query_duration = [3, 10]

        self.manifests = LazyJsonlIterator(manifest_filepath)
        self.min_delay = min_delay
        self.max_delay_after_each_mono = max_delay_after_each_mono
        self.num_speakers = num_speakers
        self.simulator_type = simulator_type
        self.query_duration = query_duration
        self.non_query_sample = non_query_sample

        self.spk2manifests = groupby(lambda x: x["speaker_id"], self.manifests)
        self.speaker_ids = list(self.spk2manifests.keys())

        if simulator_type == 'lsmix':
            self.simulator = self.libri_speech_mix_simulator_tgt
        elif simulator_type == 'meeting':
            # TODO
            raise NotImplementedError("MeetingSimulator is not implemented yet.")
        elif simulator_type == 'conversation':
            # TODO
            raise NotImplementedError("ConversationSimulator is not implemented yet.")

    def __iter__(self):
        return self

    def __next__(self):
        return self.simulator()

    def libri_speech_mix_simulator_tgt(self):
        """
        This function simulates a LibriSpeechMix-style TS-ASR training sample.

        Returns:
            mixed_cut: a mixed cut containing target-speaker audio and query speaker audio.
        """
        # Sample the speakers
        sampled_speaker_ids = random.sample(self.speaker_ids, self.num_speakers)

        # Create tracks for all speakers at once
        tracks = []
        offset = 0

        # Common custom dict to avoid recreating
        base_custom = {'pnc': 'no', 'source_lang': 'en', 'target_lang': 'en', 'task': 'asr'}

        # Create tracks in a single loop
        for speaker_id in sampled_speaker_ids:
            manifest = random.choice(self.spk2manifests[speaker_id])
            mono_cut = json_to_cut(manifest)
            mono_cut.custom.update(base_custom)
            tracks.append(MixTrack(cut=deepcopy(mono_cut), type=type(mono_cut), offset=offset))
            offset += random.uniform(self.min_delay, mono_cut.duration + self.max_delay_after_each_mono)

        # Create mixed cut
        mixed_cut = MixedCut(
            id='lsmix_' + '_'.join([track.cut.id for track in tracks]) + '_' + str(uuid4()), tracks=tracks
        )

        # Handle query speaker selection
        if self.non_query_sample:
            query_speaker_id = random.choice(list(set(self.speaker_ids) - set(sampled_speaker_ids)))
        else:
            query_speaker_id = random.choice(sampled_speaker_ids)

        # Get query cut
        query_manifest = random.choice(self.spk2manifests[query_speaker_id])
        query_cut = json_to_cut(query_manifest)

        # Create supervision
        text = self.get_text(mixed_cut, query_speaker_id) if not self.non_query_sample else ""
        sup = SupervisionSegment(
            id=mixed_cut.id, recording_id=mixed_cut.id, start=0, duration=mixed_cut.duration, text=text
        )

        # Get query segment bounds
        query_offset, query_duration = get_bounded_segment(
            query_cut.start,
            query_cut.duration,
            min_duration=self.query_duration[0],
            max_duration=self.query_duration[1],
        )

        # Update cut with final metadata
        custom = {
            **base_custom,
            'query_audio_filepath': query_cut.recording.sources[0].source,
            'query_speaker_id': query_speaker_id,
            'query_offset': query_offset,
            'query_duration': query_duration,
            'query_rttm_filepath': query_cut.rttm_filepath if hasattr(query_cut, 'rttm_filepath') else None,
            'custom': None,
        }

        mixed_cut.tracks[0].cut.supervisions = [sup]
        mixed_cut.tracks[0].cut.custom.update(custom)

        return mixed_cut

    def get_text(self, cut: MixedCut, query_speaker_id) -> str:
        """
        Get the text of the query speaker in the target utterance.

        Args:
            cut (MixedCut): The mixed cut containing target-speaker audio and query speaker audio.
            query_speaker_id (str): The id of the query speaker.
        Returns:
            text (str): The text of the query speaker in the target utterance.
        """
        for track in cut.tracks:
            if track.cut.speaker_id == query_speaker_id:
                return track.cut.text
        return ValueError('Error in finding query speaker in target utterance')

    def meeting_simulator(self):
        """Meeting simulator method."""
        raise NotImplementedError("MeetingSimulator is not implemented yet.")

    def conversation_simulator(self):
        """Conversation simulator method."""
        raise NotImplementedError("ConversationSimulator is not implemented yet.")

    # TODO: text is necessary for msasr and tsasr, but not for diar
