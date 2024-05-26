import math
import random

import librosa
import numpy as np
import parselmouth
import scipy.signal
import torch
import torchaudio.functional as AF

PRAAT_CHANGEGENDER_PITCHMEDIAN_DEFAULT = 0.0
PRAAT_CHANGEGENDER_FORMANTSHIFTRATIO_DEFAULT = 1.0
PRAAT_CHANGEGENDER_PITCHSHIFTRATIO_DEFAULT = 1.0
PRAAT_CHANGEGENDER_PITCHRANGERATIO_DEFAULT = 1.0
PRAAT_CHANGEGENDER_DURATIONFACTOR_DEFAULT = 1.0


def wav_to_Sound(wav, sampling_frequency: int = 22050) -> parselmouth.Sound:
    r""" load wav file to parselmouth Sound file

    # __init__(self: parselmouth.Sound, other: parselmouth.Sound) -> None \
    # __init__(self: parselmouth.Sound, values: numpy.ndarray[numpy.float64], sampling_frequency: Positive[float] = 44100.0, start_time: float = 0.0) -> None \
    # __init__(self: parselmouth.Sound, file_path: str) -> None

    returns:
        sound: parselmouth.Sound
    """
    if isinstance(wav, parselmouth.Sound):
        sound = wav
    elif isinstance(wav, np.ndarray):
        sound = parselmouth.Sound(wav, sampling_frequency=sampling_frequency)
    elif isinstance(wav, list):
        wav_np = np.asarray(wav)
        sound = parselmouth.Sound(np.asarray(wav_np), sampling_frequency=sampling_frequency)
    else:
        raise NotImplementedError
    return sound


def wav_to_Tensor(wav) -> torch.Tensor:
    if isinstance(wav, np.ndarray):
        wav_tensor = torch.from_numpy(wav)
    elif isinstance(wav, torch.Tensor):
        wav_tensor = wav
    elif isinstance(wav, parselmouth.Sound):
        wav_np = wav.values
        wav_tensor = torch.from_numpy(wav_np)
    else:
        raise NotImplementedError
    return wav_tensor


def get_pitch_median(wav, sr: int = None):
    sound = wav_to_Sound(wav, sr)
    pitch = None
    pitch_median = PRAAT_CHANGEGENDER_PITCHMEDIAN_DEFAULT

    try:
        pitch = parselmouth.praat.call(sound, "To Pitch", 0.8 / 75, 75, 600)
        pitch_median = parselmouth.praat.call(pitch, "Get quantile", 0.0, 0.0, 0.5, "Hertz")
    except Exception as e:
        raise e
        pass

    return pitch, pitch_median


def change_gender(
        sound, pitch=None,
        formant_shift_ratio: float = PRAAT_CHANGEGENDER_FORMANTSHIFTRATIO_DEFAULT,
        new_pitch_median: float = PRAAT_CHANGEGENDER_PITCHMEDIAN_DEFAULT,
        pitch_range_ratio: float = PRAAT_CHANGEGENDER_PITCHRANGERATIO_DEFAULT,
        duration_factor: float = PRAAT_CHANGEGENDER_DURATIONFACTOR_DEFAULT, ) -> parselmouth.Sound:
    try:
        if pitch is None:
            new_sound = parselmouth.praat.call(
                sound, "Change gender", 75, 600,
                formant_shift_ratio,
                new_pitch_median,
                pitch_range_ratio,
                duration_factor
            )
        else:
            new_sound = parselmouth.praat.call(
                (sound, pitch), "Change gender",
                formant_shift_ratio,
                new_pitch_median,
                pitch_range_ratio,
                duration_factor
            )
    except Exception as e:
        raise e

    return new_sound


def apply_formant_and_pitch_shift(
        sound: parselmouth.Sound,
        formant_shift_ratio: float = PRAAT_CHANGEGENDER_FORMANTSHIFTRATIO_DEFAULT,
        pitch_shift_ratio: float = PRAAT_CHANGEGENDER_PITCHSHIFTRATIO_DEFAULT,
        pitch_range_ratio: float = PRAAT_CHANGEGENDER_PITCHRANGERATIO_DEFAULT,
        duration_factor: float = PRAAT_CHANGEGENDER_DURATIONFACTOR_DEFAULT) -> parselmouth.Sound:
    r"""uses praat 'Change Gender' backend to manipulate pitch and formant
        'Change Gender' function: praat -> Sound Object -> Convert -> Change Gender
        see Help of Praat for more details

        # https://github.com/YannickJadoul/Parselmouth/issues/25#issuecomment-608632887 might help
    """

    # pitch = sound.to_pitch()
    pitch = None
    new_pitch_median = PRAAT_CHANGEGENDER_PITCHMEDIAN_DEFAULT
    if pitch_shift_ratio != 1.:
        try:
            pitch, pitch_median = get_pitch_median(sound, None)
            new_pitch_median = pitch_median * pitch_shift_ratio

            # https://github.com/praat/praat/issues/1926#issuecomment-974909408
            pitch_minimum = parselmouth.praat.call(pitch, "Get minimum", 0.0, 0.0, "Hertz", "Parabolic")
            newMedian = pitch_median * pitch_shift_ratio
            scaledMinimum = pitch_minimum * pitch_shift_ratio
            resultingMinimum = newMedian + (scaledMinimum - newMedian) * pitch_range_ratio
            if resultingMinimum < 0:
                new_pitch_median = PRAAT_CHANGEGENDER_PITCHMEDIAN_DEFAULT
                pitch_range_ratio = PRAAT_CHANGEGENDER_PITCHRANGERATIO_DEFAULT

            if math.isnan(new_pitch_median):
                new_pitch_median = PRAAT_CHANGEGENDER_PITCHMEDIAN_DEFAULT
                pitch_range_ratio = PRAAT_CHANGEGENDER_PITCHRANGERATIO_DEFAULT

        except Exception as e:
            raise e

    new_sound = change_gender(
        sound, pitch,
        formant_shift_ratio, new_pitch_median,
        pitch_range_ratio, duration_factor)

    return new_sound


# fs & pr
def formant_and_pitch_shift(sound: parselmouth.Sound) -> parselmouth.Sound:
    r"""calculate random factors and apply formant and pitch shift

    designed for formant shifting(fs) and pitch randomization(pr) in the paper
    """
    formant_shifting_ratio = random.uniform(1, 1.4)
    use_reciprocal = random.uniform(-1, 1) > 0
    if use_reciprocal:
        formant_shifting_ratio = 1 / formant_shifting_ratio

    pitch_shift_ratio = random.uniform(1, 2)
    use_reciprocal = random.uniform(-1, 1) > 0
    if use_reciprocal:
        pitch_shift_ratio = 1 / pitch_shift_ratio

    pitch_range_ratio = random.uniform(1, 1.5)
    use_reciprocal = random.uniform(-1, 1) > 0
    if use_reciprocal:
        pitch_range_ratio = 1 / pitch_range_ratio

    sound_new = apply_formant_and_pitch_shift(
        sound,
        formant_shift_ratio=formant_shifting_ratio,
        pitch_shift_ratio=pitch_shift_ratio,
        pitch_range_ratio=pitch_range_ratio,
        duration_factor=1.
    )
    return sound_new


# fs
def formant_shift(sound: parselmouth.Sound) -> parselmouth.Sound:
    """designed for formant shifting(fs) in the paper

    Args:
        sound: parselmouth Sound object

    Returns:

    """
    formant_shifting_ratio = random.uniform(1, 1.4)
    use_reciprocal = random.uniform(-1, 1) > 0
    if use_reciprocal:
        formant_shifting_ratio = 1 / formant_shifting_ratio

    sound_new = apply_formant_and_pitch_shift(
        sound,
        formant_shift_ratio=formant_shifting_ratio,
    )
    return sound_new


def power_ratio(r: float, a: float, b: float):
    return a * math.pow((b / a), r)


# peq
def parametric_equalizer(wav: torch.Tensor, sr: int) -> torch.Tensor:
    cutoff_low_freq = 60.
    cutoff_high_freq = 10000.

    q_min = 2
    q_max = 5

    num_filters = 8 + 2  # 8 for peak, 2 for high/low
    key_freqs = [
        power_ratio(float(z) / (num_filters), cutoff_low_freq, cutoff_high_freq)
        for z in range(num_filters)
    ]
    Qs = [
        power_ratio(random.uniform(0, 1), q_min, q_max)
        for _ in range(num_filters)
    ]
    gains = [random.uniform(-12, 12) for _ in range(num_filters)]

    # peak filters
    for i in range(1, 9):
        wav = apply_iir_filter(
            wav,
            ftype='peak',
            dBgain=gains[i],
            cutoff_freq=key_freqs[i],
            sample_rate=sr,
            Q=Qs[i]
        )

    # high-shelving filter
    wav = apply_iir_filter(
        wav,
        ftype='high',
        dBgain=gains[-1],
        cutoff_freq=key_freqs[-1],
        sample_rate=sr,
        Q=Qs[-1]
    )

    # low-shelving filter
    wav = apply_iir_filter(
        wav,
        ftype='low',
        dBgain=gains[0],
        cutoff_freq=key_freqs[0],
        sample_rate=sr,
        Q=Qs[0]
    )

    return wav


# implemented using the cookbook https://webaudio.github.io/Audio-EQ-Cookbook/audio-eq-cookbook.html
def lowShelf_coeffs(dBgain, cutoff_freq, sample_rate, Q):
    A = math.pow(10, dBgain / 40.)

    w0 = 2 * math.pi * cutoff_freq / sample_rate
    alpha = math.sin(w0) / 2 / Q
    # alpha = alpha / math.sqrt(2) * math.sqrt(A + 1 / A)

    b0 = A * ((A + 1) - (A - 1) * math.cos(w0) + 2 * math.sqrt(A) * alpha)
    b1 = 2 * A * ((A - 1) - (A + 1) * math.cos(w0))
    b2 = A * ((A + 1) - (A - 1) * math.cos(w0) - 2 * math.sqrt(A) * alpha)

    a0 = (A + 1) + (A - 1) * math.cos(w0) + 2 * math.sqrt(A) * alpha
    a1 = -2 * ((A - 1) + (A + 1) * math.cos(w0))
    a2 = (A + 1) + (A - 1) * math.cos(w0) - 2 * math.sqrt(A) * alpha
    return b0, b1, b2, a0, a1, a2


def highShelf_coeffs(dBgain, cutoff_freq, sample_rate, Q):
    A = math.pow(10, dBgain / 40.)

    w0 = 2 * math.pi * cutoff_freq / sample_rate
    alpha = math.sin(w0) / 2 / Q
    # alpha = alpha / math.sqrt(2) * math.sqrt(A + 1 / A)

    b0 = A * ((A + 1) + (A - 1) * math.cos(w0) + 2 * math.sqrt(A) * alpha)
    b1 = -2 * A * ((A - 1) + (A + 1) * math.cos(w0))
    b2 = A * ((A + 1) + (A - 1) * math.cos(w0) - 2 * math.sqrt(A) * alpha)

    a0 = (A + 1) - (A - 1) * math.cos(w0) + 2 * math.sqrt(A) * alpha
    a1 = 2 * ((A - 1) - (A + 1) * math.cos(w0))
    a2 = (A + 1) - (A - 1) * math.cos(w0) - 2 * math.sqrt(A) * alpha
    return b0, b1, b2, a0, a1, a2


def peaking_coeffs(dBgain, cutoff_freq, sample_rate, Q):
    A = math.pow(10, dBgain / 40.)

    w0 = 2 * math.pi * cutoff_freq / sample_rate
    alpha = math.sin(w0) / 2 / Q
    # alpha = alpha / math.sqrt(2) * math.sqrt(A + 1 / A)

    b0 = 1 + alpha * A
    b1 = -2 * math.cos(w0)
    b2 = 1 - alpha * A

    a0 = 1 + alpha / A
    a1 = -2 * math.cos(w0)
    a2 = 1 - alpha / A
    return b0, b1, b2, a0, a1, a2


def apply_iir_filter(wav: torch.Tensor, ftype, dBgain, cutoff_freq, sample_rate, Q, torch_backend=True):
    if ftype == 'low':
        b0, b1, b2, a0, a1, a2 = lowShelf_coeffs(dBgain, cutoff_freq, sample_rate, Q)
    elif ftype == 'high':
        b0, b1, b2, a0, a1, a2 = highShelf_coeffs(dBgain, cutoff_freq, sample_rate, Q)
    elif ftype == 'peak':
        b0, b1, b2, a0, a1, a2 = peaking_coeffs(dBgain, cutoff_freq, sample_rate, Q)
    else:
        raise NotImplementedError
    if torch_backend:
        return_wav = AF.biquad(wav, b0, b1, b2, a0, a1, a2)
    else:
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.lfilter_zi.html
        wav_numpy = wav.numpy()
        b = np.asarray([b0, b1, b2])
        a = np.asarray([a0, a1, a2])
        zi = scipy.signal.lfilter_zi(b, a) * wav_numpy[0]
        return_wav, _ = scipy.signal.lfilter(b, a, wav_numpy, zi=zi)
        return_wav = torch.from_numpy(return_wav)
    return return_wav


peq = parametric_equalizer
fs = formant_shift


def nansy_g(wav: torch.Tensor, sr: int) -> torch.Tensor:
    r"""sequentially apply peq and fs


    """
    wav = peq(wav, sr)
    wav_numpy = wav.numpy()

    sound = wav_to_Sound(wav_numpy, sampling_frequency=sr)
    sound = formant_shift(sound)

    wav = torch.from_numpy(sound.values).float().squeeze(0)
    return wav


def nansy_f(wav: torch.Tensor, sr: int) -> torch.Tensor:
    r"""sequentially apply peq, pr and fs


    """
    wav = peq(wav, sr)
    wav_numpy = wav.numpy()

    sound = wav_to_Sound(wav_numpy, sampling_frequency=sr)
    sound = formant_and_pitch_shift(sound)

    wav = torch.from_numpy(sound.values).float().squeeze(0)
    return wav