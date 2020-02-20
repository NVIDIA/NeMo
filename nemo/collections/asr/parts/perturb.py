# Taken straight from Patter https://github.com/ryanleary/patter
# TODO: review, and copyright and fix/add comments
import random

import librosa
import numpy as np
from scipy import signal

from nemo import logging
from nemo.collections.asr.parts import collections, parsers
from nemo.collections.asr.parts.segment import AudioSegment


class Perturbation(object):
    def max_augmentation_length(self, length):
        return length

    def perturb(self, data):
        raise NotImplementedError


class SpeedPerturbation(Perturbation):
    def __init__(self, min_speed_rate=0.85, max_speed_rate=1.15, rng=None):
        self._min_rate = min_speed_rate
        self._max_rate = max_speed_rate
        self._rng = random.Random() if rng is None else rng

    def max_augmentation_length(self, length):
        return length * self._max_rate

    def perturb(self, data):
        speed_rate = self._rng.uniform(self._min_rate, self._max_rate)
        if speed_rate <= 0:
            raise ValueError("speed_rate should be greater than zero.")
        logging.debug("speed: %f", speed_rate)
        data._samples = librosa.effects.time_stretch(data._samples, speed_rate)


class GainPerturbation(Perturbation):
    def __init__(self, min_gain_dbfs=-10, max_gain_dbfs=10, rng=None):
        self._min_gain_dbfs = min_gain_dbfs
        self._max_gain_dbfs = max_gain_dbfs
        self._rng = random.Random() if rng is None else rng

    def perturb(self, data):
        gain = self._rng.uniform(self._min_gain_dbfs, self._max_gain_dbfs)
        logging.debug("gain: %d", gain)
        data._samples = data._samples * (10.0 ** (gain / 20.0))


class ImpulsePerturbation(Perturbation):
    def __init__(self, manifest_path=None, rng=None):
        self._manifest = collections.ASRAudioText(manifest_path, parser=parsers.make_parser([]))
        self._rng = random.Random() if rng is None else rng

    def perturb(self, data):
        impulse_record = self._rng.sample(self._manifest.data, 1)[0]
        impulse = AudioSegment.from_file(impulse_record['audio_filepath'], target_sr=data.sample_rate)
        logging.debug("impulse: %s", impulse_record['audio_filepath'])
        data._samples = signal.fftconvolve(data.samples, impulse.samples, "full")


class ShiftPerturbation(Perturbation):
    def __init__(self, min_shift_ms=-5.0, max_shift_ms=5.0, rng=None):
        self._min_shift_ms = min_shift_ms
        self._max_shift_ms = max_shift_ms
        self._rng = random.Random() if rng is None else rng

    def perturb(self, data):
        shift_ms = self._rng.uniform(self._min_shift_ms, self._max_shift_ms)
        if abs(shift_ms) / 1000 > data.duration:
            # TODO: do something smarter than just ignore this condition
            return
        shift_samples = int(shift_ms * data.sample_rate // 1000)
        logging.debug("shift: %s", shift_samples)
        if shift_samples < 0:
            data._samples[-shift_samples:] = data._samples[:shift_samples]
            data._samples[:-shift_samples] = 0
        elif shift_samples > 0:
            data._samples[:-shift_samples] = data._samples[shift_samples:]
            data._samples[-shift_samples:] = 0


class NoisePerturbation(Perturbation):
    def __init__(
        self, manifest_path=None, min_snr_db=40, max_snr_db=50, max_gain_db=300.0, rng=None,
    ):
        self._manifest = collections.ASRAudioText(manifest_path, parser=parsers.make_parser([]))
        self._rng = random.Random() if rng is None else rng
        self._min_snr_db = min_snr_db
        self._max_snr_db = max_snr_db
        self._max_gain_db = max_gain_db

    def perturb(self, data):
        snr_db = self._rng.uniform(self._min_snr_db, self._max_snr_db)
        noise_record = self._rng.sample(self._manifest.data, 1)[0]
        noise = AudioSegment.from_file(noise_record['audio_filepath'], target_sr=data.sample_rate)
        noise_gain_db = min(data.rms_db - noise.rms_db - snr_db, self._max_gain_db)
        logging.debug("noise: %s %s %s", snr_db, noise_gain_db, noise_record['audio_filepath'])

        # calculate noise segment to use
        start_time = self._rng.uniform(0.0, noise.duration - data.duration)
        noise.subsegment(start_time=start_time, end_time=start_time + data.duration)

        # adjust gain for snr purposes and superimpose
        noise.gain_db(noise_gain_db)
        data._samples = data._samples + noise.samples


class WhiteNoisePerturbation(Perturbation):
    def __init__(self, min_level=-90, max_level=-46, rng=None):
        self.min_level = int(min_level)
        self.max_level = int(max_level)
        self._rng = np.random.RandomState() if rng is None else rng

    def perturb(self, data):
        noise_level_db = self._rng.randint(self.min_level, self.max_level, dtype='int32')
        noise_signal = self._rng.randn(data._samples.shape[0]) * (10.0 ** (noise_level_db / 20.0))
        data._samples += noise_signal


perturbation_types = {
    "speed": SpeedPerturbation,
    "gain": GainPerturbation,
    "impulse": ImpulsePerturbation,
    "shift": ShiftPerturbation,
    "noise": NoisePerturbation,
    'white_noise': WhiteNoisePerturbation,
}


class AudioAugmentor(object):
    def __init__(self, perturbations=None, rng=None):
        self._rng = random.Random() if rng is None else rng
        self._pipeline = perturbations if perturbations is not None else []

    def perturb(self, segment):
        for (prob, p) in self._pipeline:
            if self._rng.random() < prob:
                p.perturb(segment)
        return

    def max_augmentation_length(self, length):
        newlen = length
        for (prob, p) in self._pipeline:
            newlen = p.max_augmentation_length(newlen)
        return newlen

    @classmethod
    def from_config(cls, config):
        ptbs = []
        for p in config:
            if p['aug_type'] not in perturbation_types:
                logging.warning("%s perturbation not known. Skipping.", p['aug_type'])
                continue
            perturbation = perturbation_types[p['aug_type']]
            ptbs.append((p['prob'], perturbation(**p['cfg'])))
        return cls(perturbations=ptbs)
