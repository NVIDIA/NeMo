# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import librosa
import soundfile as sf


def get_samples(audio_file: str, target_sr: int = 16000, dtype: str = 'float32'):
    """
    Read the samples from the given audio_file path. If not specified, the input audio file is automatically
    resampled to 16kHz.

    Args:
        audio_file (str):
            Path to the input audio file
        target_sr (int):
            Targeted sampling rate
    Returns:
        samples (numpy.ndarray):
            Time-series sample data from the given audio file
    """
    with sf.SoundFile(audio_file, 'r') as f:
        samples = f.read(dtype=dtype)
        if f.samplerate != target_sr:
            samples = librosa.core.resample(samples, orig_sr=f.samplerate, target_sr=target_sr)
        samples = samples.transpose()
    return samples
