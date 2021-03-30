# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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

import random

import librosa
import numpy as np
import pytest
import torch

from nemo.collections.asr.parts.features import FilterbankFeatures


class TestFilterbankFeatures:
    @pytest.mark.unit
    def test_seq_len(self):
        fb_module = FilterbankFeatures(exact_pad=False, pad_to=1)
        test_1 = torch.randn(1, 800)
        test_1_len = torch.tensor([800])
        result, result_len = fb_module(test_1, test_1_len)
        assert result.shape[2] == result_len[0], f"{result.shape} != {result_len}"
        spec = librosa.stft(test_1.cpu().detach().numpy().squeeze(), n_fft=512, hop_length=160, win_length=320)

        assert spec.shape[1] == result.shape[2], f"{result.shape} != {spec.shape}"

    @pytest.mark.unit
    def test_random_stft_sizes(self):
        for _ in range(5):
            nfft = random.randint(128, 2048)
            window_size = random.randint(128, nfft)
            hop_size = random.randint(64, window_size)
            fb_module = FilterbankFeatures(
                exact_pad=False, pad_to=1, n_fft=nfft, n_window_size=window_size, n_window_stride=hop_size
            )
            audio_length = random.randint(nfft, 2 ** 16)
            test_1 = torch.randn(1, audio_length)
            test_1_len = torch.tensor([audio_length])
            result, result_len = fb_module(test_1, test_1_len)
            assert (
                result.shape[2] == result_len[0]
            ), f"{result.shape} != {result_len}: {nfft}, {window_size}, {hop_size}, {audio_length}"

            spec = librosa.stft(
                test_1.cpu().detach().numpy().squeeze(), n_fft=nfft, hop_length=hop_size, win_length=window_size
            )

            assert (
                spec.shape[1] == result.shape[2]
            ), f"{result.shape} != {spec.shape}: {nfft}, {window_size}, {hop_size}, {audio_length}"

        for _ in range(5):
            nfft = random.randint(128, 2048)
            window_size = random.randint(128, nfft)
            hop_size = random.randint(64, window_size)
            fb_module = FilterbankFeatures(
                exact_pad=True, pad_to=1, n_fft=nfft, n_window_size=window_size, n_window_stride=hop_size
            )
            audio_length = random.randint(nfft, 2 ** 16)
            test_1 = torch.randn(1, audio_length)
            test_1_len = torch.tensor([audio_length])
            result, result_len = fb_module(test_1, test_1_len)
            assert (
                result.shape[2] == result_len[0]
            ), f"{result.shape} != {result_len}: {nfft}, {window_size}, {hop_size}, {audio_length}"

            test_2 = test_1.cpu().detach().numpy().squeeze()
            test_2 = np.pad(test_2, int((window_size - hop_size) // 2), mode="reflect")
            spec = librosa.stft(test_2, n_fft=nfft, hop_length=hop_size, win_length=window_size, center=False,)

            assert (
                spec.shape[1] == result.shape[2]
            ), f"{result.shape} != {spec.shape}: {nfft}, {window_size}, {hop_size}, {audio_length}"
