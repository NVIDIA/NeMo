# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

import numpy as np
from scipy.stats import betabinom


def beta_binomial_prior_distribution(phoneme_count, mel_count, scaling_factor=1.0):
    x = np.arange(0, phoneme_count)
    mel_text_probs = []
    for i in range(1, mel_count + 1):
        a, b = scaling_factor * i, scaling_factor * (mel_count + 1 - i)
        mel_i_prob = betabinom(phoneme_count, a, b).pmf(x)
        mel_text_probs.append(mel_i_prob)
    return np.array(mel_text_probs)
