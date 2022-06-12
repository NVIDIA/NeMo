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

from omegaconf.dictconfig import DictConfig


def get_coeff_from_cfg(cfg: DictConfig):
    enc_num_layers = (cfg.get('enc_num_layers', 4),)  # total number of encoder layers
    dec_num_layers = (cfg.get('dec_num_layers', 6),)  # total number of decoder layers
    enc_cross_attention = (cfg.get('enc_cross_attention', [3]),)  # layer numbers for cross attention
    dec_cross_attention = (cfg.get('dec_cross_attention', [3, 5]),)  # layer numbers for chunked cross attention
    return get_coeff(enc_num_layers, dec_num_layers, enc_cross_attention, dec_cross_attention)


def get_coeff(enc_num_layers, dec_num_layers, enc_cross_attention, dec_cross_attention):
    N = enc_num_layers
    M = dec_num_layers
    P = len(dec_cross_attention)
    Q = len(enc_cross_attention)
    c = min(dec_cross_attention) + 1

    c0 = 2 * M + P
    c1 = P * (2 * N + Q)
    c2 = P * Q * (2 * (c - 1) + 1)
    # c3 = c2 ** 4 * c0 ** (-5) * c1 ** (-2)
    c4 = 2 ** (-1 / 2) * c2 * c0 ** (-7 / 4)
    c5 = 2 ** (1 / 2) * c1 * c0 ** (-3 / 4)

    a_d = 2 ** (1 / 2) * c0 ** (1 / 4)
    b_d = c0 ** (-1 / 4)
    a_e = c5 / c4
    b_e = c5 ** (1 / 2) / c4
    # beta = (8 * c3 ** (-1)) ** (1 / 6)
    # a_d = 2 ** (1 / 2) * (c0 ** (1 / 2)) * beta
    # a_e = (16 * c1 ** 2 * c3 ** -1 * c0 ** -1) ** (1 / 4)
    return {"b_d": b_d, "b_e": b_e, "a_d": a_d, "a_e": a_e}
