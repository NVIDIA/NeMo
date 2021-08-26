# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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

from typing import Tuple

__all__ = ['is_url', 'has_numbers']


def is_url(input_str: str):
    """ Check if a string is a URL """
    url_segments = ['www', 'http', '.org', '.com', '.tv']
    return any(segment in input_str for segment in url_segments)


def has_numbers(input_str: str):
    """ Check if a string has a number character """
    return any(char.isdigit() for char in input_str)


def get_formatted_string(strs: Tuple[str], str_max_len: int = 10, space_len: int = 2):
    """ Get a nicely formatted string from a list of strings"""
    padded_strs = []
    for cur_str in strs:
        cur_str = cur_str + ' ' * (str_max_len - len(cur_str))
        padded_strs.append(cur_str[:str_max_len])

    spaces = ' ' * space_len
    return spaces.join(padded_strs)
