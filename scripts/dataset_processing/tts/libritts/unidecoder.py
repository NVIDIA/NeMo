# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import io
import warnings

from .homoglyphs import homoglyphs
from .replacements import replacements

warnings.simplefilter('always', UserWarning)


_replacements = {uni: asc for uni, asc in replacements}
_homoglyphs = {g: asc for asc, glyphs in homoglyphs.items() for g in glyphs}


def unidecoder(s, homoglyphs=False):
    """Transliterate unicode

    Args:
        s (str): unicode string
        homoglyphs (bool): prioritize translating to homoglyphs
    """
    warned = False  # Once per utterance
    ret = ''
    for u in s:
        if ord(u) < 127:
            a = u
        elif homoglyphs:
            a = _homoglyphs.get(u, _replacements.get(u, None))
        else:
            a = _replacements.get(u, _homoglyphs.get(u, None))

        if a is None:
            if not warned:
                warnings.warn(f'Unexpected character {u}: '
                              'please revise your text cleaning rules.',
                              stacklevel=10**6)
                warned = True
        else:
            ret += a

    return ret
