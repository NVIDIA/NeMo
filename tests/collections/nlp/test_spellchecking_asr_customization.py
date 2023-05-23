# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

import pytest

from nemo.collections.nlp.data.spellchecking_asr_customization.utils import (
    apply_replacements_to_text,
    substitute_replacements_in_text,
)


@pytest.mark.unit
def test_substitute_replacements_in_text():
    text = "we began the further diversification of our revenue base with the protterra supply agreement and the navastar joint development agreement"
    replacements = [(66, 75, 'pro-terra', 0.99986), (101, 109, 'navistar', 0.996)]
    gold_text = "we began the further diversification of our revenue base with the pro-terra supply agreement and the navistar joint development agreement"
    corrected_text = substitute_replacements_in_text(text, replacements, replace_hyphen_to_space=False)
    assert corrected_text == gold_text

    gold_text_no_hyphen = "we began the further diversification of our revenue base with the pro terra supply agreement and the navistar joint development agreement"
    corrected_text = substitute_replacements_in_text(text, replacements, replace_hyphen_to_space=True)
    assert corrected_text == gold_text_no_hyphen


@pytest.mark.unit
def test_apply_replacements_to_text():

    # min_prob = 0.5
    # dp_data = None,
    # min_dp_score_per_symbol: float = -99.9

    # test more than one fragment to replace, test multiple same replacements
    text = "we began the further diversification of our revenue base with the protterra supply agreement and the navastar joint development agreement"
    replacements = [
        (66, 75, 'proterra', 0.99986),
        (66, 75, 'proterra', 0.9956),
        (101, 109, 'navistar', 0.93),
        (101, 109, 'navistar', 0.91),
        (101, 109, 'navistar', 0.92),
    ]
    gold_text = "we began the further diversification of our revenue base with the proterra supply agreement and the navistar joint development agreement"
    corrected_text = apply_replacements_to_text(
        text, replacements, min_prob=0.5, replace_hyphen_to_space=False, dp_data=None
    )
    assert corrected_text == gold_text

    # test that min_prob works
    gold_text = "we began the further diversification of our revenue base with the proterra supply agreement and the navastar joint development agreement"
    corrected_text = apply_replacements_to_text(
        text, replacements, min_prob=0.95, replace_hyphen_to_space=False, dp_data=None
    )
    assert corrected_text == gold_text
