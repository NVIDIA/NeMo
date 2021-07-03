# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
# Copyright 2015 and onwards Google, Inc.
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

from nemo_text_processing.text_normalization.data_loader_utils import get_abs_path

try:
    import pynini
    from pynini.lib import pynutil

    PYNINI_AVAILABLE = True
except (ModuleNotFoundError, ImportError):
    PYNINI_AVAILABLE = False

try:
    RU_LOWER_ALPHA = "абвгдеёжзийклмнопрстуфхцчшщъыьэюя"
    RU_UPPER_ALPHA = RU_LOWER_ALPHA.upper()
    RU_LOWER_ALPHA = pynini.union(*RU_LOWER_ALPHA).optimize()
    RU_UPPER_ALPHA = pynini.union(*RU_UPPER_ALPHA).optimize()
    RU_ALPHA = (RU_LOWER_ALPHA | RU_UPPER_ALPHA).optimize()

    RU_STRESSED_MAP = [
        ("А́", "А'"),
        ("Е́", "Е'"),
        ("Ё́", "Е'"),
        ("И́", "И'"),
        ("О́", "О'"),
        ("У́", "У'"),
        ("Ы́", "Ы'"),
        ("Э́", "Э'"),
        ("Ю́", "Ю'"),
        ("Я́", "Я'"),
        ("а́", "а'"),
        ("е́", "е'"),
        ("ё́", "е'"),
        ("и́", "и'"),
        ("о́", "о'"),
        ("у́", "у'"),
        ("ы́", "ы'"),
        ("э́", "э'"),
        ("ю́", "ю'"),
        ("я́", "я'"),
        ("ё", "е"),
        ("Ё", "Е"),
    ]

    REWRITE_STRESSED = pynini.closure(pynini.string_map(RU_STRESSED_MAP).optimize() | RU_ALPHA).optimize()

    TO_LATIN = pynini.string_file(get_abs_path("ru/data/cyrillic_to_latin.tsv"))

except (ModuleNotFoundError, ImportError):
    # Create placeholders
    RU_ALPHA = None
    LO_LATIN = None
