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

__all__ = ['LATIN_TO_RU', 'RU_ABBREVIATIONS', 'NUMBERS_TO_ENG', 'NUMBERS_TO_RU']

LATIN_TO_RU = {
    'a': 'а',
    'b': 'б',
    'c': 'к',
    'd': 'д',
    'e': 'е',
    'f': 'ф',
    'g': 'г',
    'h': 'х',
    'i': 'и',
    'j': 'ж',
    'k': 'к',
    'l': 'л',
    'm': 'м',
    'n': 'н',
    'o': 'о',
    'p': 'п',
    'q': 'к',
    'r': 'р',
    's': 'с',
    't': 'т',
    'u': 'у',
    'v': 'в',
    'w': 'в',
    'x': 'к',
    'y': 'у',
    'z': 'з',
    'à': 'а',
    'è': 'е',
    'é': 'е',
}
RU_ABBREVIATIONS = {
    ' р.': ' рублей',
    ' к.': ' копеек',
    ' коп.': ' копеек',
    ' копек.': ' копеек',
    ' т.д.': ' так далее',
    ' т. д.': ' так далее',
    ' т.п.': ' тому подобное',
    ' т. п.': ' тому подобное',
    ' т.е.': ' то есть',
    ' т. е.': ' то есть',
    ' стр. ': ' страница ',
}
NUMBERS_TO_ENG = {
    '0': 'zero ',
    '1': 'one ',
    '2': 'two ',
    '3': 'three ',
    '4': 'four ',
    '5': 'five ',
    '6': 'six ',
    '7': 'seven ',
    '8': 'eight ',
    '9': 'nine ',
}

NUMBERS_TO_RU = {
    '0': 'ноль ',
    '1': 'один ',
    '2': 'два ',
    '3': 'три ',
    '4': 'четыре ',
    '5': 'пять ',
    '6': 'шесть ',
    '7': 'семь ',
    '8': 'восемь ',
    '9': 'девять ',
}
