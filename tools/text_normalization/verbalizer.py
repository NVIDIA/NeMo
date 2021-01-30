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

# Copyright (c) 2017 Keith Ito

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
""" partly adapted from https://github.com/keithito/tacotron """

import csv
import os
from collections import OrderedDict

import inflect
import regex as re

_inflect = inflect.engine()

month_tsv = open(os.path.dirname(os.path.abspath(__file__)) + "/data/months.tsv")
read_tsv = csv.reader(month_tsv, delimiter="\t")
_month_dict = dict(read_tsv)

_date_components_whitelist = {"month", "day", "year", "suffix"}
_roman_numerals = {'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100, 'D': 500, 'M': 1000}
_magnitudes = ['trillion', 'billion', 'million', 'thousand', 'hundred', 'k', 'm', 'b', 't']

_magnitudes_tsv = open(os.path.dirname(os.path.abspath(__file__)) + "/data/magnitudes.tsv")
read_tsv = csv.reader(_magnitudes_tsv, delimiter="\t")
_magnitudes_dict = dict(read_tsv)

_currency_tsv = open(os.path.dirname(os.path.abspath(__file__)) + "/data/currency.tsv")
read_tsv = csv.reader(_currency_tsv, delimiter="\t")
_currency_dict = dict(read_tsv)

_measurements_tsv = open(os.path.dirname(os.path.abspath(__file__)) + "/data/measurements.tsv")
read_tsv = csv.reader(_measurements_tsv, delimiter="\t")
_measurements_dict = dict(read_tsv)

_whitelist_tsv = open(os.path.dirname(os.path.abspath(__file__)) + "/data/whitelist.tsv")
read_tsv = csv.reader(_whitelist_tsv, delimiter="\t")
_whitelist_dict = dict(read_tsv)


def expand_telephone(data: dict) -> str:
    raise NotImplementedError


def expand_punct(data: dict) -> str:
    raise NotImplementedError


def expand_letter(data: dict) -> str:
    raise NotImplementedError


def expand_fraction(data: dict) -> str:
    raise NotImplementedError


def expand_electronic(data: dict) -> str:
    raise NotImplementedError


def expand_digit(data: dict) -> str:
    raise NotImplementedError


def expand_verbatim(data: dict) -> str:
    """
    Verbalizes verbatim tokens.
    Args:
        data: detected data
    Returns string
    """
    return "and"


def expand_whitelist(data: dict) -> str:
    """
    Verbalizes whitelisted tokens.
    Args:
        data: detected data
    Returns string
    """
    return _whitelist_dict[data["value"]]


def expand_decimal(data: dict) -> str:
    """
    Verbalizes decimal tokens.
    Args:
        data: detected data
    Returns string
    """
    return _inflect.number_to_words(data["value"]).replace("-", " ").replace(" and ", " ").replace(",", "")


def expand_roman(data: dict) -> str:
    """
    Verbalizes roman numerals.
    Args:
        data: detected data
    Returns string
    """
    num = data.get("number_roman")
    if not num:
        return None
    result = 0
    for i, c in enumerate(num):
        if (i + 1) == len(num) or _roman_numerals[c] >= _roman_numerals[num[i + 1]]:
            result += _roman_numerals[c]
        else:
            result -= _roman_numerals[c]
    return _inflect.number_to_words(result).replace("-", " ").replace(" and ", " ").replace(",", "")


def expand_cardinal(data: dict) -> str:
    """
    Verbalizes cardinal data.
    Args:
        data: detected data
    Returns string
    """
    if not data.get("number"):
        return None
    return _inflect.number_to_words(data["number"]).replace("-", " ").replace(" and ", " ").replace(",", "")


def expand_ordinal(data: dict) -> str:
    """
    Verbalizes ordinal data.
    Args:
        data: detected data
    Returns string
    """
    num = data.get("value")
    if not num:
        return None
    result = _inflect.number_to_words(num + "th")
    return result.replace("-", " ").replace(" and ", " ").replace(",", "")


def expand_year(data: dict) -> str:
    """
    Verbalizes measurement data.
    Args:
        data: detected data
    Returns string
    """
    if data["value"] is None:
        return None
    number = int(data["value"])
    result = ""
    if number > 1000 and number < 3000:
        if number == 2000:
            result = 'two thousand'
        elif number > 2000 and number < 2010:
            result = 'two thousand ' + _inflect.number_to_words(number % 100)
        elif number % 100 == 0:
            result = _inflect.number_to_words(number // 100) + ' hundred'
        else:
            number = _inflect.number_to_words(number, andword='', zero='o', group=2).replace(', ', ' ')
            number = re.sub(r'-', ' ', number)
            result = number
    else:
        result = expand_cardinal({"number": data["value"]})
    return result


def expand_date(data: dict) -> str:
    """
    Verbalizes date data.
    Args:
        data: detected data
    Returns string
    """
    res = {x: y for x, y in data.items()}
    YEAR = "year"
    MONTH = "month"
    SUFFIX = "suffix"
    DAY = "day"
    try:
        res[MONTH] = month_mapping[data[MONTH]]
    except Exception:
        pass
    try:
        res[DAY] = expand_ordinal({"value": data[DAY]})
    except Exception:
        pass
    try:
        res[YEAR] = expand_year({"value": data[YEAR]})
    except Exception:
        pass
    res = {k: res[k] for k in res if k in _date_components_whitelist}
    meta_list = OrderedDict([x for x in data.items() if x[0] in _date_components_whitelist])
    result = None
    if [*meta_list] == [YEAR, MONTH, DAY]:
        result = "the " + res[DAY] + " of " + res[MONTH] + " " + res[YEAR]
    elif [*meta_list] == [MONTH, DAY, YEAR]:
        result = res[MONTH] + " " + res[DAY] + " " + res[YEAR]
    elif [*meta_list] == [DAY, MONTH, YEAR]:
        result = "the " + res[DAY] + ' of ' + res[MONTH] + " " + res[YEAR]
    elif [*meta_list] == [MONTH, DAY]:
        result = res[MONTH] + " " + res[DAY]
    elif [*meta_list] == [MONTH, YEAR]:
        result = res[MONTH] + " " + res[YEAR]
    elif [*meta_list] == [DAY, MONTH]:
        result = 'the ' + res[DAY] + ' of ' + res[MONTH]
    elif [*meta_list] == [YEAR, SUFFIX]:
        result = res[YEAR][:-1] + 'ies' if res[YEAR][-1] == 'y' else res[YEAR] + 's'
    elif [*meta_list] == [YEAR]:
        result = res[YEAR]
    return result.replace("-", " ") if result else None


def _expand_hundreds(text):
    number = float(text)
    if 1000 < number < 10000 and (number % 100 == 0) and (number % 1000 != 0):
        return _inflect.number_to_words(int(number / 100)) + " hundred"
    else:
        return _inflect.number_to_words(text)


def _expand_currency(data: dict) -> str:
    """
    Verbalizes currency tokens.
    Args:
        data: detected data
    Returns string
    """
    currency = _currency_dict[data['currency']]
    quantity = data['integral'] + ('.' + data['fractional'] if data.get('fractional') else '')
    magnitude = data.get('magnitude')

    # remove commas from quantity to be able to convert to numerical
    quantity = quantity.replace(',', '')

    # check for million, billion, etc...
    if magnitude is not None and magnitude.lower() in _magnitudes:
        if len(magnitude) == 1:
            magnitude = _magnitudes_dict[magnitude.lower()]
        return "{} {} {}".format(_expand_hundreds(quantity), magnitude, currency + 's')

    parts = quantity.split('.')
    if len(parts) > 2:
        return quantity + " " + currency + "s"  # Unexpected format

    dollars = int(parts[0]) if parts[0] else 0

    cents = int(parts[1]) if len(parts) > 1 and parts[1] else 0
    if dollars and cents:
        dollar_unit = currency if dollars == 1 else currency + 's'
        cent_unit = 'cent' if cents == 1 else 'cents'
        return "{} {}, {} {}".format(
            _expand_hundreds(dollars), dollar_unit, _inflect.number_to_words(cents), cent_unit
        )
    elif dollars:
        dollar_unit = currency if dollars == 1 else currency + 's'
        return "{} {}".format(_expand_hundreds(dollars), dollar_unit)
    elif cents:
        cent_unit = 'cent' if cents == 1 else 'cents'
        return "{} {}".format(_inflect.number_to_words(cents), cent_unit)
    else:
        return 'zero' + ' ' + currency + 's'


def expand_money(data: dict) -> str:
    """
    Verbalizes money data.
    Args:
        data: detected data
    Returns string
    """
    result = _expand_currency(data)
    return result.replace(',', '').replace('-', ' ').replace(' and ', ' ')


def expand_measurement(data: dict) -> str:
    """
    Verbalizes measurement data.
    Args:
        data: detected data
    Returns string
    """
    value = float(data["decimal"].replace(",", ""))
    value_verb = _inflect.number_to_words(data["decimal"]).replace(',', '').replace('-', ' ').replace(' and ', ' ')
    res = value_verb
    if data.get("measurement"):
        measure = _measurements_dict[data["measurement"]]
        if value <= 1 and measure[-1] == 's':
            measure = measure[:-1]
        res += " " + measure

    if data.get("measurement2"):
        res += " per "
        measure2 = _measurements_dict[data["measurement2"]]
        # if measure2[-1] == 's':
        #     measure2 = measure2[:-1]
        res += measure2
    return res


def expand_time(data: dict) -> str:
    """
    Verbalizes time data.
    Args:
        data: detected data
    Returns string
    """
    res = _inflect.number_to_words(data["hour"])
    if data.get("minutes") and int(data["minutes"]) != 0:
        if data["minutes"][0] == "0":
            res += " o " + _inflect.number_to_words(data["minutes"])
        else:
            res += " " + _inflect.number_to_words(data["minutes"])
    else:
        if not data.get("suffix"):
            res += " o'clock"

    if data.get("suffix"):
        res += " " + " ".join(list(data["suffix"].replace(".", "")))
    return res.replace("-", " ")
