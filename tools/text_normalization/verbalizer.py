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

import csv
import os

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


def expand_whitelist(data: dict) -> str:
    """
    Verbalizes whitelisted tokens.
    Args:
        data: detected data
    Returns string
    """
    return _whitelist_dict[data["value"]]


def expand_roman(data: dict) -> str:
    """
    Verbalizes roman numerals.
    Args:
        data: detected data
    Returns string
    """
    num = data["value"]
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
    return _inflect.number_to_words(data["value"]).replace("-", " ").replace(" and ", " ").replace(",", "")


def expand_ordinal(data: dict) -> str:
    """
    Verbalizes ordinal data.
    Args:
        data: detected data
    Returns string
    """
    if data["value"] is None:
        return None
    result = _inflect.number_to_words(data["value"] + "th")
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
        result = expand_cardinal({"value": data["value"]})
    return result


def expand_date(data: dict, verbalize: object) -> str:
    """
    Verbalizes date data.
    Args:
        data: detected data
        verbalize: verbalization function
    Returns string
    """
    try:
        data["month"] = _month_dict[data["month"]]
    except Exception:
        pass
    try:
        data["day"] = expand_ordinal({"value": data["day"]})
    except Exception:
        pass
    try:
        data["year"] = expand_year({"value": data["year"]})
    except Exception:
        pass
    data = {k: data[k] for k in data if k in _date_components_whitelist}
    result = verbalize(**data)
    return result.replace("-", " ")


def _expand_hundreds(text):
    number = float(text)
    if 1000 < number < 10000 and (number % 100 == 0) and (number % 1000 != 0):
        return _inflect.number_to_words(int(number / 100)) + " hundred"
    else:
        return _inflect.number_to_words(text)


def _expand_currency(data):
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
