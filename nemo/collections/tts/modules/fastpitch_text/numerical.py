""" adapted from https://github.com/keithito/tacotron """

import re

import inflect

_magnitudes = ['trillion', 'billion', 'million', 'thousand', 'hundred', 'm', 'b', 't']
_magnitudes_key = {'m': 'million', 'b': 'billion', 't': 'trillion'}
_measurements = '(f|c|k|d|m)'
_measurements_key = {'f': 'fahrenheit', 'c': 'celsius', 'k': 'thousand', 'm': 'meters'}
_currency_key = {'$': 'dollar', '£': 'pound', '€': 'euro', '₩': 'won'}
_inflect = inflect.engine()
_comma_number_re = re.compile(r'([0-9][0-9\,]+[0-9])')
_decimal_number_re = re.compile(r'([0-9]+\.[0-9]+)')
_currency_re = re.compile(
    r'([\$€£₩])([0-9\.\,]*[0-9]+)(?:[ ]?({})(?=[^a-zA-Z]))?'.format("|".join(_magnitudes)), re.IGNORECASE
)
_measurement_re = re.compile(r'([0-9\.\,]*[0-9]+(\s)?{}\b)'.format(_measurements), re.IGNORECASE)
_ordinal_re = re.compile(r'[0-9]+(st|nd|rd|th)')
# _range_re = re.compile(r'(?<=[0-9])+(-)(?=[0-9])+.*?')
_roman_re = re.compile(r'\b(?=[MDCLXVI]+\b)M{0,4}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{2,3})\b')  # avoid I
_multiply_re = re.compile(r'(\b[0-9]+)(x)([0-9]+)')
_number_re = re.compile(r"[0-9]+'s|[0-9]+s|[0-9]+")


def _remove_commas(m):
    return m.group(1).replace(',', '')


def _expand_decimal_point(m):
    return m.group(1).replace('.', ' point ')


def _expand_currency(m):
    currency = _currency_key[m.group(1)]
    quantity = m.group(2)
    magnitude = m.group(3)

    # remove commas from quantity to be able to convert to numerical
    quantity = quantity.replace(',', '')

    # check for million, billion, etc...
    if magnitude is not None and magnitude.lower() in _magnitudes:
        if len(magnitude) == 1:
            magnitude = _magnitudes_key[magnitude.lower()]
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


def _expand_hundreds(text):
    number = float(text)
    if 1000 < number < 10000 and (number % 100 == 0) and (number % 1000 != 0):
        return _inflect.number_to_words(int(number / 100)) + " hundred"
    else:
        return _inflect.number_to_words(text)


def _expand_ordinal(m):
    return _inflect.number_to_words(m.group(0))


def _expand_measurement(m):
    _, number, measurement = re.split('(\d+(?:\.\d+)?)', m.group(0))
    number = _inflect.number_to_words(number)
    measurement = "".join(measurement.split())
    measurement = _measurements_key[measurement.lower()]
    return "{} {}".format(number, measurement)


def _expand_range(m):
    return ' to '


def _expand_multiply(m):
    left = m.group(1)
    right = m.group(3)
    return "{} by {}".format(left, right)


def _expand_roman(m):
    # from https://stackoverflow.com/questions/19308177/converting-roman-numerals-to-integers-in-python
    roman_numerals = {'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100, 'D': 500, 'M': 1000}
    result = 0
    num = m.group(0)
    for i, c in enumerate(num):
        if (i + 1) == len(num) or roman_numerals[c] >= roman_numerals[num[i + 1]]:
            result += roman_numerals[c]
        else:
            result -= roman_numerals[c]
    return str(result)


def _expand_number(m):
    _, number, suffix = re.split(r"(\d+(?:'?\d+)?)", m.group(0))
    number = int(number)
    if number > 1000 < 10000 and (number % 100 == 0) and (number % 1000 != 0):
        text = _inflect.number_to_words(number // 100) + " hundred"
    elif number > 1000 and number < 3000:
        if number == 2000:
            text = 'two thousand'
        elif number > 2000 and number < 2010:
            text = 'two thousand ' + _inflect.number_to_words(number % 100)
        elif number % 100 == 0:
            text = _inflect.number_to_words(number // 100) + ' hundred'
        else:
            number = _inflect.number_to_words(number, andword='', zero='oh', group=2).replace(', ', ' ')
            number = re.sub(r'-', ' ', number)
            text = number
    else:
        number = _inflect.number_to_words(number, andword='and')
        number = re.sub(r'-', ' ', number)
        number = re.sub(r',', '', number)
        text = number

    if suffix in ("'s", "s"):
        if text[-1] == 'y':
            text = text[:-1] + 'ies'
        else:
            text = text + suffix

    return text


def normalize_numbers(text):
    text = re.sub(_comma_number_re, _remove_commas, text)
    text = re.sub(_currency_re, _expand_currency, text)
    text = re.sub(_decimal_number_re, _expand_decimal_point, text)
    text = re.sub(_ordinal_re, _expand_ordinal, text)
    # text = re.sub(_range_re, _expand_range, text)
    # text = re.sub(_measurement_re, _expand_measurement, text)
    text = re.sub(_roman_re, _expand_roman, text)
    text = re.sub(_multiply_re, _expand_multiply, text)
    text = re.sub(_number_re, _expand_number, text)
    return text
