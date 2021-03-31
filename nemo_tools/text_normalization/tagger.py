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


import regex as re
from nemo_tools.text_normalization.tag import Tag, TagType
from nemo_tools.text_normalization.verbalizer import _currency_dict, _measurements_dict, _whitelist_dict


def make_re(re_inner: str, *args):
    """
    Takes given regex expression and decorates it with left and right word boundaries
    Args:
        re_inner: regex
        args: list of optional regex arguments
    Returns compiled regex
    """
    return re.compile(rf'{_re_left_boundary}(?P<value>{re_inner}){_re_right_boundary}', *args)


'''
List of regex for detection
'''

_re_left_boundary = r'(^|[\s\(\[\{\<\'\"\`])'
_re_right_boundary = r'($|(\s|\)|\]|\}|\>|(\'|\"|\`|\.|\,|\;|\:|\?|\!)([^\w]|$)))'
_re_time_hour = r'[0-1]?[0-9]|2[0-3]'
_re_date_month = r'0?[1-9]|1[012]'
_re_date_month2 = r'(Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sept|Sep|Oct|Nov|Dec)\.?|January|February|March|April|May|June|July|August|September|October|November|December'
_re_date_year = r'\d{4}'
_re_date_day = r'0?[1-9]|[12][0-9]|3[01]'
_currency_keys = map(re.escape, _currency_dict.keys())
_re_currency = f"({'|'.join(_currency_keys)})"
_re_magnitute = r'k|m|b|t|hundred|thousand|million|billion|trillion'
_measure_keys = map(re.escape, _measurements_dict.keys())
_re_measure = f"({'|'.join(_measure_keys)})"
_re_measure_decimal = r'(\d+(\,\d+)*(\.(\d+))?|\.(\d+))'
_re_roman = r'M{0,4}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{2,3})'
_re_time_minutes = r'[0-5][0-9]'
_re_time_suffix = r'(a.m.|am|pm|p.m.)'
_whitelist_keys = map(re.escape, _whitelist_dict.keys())
_re_whitelist = f"({'|'.join(_whitelist_keys)})"

re_whitelist = make_re(rf'{_re_whitelist}')
re_cardinal = make_re(rf'(?P<number>-?(\d+)(\,[0-9]+)*)')
re_ordinal = make_re(rf'(?P<number>[0-9]+)(st|nd|rd|th)')
re_roman = make_re(rf'(?P<number_roman>{_re_roman})')
re_decimal = make_re(rf'-?(\d+(\,\d+)*)\.(\d+)')
re_decimal2 = make_re(rf'-?\.\d+')

re_verbatim_and = make_re(rf'&')
# re_verbatim_silence = make_re(rf'[-]')

re_money_with_magnitude = make_re(
    rf'''
    (?P<currency>{_re_currency})
    (?P<integral>(\d+(\,\d+)*))
    (\.(?P<fractional>\d+))?
    \s?(?P<magnitude>{_re_magnitute})
''',
    re.VERBOSE,
)
re_money = make_re(
    rf'''
    (?P<currency>{_re_currency})
    (?P<integral>(\d+(\,\d+)*))
    (\.(?P<fractional>\d{{2}}))?
''',
    re.VERBOSE,
)

# e.g. 2010-01-31
re_date_ymd = make_re(
    rf'(?P<year>{_re_date_year})(?P<sep>[- /.])(?P<month>{_re_date_month})(?P=sep)(?P<day>{_re_date_day})'
)
# August 23, 2014 or Aug. 4 1999
re_date_mdy = make_re(
    rf'''(?P<month>{_re_date_month2})
    \s
    (?P<day>{_re_date_day})
    ,?\s
    (?P<year>{_re_date_year})
''',
    re.VERBOSE,
)
# Aug. 4
re_date_md = make_re(
    rf'''(?P<month>{_re_date_month2})
    \s
    (?P<day>{_re_date_day})
''',
    re.VERBOSE,
)
# Aug. 2015
re_date_my = make_re(
    rf'''(?P<month>{_re_date_month2})
    \s
    (?P<year>{_re_date_year})
''',
    re.VERBOSE,
)
# 1 December 2013
re_date_dmy = make_re(rf'(?P<day>{_re_date_day})\s(?P<month>{_re_date_month2})\s(?P<year>{_re_date_year})')
# 1 December
re_date_dm = make_re(rf'(?P<day>{_re_date_day})\s(?P<month>{_re_date_month2})')
# 1567
re_date_y = make_re(rf'(?P<year>[12]\d{{3}})')
# 1570s
re_date_ys = make_re(rf'(?P<year>[12]\d{{3}})(?P<suffix>\'?s)')

re_measure = make_re(
    rf'''
    (?P<decimal>{_re_measure_decimal})
    \s?
    (?P<measurement>{_re_measure})
''',
    re.VERBOSE,
)
re_measure2 = make_re(
    rf'''
    (?P<decimal>{_re_measure_decimal})
    \s?
    /
    (?P<measurement2>{_re_measure})
''',
    re.VERBOSE,
)
re_measure3 = make_re(
    rf'''
    (?P<decimal>{_re_measure_decimal})
    \s?
    (?P<measurement>{_re_measure})
    /
    (?P<measurement2>{_re_measure})
''',
    re.VERBOSE,
)

# e.g. '1:00' or '14:59 p.m.'
re_time = make_re(
    rf'''
        (?P<hour>{_re_time_hour})
        :(?P<minutes>{_re_time_minutes})
        \s?(?P<suffix>{_re_time_suffix})?
        ''',
    re.VERBOSE,
)
re_time2 = make_re(
    rf'''
        (?P<hour>{_re_time_hour})
        \s?(?P<suffix>{_re_time_suffix})
        ''',
    re.VERBOSE,
)
re_time3 = make_re(
    rf'''
        (?P<hour>{_re_time_hour})
        .(?P<minutes>{_re_time_minutes})
        \s?(?P<suffix>{_re_time_suffix})
        ''',
    re.VERBOSE,
)


def re_tag(text, kind: TagType, regex):
    """
    Detects and returns all tags in the text.
    Args:
        text: string
        kind: Tag type
        regex: compiled regex for detection
    Returns: generates all semiotic class tags that appear in the text
    """
    for match in re.finditer(regex, text, overlapped=True):
        yield Tag(
            kind=kind, start=match.start("value"), end=match.end("value"), data=match.groupdict(),
        )


def tag_whitelist(text: str):
    """
    Tags whitelisted tokens in text
    Args:
        text: input string
    Returns: Generates whitelisted tags from text
    """
    yield from re_tag(text, TagType.WHITELIST, re_whitelist)


def tag_cardinal(text: str):
    """
    Tags cardinals in text:
    E.g. - '11'
    Args:
        text: input string
    Returns: Generates all cardinal tags from text
    """
    yield from re_tag(text, TagType.CARDINAL, re_cardinal)
    yield from re_tag(text, TagType.CARDINAL, re_roman)


def tag_decimal(text: str):
    """
    Tags decimals in text:
    E.g. - '11.12'
    Args:
        text: input string
    Returns: Generates all decimal tags from text
    """
    yield from re_tag(text, TagType.DECIMAL, re_decimal)
    yield from re_tag(text, TagType.DECIMAL, re_decimal2)


def tag_date(text: str):
    """
    Tags dates in text:
    E.g. - 'Apr 08, 2020'
    Args:
        text: input string
    Returns: Generates all date tags from text
    """

    def helper(regex):
        yield from re_tag(text, TagType.DATE, regex)

    yield from helper(re_date_ymd)
    yield from helper(re_date_mdy)
    yield from helper(re_date_dmy)
    yield from helper(re_date_md)
    yield from helper(re_date_my)
    yield from helper(re_date_dm)
    yield from helper(re_date_ys)
    yield from helper(re_date_y)


def tag_verbatim(text: str):
    """
    Tags verbatims in text:
    E.g. - '&' -> 'and'
    Args:
        text: input string
    Returns: Generates all verbatim tags from text
    """
    yield from re_tag(text, TagType.VERBATIM, re_verbatim_and)


def tag_ordinal(text: str):
    """
    Tags ordinals in text:
    E.g. - '11th'
    Args:
        text: input string
    Returns: Generates all ordinal tags from text
    """
    yield from re_tag(text, TagType.ORDINAL, re_ordinal)


def tag_money(text: str):
    """
    Tags currencies in text:
    E.g. - '$11.50'
    Args:
        text: input string
    Returns: Generates all money tags from text
    """
    yield from re_tag(text.lower(), TagType.MONEY, re_money_with_magnitude)
    yield from re_tag(text.lower(), TagType.MONEY, re_money)


def tag_measure(text: str):
    """
    Tags measures in text:
    E.g. - '11kg'
    Args:
        text: input string
    Returns: Generates all measure tags from text
    """
    yield from re_tag(text.lower(), TagType.MEASURE, re_measure3)
    yield from re_tag(text.lower(), TagType.MEASURE, re_measure2)
    yield from re_tag(text.lower(), TagType.MEASURE, re_measure)


def tag_time(text: str):
    """
    Tags times in text:
    E.g. - '11.30am'
    Args:
        text: input string
    Returns: Generates all time tags from text
    """
    yield from re_tag(text.lower(), TagType.TIME, re_time)
    yield from re_tag(text.lower(), TagType.TIME, re_time3)
    yield from re_tag(text.lower(), TagType.TIME, re_time2)
