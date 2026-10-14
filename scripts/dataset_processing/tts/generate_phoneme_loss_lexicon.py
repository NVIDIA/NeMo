# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generate a reproducible English normalization-vocabulary IPA lexicon.

Run in the project's eSpeak-NG 1.52 container. No phonemizer is needed during
training: the generated YAML contains all pronunciations.
"""

import argparse
import collections
import concurrent.futures
import json
import pathlib
import subprocess

import yaml


def vocabulary():
    entries = collections.defaultdict(set)

    def add(categories, words):
        for word in words.split('|'):
            if word.strip():
                entries[word.strip()].update(categories.split())

    numbers = (
        'zero|oh|nought|naught|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|'
        'thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|'
        'sixty|seventy|eighty|ninety|hundred|thousand|million|billion|trillion|quadrillion|'
        'quintillion|sextillion|septillion|octillion|nonillion|decillion|undecillion|duodecillion|'
        'tredecillion|quattuordecillion|quindecillion|sexdecillion|septendecillion|octodecillion|'
        'novemdecillion|vigintillion|centillion|googol|googolplex'
    )
    numeric_classes = 'cardinal long_digit_runs comma_grouped_numbers decimal currency percent phone years dates time'
    add(numeric_classes, numbers)
    add('cardinal long_digit_runs', 'hundreds|thousands|millions|billions|trillions|dozen|dozens|score|scores')
    ordinals = (
        'zeroth|first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|eleventh|twelfth|'
        'thirteenth|fourteenth|fifteenth|sixteenth|seventeenth|eighteenth|nineteenth|twentieth|'
        'thirtieth|fortieth|fiftieth|sixtieth|seventieth|eightieth|ninetieth|hundredth|thousandth|'
        'millionth|billionth|trillionth|quadrillionth|quintillionth|sextillionth|septillionth|'
        'octillionth|nonillionth|decillionth'
    )
    add('ordinal dates fractions', ordinals)
    add('fractions', '|'.join(w + 's' for w in ordinals.split('|') if w not in {'zeroth', 'first', 'second'}))
    add('fractions', 'half|halves|quarter|quarters|whole|wholes|over|divided by|one half|three quarters')
    add('decimal signed percent currency', 'minus|negative|plus|positive|point|decimal|decimal point')
    add('percent', 'percent|per cent|percentage|percentages|per mille|permille|basis point|basis points')
    add(
        'time colon_numbers_time_or_ratio',
        'hour|hours|minute|minutes|second|seconds|millisecond|milliseconds|'
        'microsecond|microseconds|nanosecond|nanoseconds|o\'clock|noon|midnight|AM|PM|ante meridiem|'
        'post meridiem|morning|afternoon|evening|night|UTC|GMT|EST|EDT|CST|CDT|MST|MDT|PST|PDT|'
        'CET|CEST|BST|IST|JST|AEST|AEDT|coordinated universal time|Greenwich mean time|'
        'eastern standard time|eastern daylight time|central standard time|central daylight time|'
        'mountain standard time|mountain daylight time|Pacific standard time|Pacific daylight time|'
        'standard time|daylight time|daylight saving time|time zone|time zones',
    )
    add(
        'dates years',
        'January|February|March|April|May|June|July|August|September|October|November|December|'
        'Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday|day|days|week|weeks|month|months|year|years|'
        'century|centuries|millennium|millennia|leap year|AD|BC|BCE|CE|Anno Domini|before Christ|'
        'before common era|common era',
    )
    add('phone', 'extension|extensions|country code|area code|double|triple')
    currency_units = (
        'dollar|cent|pound|penny|pence|euro|yen|yuan|renminbi|rupee|paise|paisa|franc|centime|'
        'won|ruble|rouble|kopeck|real|reais|peso|centavo|crown|krona|kronor|krone|kroner|'
        'zloty|grosz|forint|filler|koruna|haler|lira|kurus|rand|dirham|fils|riyal|halala|'
        'dinar|piastre|rial|baht|satang|dong|ringgit|sen|rupiah|rufiyaa|loti|kwacha|naira|'
        'cedi|pesewa|shilling|hryvnia|bitcoin|satoshi|ether'
    )
    add('currency', currency_units)
    irregular_currency = {
        'yen',
        'yuan',
        'renminbi',
        'won',
        'reais',
        'real',
        'pence',
        'penny',
        'paise',
        'paisa',
        'krona',
        'kronor',
        'krone',
        'kroner',
        'fils',
        'grosz',
        'haler',
        'filler',
        'kurus',
        'halala',
        'satang',
        'dong',
        'sen',
        'rupiah',
        'loti',
    }
    add('currency', '|'.join(w + 's' for w in currency_units.split('|') if w not in irregular_currency))
    add('currency', 'pennies|paisas|groszy|halalas|maloti')
    add(
        'currency',
        'US|USD|GBP|EUR|JPY|CNY|INR|CHF|CAD|AUD|NZD|HKD|SGD|KRW|BRL|MXN|RUB|ZAR|'
        'SEK|NOK|DKK|PLN|TRY|AED|SAR|ILS|TWD|THB|VND|American|Canadian|Australian|British|'
        'Japanese|Chinese|Indian|Swiss|Brazilian|Mexican|United States|sterling',
    )
    unit_words = (
        'meter|metre|kilometer|kilometre|centimeter|centimetre|millimeter|millimetre|micrometer|micrometre|'
        'nanometer|nanometre|mile|yard|foot|inch|gram|kilogram|milligram|microgram|ton|tonne|ounce|'
        'liter|litre|milliliter|millilitre|gallon|quart|pint|cup|tablespoon|teaspoon|hertz|kilohertz|'
        'megahertz|gigahertz|terahertz|byte|kilobyte|megabyte|gigabyte|terabyte|petabyte|kibibyte|'
        'mebibyte|gibibyte|tebibyte|bit|kilobit|megabit|gigabit|terabit|watt|kilowatt|megawatt|'
        'gigawatt|volt|kilovolt|millivolt|ampere|amp|milliampere|ohm|kilohm|joule|kilojoule|'
        'calorie|kilocalorie|pascal|kilopascal|megapascal|bar|millibar|atmosphere|degree|radian|'
        'kelvin|mole|millimole|micromole|newton|lumen|lux|candela|decibel|becquerel|sievert|'
        'gray|tesla|gauss|hectare|acre|knot|nautical mile|revolution|rotation'
    )
    add('unit_abbreviations', unit_words)
    add(
        'unit_abbreviations',
        '|'.join(
            w + 's'
            for w in unit_words.split('|')
            if w not in {'lux', 'gauss', 'foot', 'inch'} and not w.endswith('hertz')
        ),
    )
    add(
        'unit_abbreviations',
        'feet|inches|Celsius|Fahrenheit|centigrade|per|square|squared|cubic|cubed|'
        'miles per hour|kilometers per hour|kilometres per hour|meters per second|metres per second|'
        'kilowatt hour|kilowatt hours|parts per million|parts per billion|revolutions per minute|'
        'beats per minute|pounds per square inch|millimeters of mercury|millimetres of mercury',
    )
    add(
        'symbols urls email phone alphanumeric',
        'dot|dots|at|at sign|ampersand|and sign|hash|hashes|hashtag|'
        'number sign|pound sign|pound key|star|asterisk|asterisks|plus sign|minus sign|dash|dashes|hyphen|'
        'hyphens|underscore|underscores|slash|slashes|forward slash|backslash|backslashes|colon|colons|'
        'semicolon|semicolons|comma|commas|period|periods|full stop|question mark|exclamation mark|'
        'exclamation point|equals|equal sign|tilde|caret|pipe|vertical bar|bar|macron|overbar|'
        'apostrophe|apostrophes|quote|quotes|quotation mark|double quote|single quote|'
        'left parenthesis|right parenthesis|open parenthesis|close parenthesis|parentheses|'
        'left bracket|right bracket|open bracket|close bracket|left brace|right brace|'
        'open brace|close brace|curly brace|curly braces|angle bracket|angle brackets|'
        'less than|greater than|percent sign|degree sign|space|spaces',
    )
    add(
        'urls email',
        'www|world wide web|http|https|ftp|ftps|sftp|ssh|mailto|email|e-mail|URL|URI|'
        'com|org|net|edu|gov|mil|int|io|co|uk|us|de|fr|es|it|pt|br|ca|au|nz|in|jp|cn|kr|'
        'ai|app|dev|info|biz|me|tv|tech|online|site|store|cloud|xyz|html|htm|php|asp|aspx|'
        'pdf|json|xml|txt|csv|zip|png|jpg|jpeg|gif|webp|mp3|wav|mp4|webm',
    )
    add(
        'alphanumeric roman_numerals',
        'alpha|beta|gamma|delta|epsilon|zeta|eta|theta|iota|kappa|lambda|mu|'
        'nu|xi|omicron|pi|rho|sigma|tau|upsilon|phi|chi|psi|omega',
    )
    # Canonical A-Z entries carry the letter-name IPA. Respellings such as
    # "bee" and "ee" are redundant and do not belong in the written vocabulary.
    add('letters alphanumeric roman_numerals urls email phone', '|'.join(chr(i) for i in range(65, 91)))
    return entries


def phonemize(task):
    word, voice = task
    # A in isolation is normally read as an article; explicitly request letter names.
    letter = len(word) == 1 and word.isupper() and word.isascii()
    query = 'letter ' + word if letter else word
    if word in {'AM', 'PM', 'AD', 'BC', 'BCE', 'CE'}:
        # Speak every letter explicitly: eSpeak can otherwise read A as an article.
        pronunciation = ' '.join(phonemize((character, voice))[2] for character in word)
        return word, voice, pronunciation
    text = subprocess.run(
        ['espeak-ng', '-q', '--ipa', '-v', voice, '--stdin'],
        input=query,
        text=True,
        encoding='utf-8',
        capture_output=True,
        check=True,
    ).stdout
    pieces = text.split()
    pronunciation = ' '.join(pieces[1:] if letter else pieces).strip('.,;:!?')
    if not pronunciation:
        raise ValueError(f'Empty IPA for {word!r}, {voice}')
    return word, voice, pronunciation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=pathlib.Path, required=True)
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()
    version = subprocess.run(['espeak-ng', '--version'], capture_output=True, text=True, check=True).stdout.strip()
    if '1.52' not in version:
        raise RuntimeError(f'Expected eSpeak-NG 1.52, found {version}')
    words = vocabulary()
    pronunciations = collections.defaultdict(dict)
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        for word, voice, ipa in pool.map(phonemize, [(w, 'en-us') for w in sorted(words)]):
            pronunciations[word]['en-US'] = [ipa]
    entries = [dict(word=w, categories=sorted(words[w]), pronunciations=pronunciations[w]) for w in sorted(words)]
    payload = dict(
        schema_version=1,
        languages=['en'],
        metadata=dict(
            phonemizer=version,
            voices={'en-US': 'en-us'},
            scope='English normalization vocabulary; compositional components, not every literal entity.',
            vocabulary_policy=(
                'Canonical words, phrases, abbreviations, and A-Z letters; '
                'no letter-name respellings or generic context labels.'
            ),
            limitations=[
                'URLs, identifiers, names, loanwords, and domain labels form an open vocabulary.',
                'IPA homophones cannot distinguish two/to/too, four/for, eight/ate, and one/won.',
                'Matches require IPA word boundaries; pronunciations fused across words may not match.',
                'Pronunciations are EN-US only; existing English language IDs are supported.',
            ],
        ),
        entries=entries,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(yaml.safe_dump(payload, allow_unicode=True, sort_keys=False, width=110), encoding='utf-8')
    print(
        json.dumps(
            dict(
                entries=len(entries),
                categories=sorted({c for v in words.values() for c in v}),
                output=str(args.output),
            ),
            indent=2,
        )
    )


if __name__ == '__main__':
    main()
