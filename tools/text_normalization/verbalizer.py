import regex as re
import inflect
import csv
_inflect = inflect.engine()

month_tsv = open("data/months.tsv")
read_tsv = csv.reader(month_tsv, delimiter="\t")
month_mapping = dict(read_tsv)  

_date_components_whitelist = {"month", "day", "year", "suffix"}
_roman_numerals = {'I':1, 'V':5, 'X':10, 'L':50, 'C':100, 'D':500, 'M':1000}
_magnitudes = ['trillion', 'billion', 'million', 'thousand', 'hundred', 'k', 'm', 'b', 't']

_magnitudes_tsv = open("data/magnitudes.tsv")
read_tsv = csv.reader(_magnitudes_tsv, delimiter="\t")
_magnitudes_abbrev = dict(read_tsv)
_currency_tsv = open("data/currency.tsv")
read_tsv = csv.reader(_currency_tsv, delimiter="\t")
_currency_abbrev = dict(read_tsv)

_measurements_tsv = open("data/measurements.tsv")
read_tsv = csv.reader(_measurements_tsv, delimiter="\t")
_measurements_abbrev = dict(read_tsv)  

_whitelist_tsv = open("data/whitelist.tsv")
read_tsv = csv.reader(_whitelist_tsv, delimiter="\t")
_whitelist_dict = dict(read_tsv) 

def expand_whitelist(data):
    return _whitelist_dict[data["value"]]

def expand_roman(data):
    num = data["value"]
    result = 0
    for i, c in enumerate(num):
        if (i+1) == len(num) or _roman_numerals[c] >= _roman_numerals[num[i+1]]:
            result += _roman_numerals[c]
        else:
            result -= _roman_numerals[c]
    return _inflect.number_to_words(result).replace("-", " ").replace(" and ", " ").replace(",", "")


def expand_cardinal(data):
    return  _inflect.number_to_words(data["value"]).replace("-", " ").replace(" and ", " ").replace(",", "")


def expand_ordinal(data):
    if data["value"] is None:
        return None
    result = _inflect.number_to_words(data["value"] + "th")
    return result.replace("-", " ").replace(" and ", " ").replace(",", "")


def expand_year(data):
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


def expand_date(data, verbalize):
    try:
        data["month"] = month_mapping[data["month"]]
    except:
        pass
    try:
        data["day"] = expand_ordinal({"value": data["day"]})
    except:
        pass
    try:
        data["year"] = expand_year({"value": data["year"]})
    except:
        pass
    data = {k: data[k] for k in data if k in _date_components_whitelist}
    result = verbalize(**data)
    return result.replace("-", " ")


def _expand_hundreds(text):
    number = float(text)
    if number > 1000 < 10000 and (number % 100 == 0) and (number % 1000 != 0):
        return _inflect.number_to_words(int(number / 100)) + " hundred"
    else:
        return _inflect.number_to_words(text)


def _expand_currency(data):
    currency = _currency_abbrev[data['currency']]
    integral = data['integral']
    quantity = data['integral'] + ('.' + data['fractional'] if data.get('fractional') else '')
    magnitude = data.get('magnitude')

    # remove commas from quantity to be able to convert to numerical
    quantity = quantity.replace(',', '')

    # check for million, billion, etc...
    if magnitude is not None and magnitude.lower() in _magnitudes:
        if len(magnitude) == 1:
            magnitude = _magnitudes_abbrev[magnitude.lower()]
        return "{} {} {}".format(_expand_hundreds(quantity), magnitude, currency+'s')

    parts = quantity.split('.')
    if len(parts) > 2:
        return quantity + " " + currency + "s"    # Unexpected format

    dollars = int(parts[0]) if parts[0] else 0

    cents = int(parts[1]) if len(parts) > 1 and parts[1] else 0
    if dollars and cents:
        dollar_unit = currency if dollars == 1 else currency+'s'
        cent_unit = 'cent' if cents == 1 else 'cents'
        return "{} {}, {} {}".format(
            _expand_hundreds(dollars), dollar_unit,
            _inflect.number_to_words(cents), cent_unit)
    elif dollars:
        dollar_unit = currency if dollars == 1 else currency+'s'
        return "{} {}".format(_expand_hundreds(dollars), dollar_unit)
    elif cents:
        cent_unit = 'cent' if cents == 1 else 'cents'
        return "{} {}".format(_inflect.number_to_words(cents), cent_unit)
    else:
        return 'zero' + ' ' + currency + 's'


def expand_money(data):
    result = _expand_currency(data)
    return result.replace(',', '').replace('-', ' ').replace(' and ', ' ')


def expand_measurement(data):
    value = float(data["decimal"].replace(",", ""))
    value_verb =  _inflect.number_to_words(data["decimal"]).replace(',', '').replace('-', ' ').replace(' and ', ' ')
    res = value_verb
    if data.get("measurement"):
        measure =  _measurements_abbrev[data["measurement"]]
        if value <= 1 and measure[-1] == 's':
            measure = measure[:-1]
        res += " " + measure
    
    if data.get("measurement2"):
        res += " per "
        measure2 = _measurements_abbrev[data["measurement2"]]
        # if measure2[-1] == 's':
        #     measure2 = measure2[:-1]
        res += measure2
    return res


def expand_time(data):
    hrs = int(data["hour"])
    res = _inflect.number_to_words(data["hour"]) 
    if data.get("minutes") and int(data["minutes"]) != 0:
        if data["minutes"][0] == "0":
            res += " o " + _inflect.number_to_words(data["minutes"])      
        else:
            res += " " + _inflect.number_to_words(data["minutes"])         
    else: 
        if not data.get("suffix") :
            res += " o'clock"
    
    if data.get("suffix"):
        res += " " + " ".join(list(data["suffix"].replace(".", "")))
    return res.replace("-", " ")