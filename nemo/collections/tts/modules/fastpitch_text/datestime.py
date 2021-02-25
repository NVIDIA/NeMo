import re

_ampm_re = re.compile(r'([0-9]|0[0-9]|1[0-9]|2[0-3]):?([0-5][0-9])?\s*([AaPp][Mm]\b)')


def _expand_ampm(m):
    matches = list(m.groups(0))
    txt = matches[0]
    txt = txt if int(matches[1]) == 0 else txt + ' ' + matches[1]

    if matches[2][0].lower() == 'a':
        txt += ' a.m.'
    elif matches[2][0].lower() == 'p':
        txt += ' p.m.'

    return txt


def normalize_datestime(text):
    text = re.sub(_ampm_re, _expand_ampm, text)
    # text = re.sub(r"([0-9]|0[0-9]|1[0-9]|2[0-3]):([0-5][0-9])?", r"\1 \2", text)
    return text
