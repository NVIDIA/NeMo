import re
from . import cmudict

_letter_to_arpabet = {
    'A': 'EY1',
    'B': 'B IY1',
    'C': 'S IY1',
    'D': 'D IY1',
    'E': 'IY1',
    'F': 'EH1 F',
    'G': 'JH IY1',
    'H': 'EY1 CH',
    'I': 'AY1',
    'J': 'JH EY1',
    'K': 'K EY1',
    'L': 'EH1 L',
    'M': 'EH1 M',
    'N': 'EH1 N',
    'O': 'OW1',
    'P': 'P IY1',
    'Q': 'K Y UW1',
    'R': 'AA1 R',
    'S': 'EH1 S',
    'T': 'T IY1',
    'U': 'Y UW1',
    'V': 'V IY1',
    'X': 'EH1 K S',
    'Y': 'W AY1',
    'W': 'D AH1 B AH0 L Y UW0',
    'Z': 'Z IY1',
    's': 'Z',
}

# must ignore roman numerals
# _acronym_re = re.compile(r'([A-Z][A-Z]+)s?|([A-Z]\.([A-Z]\.)+s?)')
_acronym_re = re.compile(r'([A-Z][A-Z]+)s?')


def _expand_acronyms(m, add_spaces=True):
    acronym = m.group(0)

    # remove dots if they exist
    acronym = re.sub('\.', '', acronym)

    acronym = "".join(acronym.split())
    arpabet = cmudict.lookup(acronym)

    if arpabet is None:
        acronym = list(acronym)
        arpabet = ["{" + _letter_to_arpabet[letter] + "}" for letter in acronym]
        # temporary fix
        if arpabet[-1] == '{Z}' and len(arpabet) > 1:
            arpabet[-2] = arpabet[-2][:-1] + ' ' + arpabet[-1][1:]
            del arpabet[-1]

        arpabet = ' '.join(arpabet)
    elif len(arpabet) == 1:
        arpabet = "{" + arpabet[0] + "}"
    else:
        arpabet = acronym

    return arpabet


def normalize_acronyms(text):
    text = re.sub(_acronym_re, _expand_acronyms, text)
    return text
