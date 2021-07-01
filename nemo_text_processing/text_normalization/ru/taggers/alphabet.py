try:
    import pynini
    from pynini.lib import pynutil

    PYNINI_AVAILABLE = True
except (ModuleNotFoundError, ImportError):
    PYNINI_AVAILABLE = False

RU_LOWER_ALPHA = [
    "а",
    "б",
    "в",
    "г",
    "д",
    "е",
    "ё",
    "ж",
    "з",
    "и",
    "й",
    "к",
    "л",
    "м",
    "н",
    "о",
    "п",
    "р",
    "с",
    "т",
    "у",
    "ф",
    "х",
    "ц",
    "ч",
    "ш",
    "щ",
    "ъ",
    "ы",
    "ь",
    "э",
    "ю",
    "я",
]

RU_UPPER_ALPHA = [alpha.upper() for alpha in RU_LOWER_ALPHA]


RU_STRESSED_MAP = (
    ("А́", "А'")
    | ("Е́", "Е'")
    | ("Ё́", "Е'")
    | ("И́", "И'")
    | ("О́", "О'")
    | ("У́", "У'")
    | ("Ы́", "Ы'")
    | ("Э́", "Э'")
    | ("Ю́", "Ю'")
    | ("Я́", "Я'")
    | ("а́", "а'")
    | ("е́", "е'")
    | ("ё́", "е'")
    | ("и́", "и'")
    | ("о́", "о'")
    | ("у́", "у'")
    | ("ы́", "ы'")
    | ("э́", "э'")
    | ("ю́", "ю'")
    | ("я́", "я'")
    | ("ё", "е")
    | ("Ё", "Е")
).optimize()

# # Pre-reform characters, just in case.
# export kRussianPreReform = Optimize[
#     "ѣ" | "Ѣ"   # http://en.wikipedia.org/wiki/Yat
# ];
#
# export kCyrillicAlphaStressed = Optimize[
#   kRussianLowerAlphaStressed | kRussianUpperAlphaStressed
# ];
#
# export kCyrillicAlpha = Optimize[
#     kRussianLowerAlpha | kRussianUpperAlpha | kRussianPreReform
# ];
