try:
    import pynini
    from pynini.lib import pynutil

    PYNINI_AVAILABLE = True
except (ModuleNotFoundError, ImportError):
    PYNINI_AVAILABLE = False

try:
    RU_LOWER_ALPHA = "абвгдеёжзийклмнопрстуфхцчшщъыьэюя"
    RU_UPPER_ALPHA = RU_LOWER_ALPHA.upper()
    RU_LOWER_ALPHA = pynini.union(*RU_LOWER_ALPHA).optimize()
    RU_UPPER_ALPHA = pynini.union(*RU_UPPER_ALPHA).optimize()
    RU_ALPHA = (RU_LOWER_ALPHA | RU_UPPER_ALPHA).optimize()

    RU_STRESSED_MAP = [
        ("А́", "А'"),
        ("Е́", "Е'"),
        ("Ё́", "Е'"),
        ("И́", "И'"),
        ("О́", "О'"),
        ("У́", "У'"),
        ("Ы́", "Ы'"),
        ("Э́", "Э'"),
        ("Ю́", "Ю'"),
        ("Я́", "Я'"),
        ("а́", "а'"),
        ("е́", "е'"),
        ("ё́", "е'"),
        ("и́", "и'"),
        ("о́", "о'"),
        ("у́", "у'"),
        ("ы́", "ы'"),
        ("э́", "э'"),
        ("ю́", "ю'"),
        ("я́", "я'"),
        ("ё", "е"),
        ("Ё", "Е"),
    ]

    REWRITE_STRESSED = pynini.closure(pynini.string_map(RU_STRESSED_MAP).optimize() | RU_ALPHA).optimize()

except (ModuleNotFoundError, ImportError):
    # Create placeholders
    RU_ALPHA = None
