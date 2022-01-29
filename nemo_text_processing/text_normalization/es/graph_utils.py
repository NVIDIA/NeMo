import pynini

from nemo_text_processing.text_normalization.en.graph_utils import NEMO_SPACE

cardinal_seperator = pynini.string_map([".", NEMO_SPACE])
decimal_seperator = pynini.accep(",")

strip_accents = pynini.string_map([
	("á","a"),
	("é","e"),
	("í","i"),
	("ó","o"),
	("ú","u")
]) 

articles_for_dates = pynini.union("de", "del", "el", "del año")
