import pynini

from nemo_text_processing.text_normalization.en.graph_utils import NEMO_SIGMA, NEMO_SPACE, NEMO_NOT_QUOTE

cardinal_seperator = pynini.string_map([".", NEMO_SPACE])
decimal_seperator = pynini.accep(",")

strip_accents = pynini.string_map([
	("á","a"),
	("é","e"),
	("í","i"),
	("ó","o"),
	("ú","u")
]) 
