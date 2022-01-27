# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
# Copyright 2015 and onwards Google, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#	 http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from nemo_text_processing.text_normalization.en.graph_utils import NEMO_CHAR, NEMO_NOT_QUOTE, NEMO_NOT_SPACE, delete_space, NEMO_SIGMA, NEMO_SPACE, GraphFst, insert_space
from nemo_text_processing.text_normalization.en.verbalizers.ordinal import OrdinalFst
from nemo_text_processing.text_normalization.es.utils import get_abs_path
from nemo_text_processing.text_normalization.es.graph_utils import strip_accents
try:
	import pynini
	from pynini.lib import pynutil

	PYNINI_AVAILABLE = True
except (ModuleNotFoundError, ImportError):
	PYNINI_AVAILABLE = False


class FractionFst(GraphFst):
	"""
	Finite state transducer for verbalizing fraction
		e.g. tokens { fraction { integer: "twenty three" numerator: "four" denominator: "five" } } ->
		twenty three four fifth

	Args:
		deterministic: if True will provide a single transduction option,
			for False multiple transduction are generated (used for audio-based normalization)
	"""

	def __init__(self, deterministic: bool = True):
		super().__init__(name="fraction", kind="verbalize", deterministic=deterministic)

		# Cardinal strings append 'avo' as a suffix.
		fraction_stem = pynutil.insert(" avo") # This is the stem for production. Adding a space to manage preprocessing.
		denominator_add_stem =pynutil.delete("denominator: \"") + (
			pynini.closure(NEMO_NOT_QUOTE) + fraction_stem 
			+ pynutil.delete("\" morphosyntactic_feautures: \"cardinal\"")
		)
		   
		# Ordinals we take as is 
		denominator_no_change = pynutil.delete("denominator: \"") + (
			pynini.closure(NEMO_NOT_QUOTE) 
			+ pynutil.delete("\" morphosyntactic_feautures: \"ordinal\"")
		)

		denominator_singular = pynini.union(denominator_add_stem, denominator_no_change)

		plural = pynutil.insert("s")

		denominator_plural = denominator_singular + plural

		if not deterministic:
		# Occasional exceptions
		# Eleven and twelve
			denominator_singular |= (pynutil.delete("denominator: \"") +
			pynini.string_map([
				 ("once", "undécimo"),
				 ("doce", "duodécimo")
			 ])
			+ pynutil.delete("\" morphosyntactic_feautures: \"cardinal\""))

		# The denominator must be a single word, with the conjunction "y" replaced by i
		merge = delete_space | pynini.cross(" y ", "i")
		merge = pynini.cdrewrite(merge, "", pynini.difference(NEMO_CHAR, "p"), NEMO_SIGMA) # The "p" will only show up for "parte"
		# The merger can produce duplicate vowels. This is not allowed in orthography
		delete_duplicates = pynini.string_map([ # Removes vowels
			("aa", "a"),
			("oo", "o")
		])
		delete_duplicates = pynini.cdrewrite(delete_duplicates, "", "", NEMO_SIGMA)
		remove_accents = pynini.cdrewrite( strip_accents, pynini.union(NEMO_SPACE, pynini.accep("[BOS]")) + pynini.closure(NEMO_NOT_SPACE), pynini.closure(NEMO_NOT_SPACE) + pynini.union("avo", "ava", "ésimo", "ésima") + pynini.accep("s").ques, NEMO_SIGMA)
		merge_into_single_word = merge @ remove_accents @ delete_duplicates
		
		# Cardinals assume apocope with final ones, need to remove this
		final_one = pynini.union("un", "ún")
		remove_apocope = pynini.cdrewrite(pynini.cross(final_one, "uno"), "", pynini.accep("[EOS]"), NEMO_SIGMA)  

		integer = pynutil.delete("integer_part: \"") + pynini.closure(NEMO_NOT_QUOTE) + pynutil.delete("\" ")
		integer @= remove_apocope

		numerator_one = pynutil.delete("numerator: \"") + pynini.accep("un") + pynutil.delete("\" ")

		numerator = pynutil.delete("numerator: \"") + pynini.closure(NEMO_NOT_QUOTE) + pynutil.delete("\" ")
		numerator @= pynini.cdrewrite(pynutil.delete("un"), pynini.accep("[BOS]"), pynini.accep("[EOS]"), NEMO_SIGMA) @ NEMO_CHAR.plus # To block "un"

		fraction_default = (
				numerator + insert_space + (denominator_plural @ merge_into_single_word)
			)

		fraction_with_one = (
				numerator_one + insert_space + (denominator_singular @ merge_into_single_word)
				)

		conjunction = pynutil.insert(" y ")

		if not deterministic:
		   	# There is an alternative rendering where ordinals act as adjectives for 'parte'. This requires use of the feminine
			# Other rules will manage use of "un" at end, so just worry about endings
			change_gender_ending = pynini.cdrewrite(pynini.cross("o", "a"), "", pynini.accep("[EOS]"), NEMO_SIGMA)
			change_gender_hundred = pynini.cdrewrite(pynini.cross("ientos", "ientas"), "", "", NEMO_SIGMA)
			gender_allignment = remove_apocope @ change_gender_hundred @ change_gender_ending

			denominator_singular_fem = denominator_singular @ gender_allignment
			denominator_plural_fem = denominator_singular @ gender_allignment + plural

			numerator_one_fem = numerator_one @  gender_allignment
			
			numerator_fem = numerator @ gender_allignment

			# Still need to manage stems
			merge_stem = pynini.cdrewrite(delete_space, "", pynini.union("avo", "ava", "avos", "avas"), NEMO_SIGMA) # For managing alternative spacing
			merge_stem @= remove_accents @ delete_duplicates

			fraction_with_one_fem = numerator_one_fem + insert_space 
			fraction_with_one_fem += pynini.union(denominator_singular_fem @ merge_stem, denominator_singular_fem @ merge_into_single_word) # Both forms exists 
			fraction_with_one_fem @= pynini.cdrewrite(pynini.cross("una media", "media"), "", "", NEMO_SIGMA)
			fraction_with_one_fem += pynutil.insert(" parte")

			fraction_default_fem = numerator_fem + insert_space 
			fraction_default_fem += pynini.union(denominator_plural_fem @ merge_stem, denominator_plural_fem @ merge_into_single_word)
			fraction_default_fem += pynutil.insert(" partes")

			fraction_default |= numerator + insert_space + denominator_plural @ merge_stem # Case of no merger
			fraction_default |= fraction_default_fem

			fraction_with_one |= numerator_one + insert_space + denominator_singular @ merge_stem
			fraction_with_one |= fraction_with_one_fem

            # Integers are influenced by dominant noun, need to allow feminine forms as well
			integer |= integer @ gender_allignment

		# Remove 'un medio'
		fraction_with_one @= pynini.cdrewrite(pynini.cross("un medio", "medio"), "", "", NEMO_SIGMA)

		integer = pynini.closure(integer + conjunction, 0, 1)

		fraction = fraction_with_one | fraction_default

		graph = integer + fraction

		self.graph = graph
		delete_tokens = self.delete_tokens(self.graph)
		self.fst = delete_tokens.optimize()
