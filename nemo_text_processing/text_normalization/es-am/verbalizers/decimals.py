# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

#from nemo_text_processing.text_normalization.es.taggers.decimals import quantities
from nemo_text_processing.text_normalization.en.graph_utils import (
	NEMO_NOT_QUOTE,
	GraphFst,
	delete_preserve_order,
	insert_space,
	NEMO_SIGMA
)
from pynini.lib import rewrite
try:
	import pynini
	from pynini.lib import pynutil

	PYNINI_AVAILABLE = True
except (ModuleNotFoundError, ImportError):
	PYNINI_AVAILABLE = False


class DecimalFst(GraphFst):
	"""
	Finite state transducer for classifying decimal, e.g. 
		decimal { negative: "true" integer_part: "dos"  fractional_part: "quatro cero" quantity: "billonen" } -> menos dos coma quatro cero billones 
		decimal { integer_part: "eins" quantity: "billion" } -> eins billion

	"""

	def __init__(self, deterministic: bool = True):
		super().__init__(name="decimal", kind="classify", deterministic=deterministic)

		delete_space = pynutil.delete(" ")
		self.optional_sign = pynini.closure(pynini.cross("negative: \"true\"", "menos ") + delete_space, 0, 1)
		self.integer = pynutil.delete("integer_part: \"") + pynini.closure(NEMO_NOT_QUOTE, 1) + pynutil.delete("\"")
		self.fractional_default = (
			pynutil.delete("fractional_part: \"") + pynini.closure(NEMO_NOT_QUOTE, 1) + pynutil.delete("\"")
		)

		conjunction = pynutil.insert(" coma ")
		if not deterministic:
			conjunction |= pynutil.insert(" con ")
			conjunction |= pynutil.insert(" y ")
		self.fractional = conjunction + self.fractional_default

		self.quantity = (
			delete_space + insert_space + pynutil.delete("quantity: \"") + NEMO_SIGMA + pynutil.delete("\"")
		)
		self.optional_quantity = pynini.closure(self.quantity, 0, 1)

		graph = self.optional_sign + (
			self.integer + self.quantity | self.integer + delete_space + self.fractional + self.optional_quantity
		)

		self.numbers = graph
		graph += delete_preserve_order

		if not deterministic:
			no_adjust = graph
			fem_adjust = graph + pynutil.delete(" morphosyntactic_features: \"gender_fem\"")
			
			fem_hundreds = pynini.cross("ientos", "ientas")
			fem_ones_final = pynini.cross("un", "una") | pynini.cross("ún", "una")
			fem_ones_rest = pynini.cross("uno", "una")
			fem_allign = pynini.cdrewrite(fem_hundreds, "", "", NEMO_SIGMA)
			fem_allign @= pynini.cdrewrite(fem_ones_final, "", "[EOS]", NEMO_SIGMA)
			fem_allign @= pynini.cdrewrite(fem_ones_rest, "", "", NEMO_SIGMA)

			fem_adjust @= fem_allign

			apocope_adjust =  graph + pynutil.delete(" morphosyntactic_features: \"no_apocope\"")
			strip_apocope = pynini.cross("un", "uno") | pynini.cross("ún", "uno")
			strip_apocope = pynini.cdrewrite(strip_apocope, "", "[EOS]", NEMO_SIGMA)

			apocope_adjust @= strip_apocope

			graph = no_adjust | fem_adjust | apocope_adjust
		delete_tokens = self.delete_tokens(graph)
		self.fst = delete_tokens.optimize()
