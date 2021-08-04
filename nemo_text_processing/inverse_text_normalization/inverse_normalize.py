# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from argparse import ArgumentParser
from typing import List

from nemo_text_processing.text_normalization.normalize import Normalizer
from nemo_text_processing.text_normalization.token_parser import TokenParser


class InverseNormalizer(Normalizer):
    """
    Inverse normalizer that converts text from spoken to written form. Useful for ASR postprocessing. 
    Input is expected to have no punctuation and be lower cased.

    Args:
        lang: language specifying the ITN, by default: English.
    """

    def __init__(self, lang: str = 'en'):
        if lang == 'en':
            from nemo_text_processing.inverse_text_normalization.en.taggers.tokenize_and_classify import ClassifyFst
            from nemo_text_processing.inverse_text_normalization.en.verbalizers.verbalize_final import (
                VerbalizeFinalFst,
            )

        elif lang == 'de':
            from nemo_text_processing.inverse_text_normalization.de.taggers.tokenize_and_classify import ClassifyFst
            from nemo_text_processing.inverse_text_normalization.de.verbalizers.verbalize_final import (
                VerbalizeFinalFst,
            )

        self.tagger = ClassifyFst()
        self.verbalizer = VerbalizeFinalFst()
        self.parser = TokenParser()

    def inverse_normalize_list(self, texts: List[str], verbose=False) -> List[str]:
        """
        NeMo inverse text normalizer 

        Args:
            texts: list of input strings
            verbose: whether to print intermediate meta information

        Returns converted list of input strings
        """
        return self.normalize_list(texts=texts, verbose=verbose)

    def inverse_normalize(self, text: str, verbose: bool) -> str:
        """
        Main function. Inverse normalizes tokens from spoken to written form
            e.g. twelve kilograms -> 12 kg

        Args:
            text: string that may include semiotic classes
            verbose: whether to print intermediate meta information

        Returns: written form
        """
        return self.normalize(text=text, verbose=verbose)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("input_string", help="input string", type=str)
    parser.add_argument("--language", help="language", choices=['en', 'de'], default="en", type=str)
    parser.add_argument("--verbose", help="print info for debugging", action='store_true')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    inverse_normalizer = InverseNormalizer(lang=args.language)
    print(inverse_normalizer.inverse_normalize(args.input_string, verbose=args.verbose))
