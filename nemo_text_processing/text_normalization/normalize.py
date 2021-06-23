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

import itertools
from argparse import ArgumentParser
from collections import OrderedDict
from typing import List

from nemo_text_processing.text_normalization.data_loader_utils import post_process_punctuation, pre_process
from nemo_text_processing.text_normalization.taggers.tokenize_and_classify import ClassifyFst
from nemo_text_processing.text_normalization.token_parser import PRESERVE_ORDER_KEY, TokenParser
from nemo_text_processing.text_normalization.verbalizers.verbalize_final import VerbalizeFinalFst
from tqdm import tqdm

try:
    import pynini

    PYNINI_AVAILABLE = True
    from pynini.lib import rewrite

except (ModuleNotFoundError, ImportError):
    PYNINI_AVAILABLE = False


class Normalizer:
    """
    Normalizer class that converts text from written to spoken form. 
    Useful for TTS preprocessing. 

    Args:
        input_case: expected input capitalization
    """

    def __init__(self, input_case: str):
        assert input_case in ["lower_cased", "cased"]

        self.tagger = ClassifyFst(input_case=input_case, deterministic=True)
        self.verbalizer = VerbalizeFinalFst(deterministic=True)
        self.parser = TokenParser()

    def normalize_list(self, texts: List[str], verbose=False) -> List[str]:
        """
        NeMo text normalizer 

        Args:
            texts: list of input strings
            verbose: whether to print intermediate meta information

        Returns converted list input strings
        """
        res = []
        for input in tqdm(texts):
            try:
                text = self.normalize(input, verbose=verbose)
            except:
                print(input)
                raise Exception
            res.append(text)
        return res

    def normalize(
        self, text: str, verbose: bool, punct_pre_process: bool = False, punct_post_process: bool = False
    ) -> str:
        """
        Main function. Normalizes tokens from written to spoken form
            e.g. 12 kg -> twelve kilograms

        Args:
            text: string that may include semiotic classes
            verbose: whether to print intermediate meta information
            punct_pre_process: whether to perform punctuation pre-processing, for example, [25] -> [ 25 ]
            punct_post_process: whether to normalize punctuation

        Returns: spoken form
        """
        if punct_pre_process:
            text = pre_process(text)
        text = text.strip()
        if not text:
            if verbose:
                print(text)
            return text
        text = pynini.escape(text)

        try:
            tagged_text = self.select_tag(text)
        except:
            print(text)
            import pdb

            pdb.set_trace()
            print()
        if verbose:
            print(tagged_text)
        self.parser(tagged_text)
        tokens = self.parser.parse()
        tags_reordered = self.generate_permutations(tokens)

        for tagged_text in tags_reordered:
            tagged_text = pynini.escape(tagged_text)
            try:
                output = self.select_verbalizer(tagged_text)
                if punct_post_process:
                    output = post_process_punctuation(output)
                return output
            except:
                continue
        raise ValueError()

    def _permute(self, d: OrderedDict) -> List[str]:
        """
        Creates reorderings of dictionary elements and serializes as strings

        Args:
            d: (nested) dictionary of key value pairs

        Return permutations of different string serializations of key value pairs
        """
        l = []
        if PRESERVE_ORDER_KEY in d.keys():
            d_permutations = [d.items()]
        else:
            d_permutations = itertools.permutations(d.items())
        for perm in d_permutations:
            subl = [""]
            for k, v in perm:
                if isinstance(v, str):
                    subl = ["".join(x) for x in itertools.product(subl, [f"{k}: \"{v}\" "])]
                elif isinstance(v, OrderedDict):
                    rec = self._permute(v)
                    subl = ["".join(x) for x in itertools.product(subl, [f" {k} {{ "], rec, [f" }} "])]
                elif isinstance(v, bool):
                    subl = ["".join(x) for x in itertools.product(subl, [f"{k}: true "])]
                else:
                    raise ValueError()
            l.extend(subl)
        return l

    def generate_permutations(self, tokens: List[dict]):
        """
        Generates permutations of string serializations of list of dictionaries

        Args:
            tokens: list of dictionaries

        Returns string serialization of list of dictionaries
        """

        def _helper(prefix: str, tokens: List[dict], idx: int):
            """
            Generates permutations of string serializations of given dictionary

            Args:
                tokens: list of dictionaries
                prefix: prefix string
                idx:    index of next dictionary

            Returns string serialization of dictionary
            """
            if idx == len(tokens):
                yield prefix
                return
            token_options = self._permute(tokens[idx])
            for token_option in token_options:
                yield from _helper(prefix + token_option, tokens, idx + 1)

        return _helper("", tokens, 0)

    def select_tag(self, text: str) -> str:
        """
        Return shortest path for the given text

        Args:
-            text: sentence

        Returns: shortest path
        """
        tagged_text = rewrite.top_rewrite(text, self.tagger.fst)
        return tagged_text

    def select_verbalizer(self, tagged_text: str) -> str:
        """
        Return shortest path given tagged text

        Args:
            tagged_text: input text

        Returns: shortest path
        """
        output = rewrite.top_rewrite(tagged_text, self.verbalizer.fst)
        return output


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("input_string", help="input string", type=str)
    parser.add_argument(
        "--input_case", help="input capitalization", choices=["lower_cased", "cased"], default="cased", type=str
    )
    parser.add_argument("--verbose", help="print info for debugging", action='store_true')
    parser.add_argument(
        "--punct_post_process", help="set to True to enable punctuation post processing", action="store_true"
    )
    parser.add_argument(
        "--punct_pre_process", help="set to True to enable punctuation pre processing", action="store_true"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    normalizer = Normalizer(input_case=args.input_case)
    print(
        normalizer.normalize(
            args.input_string,
            verbose=args.verbose,
            punct_pre_process=args.punct_pre_process,
            punct_post_process=args.punct_post_process,
        )
    )
