# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

from nemo.collections.nlp.data.spellchecking_asr_customization.utils import get_candidates, load_index

parser = ArgumentParser(description="Get candidates, given custom index and input sentences")
parser.add_argument("--index_file", required=True, type=str, help="Path to index with custom phrases")
parser.add_argument("--input_file", required=True, type=str, help="Path to file with input phrases")
parser.add_argument("--output_file", required=True, type=str, help="Output file")
parser.add_argument("--max_candidates", type=int, default=10, help="Maximum number of candidates")
parser.add_argument(
    "--min_real_coverage", type=float, default=0.8, help="Minimum fraction of phrase covered by matching ngrams"
)
parser.add_argument(
    "--match_whole_input",
    type=bool,
    default=False,
    help="If true, will look for candidates matching the whole input text",
)
parser.add_argument(
    "--skip_empty", type=bool, default=False, help="If true, will skip lines for which no candidates were found"
)
parser.add_argument(
    "--skip_same", type=bool, default=False, help="If true, will skip candidates that are exactly same as input"
)

args = parser.parse_args()

phrases, ngram2phrases = load_index(args.index_file)
print("len(phrases)=", len(phrases), "; len(ngram2phrases)=", len(ngram2phrases))

phrase_lengths = [phrase.count(" ") + 1 for phrase in phrases]

out = open(args.output_file, "w", encoding="utf-8")
with open(args.input_file, "r", encoding="utf-8") as f:
    n = 0
    for line in f:
        text = line.strip().split("\t")[0]  # if line is tab-separated only first part is considered as text
        letters = list(text.replace(" ", "_"))
        candidates = get_candidates(
            ngram2phrases,
            phrases,
            phrase_lengths,
            letters,
            max_candidates=10,
            min_real_coverage=0.8,
            match_whole_input=args.match_whole_input,
        )
        result = []
        for cand in candidates:
            cand_text = "".join(cand.split(" ")).replace("_", " ")
            if args.skip_same and cand_text == text:
                continue
            else:
                result.append(cand_text)
        if args.skip_empty and len(result) == 0:
            continue
        out.write(text + "\t" + ";".join(result) + "\n")
        n += 1
        if n % 100 == 0:
            print(n)
out.close()
