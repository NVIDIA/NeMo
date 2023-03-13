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

from nemo.collections.nlp.data.spellchecking_asr_customization.utils import load_index


parser = ArgumentParser(description="Create index for custom phrases, allows to use parameters")
parser.add_argument("--input_file", required=True, type=str, help="Path to input file with custom phrases")
parser.add_argument("--output_file", type=str, required=True, help="Output file")

args = parser.parse_args()

phrases, ngram2phrases = load_index(args.input_file)
print("len(phrases)=", len(phrases), "; len(ngram2phrases)=", len(ngram2phrases))

with open(args.output_file, "w", encoding="utf-8") as out:
    n = 0
    for phrase in phrases:
        letters = phrase.split(" ")
        begin = 0
        related_phrases = {}
        for begin in range(len(letters)):
            for end in range(begin + 1, min(len(letters) + 1, begin + 7)):
                ngram = " ".join(letters[begin:end])
                if ngram not in ngram2phrases:
                    continue
                for phrase_id, b, size, lp in ngram2phrases[ngram]:
                    if phrase_id not in related_phrases:
                        related_phrases[phrase_id] = [0] * len(letters)  # set()
                    related_phrases[phrase_id][begin:end] = [1] * (end - begin)  # .add(ngram)
        k = 0
        for related_phrase_id, mask in sorted(related_phrases.items(), key=lambda item: sum(item[1]), reverse=True):
            phr = phrases[related_phrase_id]
            out.write("".join(letters) + "\t" + "".join(phr.split()) + "\t" + str(sum(mask)) + "\n")
            k += 1
            if k > 10:
                break
        n += 1
        if n % 10000 == 0:
            print(n)
