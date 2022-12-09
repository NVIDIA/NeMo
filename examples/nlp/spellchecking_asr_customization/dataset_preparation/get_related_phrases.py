import sys
from argparse import ArgumentParser
from collections import defaultdict
from os.path import join
from typing import List, Dict, Optional, TextIO, Tuple


## !!!this is temporary hack for my windows machine since is misses some installs 
sys.path.insert(1, "D:\\data\\work\\nemo\\nemo\\collections\\nlp\\data\\spellchecking_asr_customization")
print(sys.path)
from utils import load_index
# from nemo.collections.nlp.data.spellchecking_asr_customization.utils import load_index


parser = ArgumentParser(
    description="Create index for custom phrases, allows to use parameters"
)
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