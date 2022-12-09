from argparse import ArgumentParser
from collections import Counter


parser = ArgumentParser(
    description="Get list of all target phrases from Yago corpus with their frequencies (counts one occurrence per paragraph)"
)
parser.add_argument("--input_file", required=True, type=str, help="Path to input file with phrases")
parser.add_argument("--output_file", type=str, required=True, help="Output file")

args = parser.parse_args()


vocab = Counter()
with open(args.input_file, "r", encoding="utf-8") as f:
    for line in f:
        text = line.strip().split("\t")[0]  # if line is tab-separated only first part is considered as text
        phrases = text.split(";")
        for phrase in phrases:
            vocab[phrase] += 1

with open(args.output_file, "w", encoding="utf-8") as out:
    for phrase, freq in vocab.most_common(10000000):
        out.write(phrase + "\t" + str(freq) + "\n")