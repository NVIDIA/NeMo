import re
from argparse import ArgumentParser
from collections import defaultdict

parser = ArgumentParser(description="Clean YAGO entities")
parser.add_argument("--input_name", type=str, required=True, help="Input file")
parser.add_argument("--output_name", type=str, required=True, help="Output file")
parser.add_argument("--vocab_name", type=str, required=True, help="Output vocab file")

args = parser.parse_args()


def replace_diacritics(text):
    text = re.sub(r"[éèëēêęěė]", "e", text)
    text = re.sub(r"[ãâāáäăâàąåạả]", "a", text)
    text = re.sub(r"[úūüùưûů]", "u", text)
    text = re.sub(r"[ôōóöõòő]", "o", text)
    text = re.sub(r"[ćçč]", "c", text)
    text = re.sub(r"[ïīíîıì]", "i", text)
    text = re.sub(r"[ñńňņ]", "n", text)
    text = re.sub(r"[țť]", "t", text)
    text = re.sub(r"[łľ]", "l", text)
    text = re.sub(r"[żžź]", "z", text)
    text = re.sub(r"[ğ]", "g", text)
    text = re.sub(r"[ř]", "r", text)
    text = re.sub(r"[ý]", "y", text)
    text = re.sub(r"[æ]", "ae", text)
    text = re.sub(r"[œ]", "oe", text)
    text = re.sub(r"[șşšś]", "s", text)
    return text


out = open(args.output_name, "w", encoding="utf-8")

vocab = set()

with open(args.input_name, "r", encoding="utf-8") as inp:
    for line in inp:
        s = line.strip()
        s = s.replace("<", "").replace(">", "")
        s = s.casefold()
        s = re.sub(r"\(.+\)", r"", s)  # delete brackets
        s = s.replace("_", " ")
        s = s.replace("/", ",")
        parts = s.split(",")
        for p in parts:
            sp = p.strip()
            if len(sp) < 3:
                continue
            if "." in sp:
                continue
            if re.match(r".*\d", sp):
                continue
            sp = replace_diacritics(sp)
            sp = " ".join(sp.split())
            if len(set(list(sp)) - set(list(" -'abcdefghijklmnopqrstuvwxyz"))) == 0:
                out.write(sp + "\n")
                words = sp.replace("-", " ").split(" ")
                for w in words:
                    while w.startswith("'") and w.endswith("'"):
                        w = w[1:-1]
                    w = w.strip()
                    if len(w) > 0:
                        vocab.add(w)
            else:
                print(str(set(list(sp)) - set(list(" -'abcdefghijklmnopqrstuvwxyz"))))

out.close()

with open(args.vocab_name, "w", encoding="utf-8") as out:
    for w in sorted(list(vocab)):
        out.write(" ".join(list(w)) + "\n")
