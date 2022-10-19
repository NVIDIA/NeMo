from argparse import ArgumentParser

parser = ArgumentParser(description="Prepare input for TTS")
parser.add_argument("--yago_input_name", type=str, required=True, help="Input file")
parser.add_argument("--phonematic_name", type=str, required=True, help="Input file with tagger g2p output")
parser.add_argument("--output_name", type=str, required=True, help="Output file")

args = parser.parse_args()

vocab = {}

with open(args.phonematic_name, "r", encoding="utf-8") as inp:
    for line in inp:
        parts = line.strip().split("\t")
        pred = parts[2]
        pred = pred.replace("<DELETE>", "").replace("_", " ")
        pred = " ".join(pred.split())
        if pred == "":
            continue
        inp = parts[1]
        vocab[parts[1]] = pred

out = open(args.output_name, "w", encoding="utf-8")

with open(args.yago_input_name, "r", encoding="utf-8") as inp:
    for line in inp:
        s = line.strip()
        s = " ".join(s.split())
        s2 = " ".join(s.replace("-", " ").split())
        parts = s2.split(" ")
        res = []
        ok = True
        for p in parts:
            k = " ".join(list(p))
            if k not in vocab:
                print("not found: " + k)
                ok = False
                continue
            v = vocab[k]
            res.extend(v.split())
            res.append(" ")
        if len(res) > 26:
            print("too long: ", s, res)
            ok = False
        if ok:
            res = res[:-1]
            out.write(s + "\t" + ",".join(res) + "\n")

out.close()
