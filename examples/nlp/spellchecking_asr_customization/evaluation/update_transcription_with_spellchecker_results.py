import argparse
import json
import os
import re
from collections import defaultdict

from tqdm.auto import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--asr_hypotheses_folder", required=True, type=str, help="Input folder with asr hypotheses")
parser.add_argument(
    "--spellchecker_results_folder", required=True, type=str, help="Input folder with spellchecker output"
)
parser.add_argument("--input_manifest", required=True, type=str, help="Manifest with trancription before correction")
parser.add_argument("--output_manifest", required=True, type=str, help="Manifest with trancription after correction")
args = parser.parse_args()


def read_manifest(path):
    manifest = []
    with open(path, 'r') as f:
        for line in tqdm(f, desc="Reading manifest data"):
            line = line.replace("\n", "")
            data = json.loads(line)
            manifest.append(data)
    return manifest


def check_banned_replacements(src, dst):
    if src.endswith(" l") and dst.endswith(" l y") and src[0:-2] == dst[0:-4]:
        return True
    if dst.endswith(" l") and src.endswith(" l y") and dst[0:-2] == src[0:-4]:
        return True
    if src.endswith(" l") and dst.endswith(" l l y") and src[0:-2] == dst[0:-6]:
        return True
    if dst.endswith(" l") and src.endswith(" l l y") and dst[0:-2] == src[0:-6]:
        return True
    if src.endswith(" e") and dst.endswith(" e s") and src[0:-2] == dst[0:-4]:
        return True
    if dst.endswith(" e") and src.endswith(" e s") and dst[0:-2] == src[0:-4]:
        return True
    if src.endswith(" e") and dst.endswith(" a l") and src[0:-2] == dst[0:-4]:
        return True
    if dst.endswith(" e") and src.endswith(" a l") and dst[0:-2] == src[0:-4]:
        return True
    if src.endswith(" a") and dst.endswith(" a n") and src[0:-2] == dst[0:-4]:
        return True
    if dst.endswith(" a") and src.endswith(" a n") and dst[0:-2] == src[0:-4]:
        return True
    if src.endswith(" e") and dst.endswith(" y") and src[0:-2] == dst[0:-2]:
        return True
    if dst.endswith(" e") and src.endswith(" y") and dst[0:-2] == src[0:-2]:
        return True
    if src.endswith(" i e s") and dst.endswith(" y") and src[0:-6] == dst[0:-2]:
        return True
    if dst.endswith(" i e s") and src.endswith(" y") and dst[0:-6] == src[0:-2]:
        return True
    if src.endswith(" i e s") and dst.endswith(" y ' s") and src[0:-6] == dst[0:-6]:
        return True
    if dst.endswith(" i e s") and src.endswith(" y ' s") and dst[0:-6] == src[0:-6]:
        return True
    if src.endswith(" e") and dst.endswith(" i n g") and src[0:-2] == dst[0:-6]:
        return True
    if dst.endswith(" e") and src.endswith(" i n g") and dst[0:-2] == src[0:-6]:
        return True
    if src.endswith(" e s") and dst.endswith(" i n g") and src[0:-4] == dst[0:-6]:
        return True
    if dst.endswith(" e s") and src.endswith(" i n g") and dst[0:-4] == src[0:-6]:
        return True
    if src.endswith(" c e s") and dst.endswith(" x") and src[0:-6] == dst[0:-2]:
        return True
    if dst.endswith(" c e s") and src.endswith(" x") and dst[0:-6] == src[0:-2]:
        return True
    if src.endswith(" e") and dst.endswith(" e d") and src[0:-2] == dst[0:-4]:
        return True
    if dst.endswith(" e") and src.endswith(" e d") and dst[0:-2] == src[0:-4]:
        return True
    if src.endswith(" e") and dst.endswith(" i c") and src[0:-2] == dst[0:-4]:
        return True
    if dst.endswith(" e") and src.endswith(" i c") and dst[0:-2] == src[0:-4]:
        return True
    if src.endswith(" y") and dst.endswith(" i c") and src[0:-2] == dst[0:-4]:
        return True
    if dst.endswith(" y") and src.endswith(" i c") and dst[0:-2] == src[0:-4]:
        return True
    if src.endswith(" n") and dst.endswith(" n y") and src[0:-2] == dst[0:-4]:
        return True
    if dst.endswith(" n") and src.endswith(" n y") and dst[0:-2] == src[0:-4]:
        return True
    if src.endswith(" e d") and dst.endswith(" i n g") and src[0:-4] == dst[0:-6]:
        return True
    if dst.endswith(" e d") and src.endswith(" i n g") and dst[0:-4] == src[0:-6]:
        return True
    if src.endswith(" e n c e") and dst.endswith(" i n g") and src[0:-8] == dst[0:-6]:
        return True
    if dst.endswith(" e n c e") and src.endswith(" i n g") and dst[0:-8] == src[0:-6]:
        return True
    if src.endswith(" a n c e") and dst.endswith(" i n g") and src[0:-8] == dst[0:-6]:
        return True
    if dst.endswith(" a n c e") and src.endswith(" i n g") and dst[0:-8] == src[0:-6]:
        return True
    if src.endswith(" e d") and dst.endswith(" e s") and src[0:-4] == dst[0:-4]:
        return True
    if dst.endswith(" e d") and src.endswith(" e s") and dst[0:-4] == src[0:-4]:
        return True
    if src.endswith(" _ i s") and dst.endswith(" e s") and src[0:-6] == dst[0:-4]:
        return True
    if dst.endswith(" _ i s") and src.endswith(" e s") and dst[0:-6] == src[0:-4]:
        return True
    if src.endswith(" _ i s") and dst.endswith(" ' s") and src[0:-6] == dst[0:-4]:
        return True
    if dst.endswith(" _ i s") and src.endswith(" ' s") and dst[0:-6] == src[0:-4]:
        return True
    if src.endswith(" _ i s") and dst.endswith(" s") and src[0:-6] == dst[0:-2]:
        return True
    if dst.endswith(" _ i s") and src.endswith(" s") and dst[0:-6] == src[0:-2]:
        return True
    if src.endswith(" _ a s") and dst.endswith(" ' s") and src[0:-6] == dst[0:-4]:
        return True
    if dst.endswith(" _ a s") and src.endswith(" ' s") and dst[0:-6] == src[0:-4]:
        return True
    if src.endswith(" _ h a s") and dst.endswith(" ' s") and src[0:-8] == dst[0:-4]:
        return True
    if dst.endswith(" _ h a s") and src.endswith(" ' s") and dst[0:-8] == src[0:-4]:
        return True
    if src.endswith(" t i o n") and dst.endswith(" t e d") and src[0:-8] == dst[0:-6]:
        return True
    if dst.endswith(" t i o n") and src.endswith(" t e d") and dst[0:-8] == src[0:-6]:
        return True
    if src.endswith(" t i o n") and dst.endswith(" t i v e") and src[0:-8] == dst[0:-8]:
        return True
    if dst.endswith(" t i o n") and src.endswith(" t i v e") and dst[0:-8] == src[0:-8]:
        return True
    if src.endswith(" s m") and dst.endswith(" s t s") and src[0:-4] == dst[0:-6]:
        return True
    if dst.endswith(" s m") and src.endswith(" s t s") and dst[0:-4] == src[0:-6]:
        return True
    if src.endswith(" s m") and dst.endswith(" s t") and src[0:-4] == dst[0:-4]:
        return True
    if dst.endswith(" s m") and src.endswith(" s t") and dst[0:-4] == src[0:-4]:
        return True

    if src.endswith(" '") and src[0:-2] == dst:
        return True
    if dst.endswith(" '") and dst[0:-2] == src:
        return True
    if src.endswith(" ' s") and dst.endswith(" s") and src[0:-4] == dst[0:-2]:
        return True
    if dst.endswith(" ' s") and src.endswith(" s") and dst[0:-4] == src[0:-2]:
        return True
    if src.endswith(" ' s") and dst.endswith(" e s") and src[0:-4] == dst[0:-4]:
        return True
    if dst.endswith(" ' s") and src.endswith(" e s") and dst[0:-4] == src[0:-4]:
        return True
    if src.endswith(" ' s") and dst.endswith(" y") and src[0:-4] == dst[0:-2]:
        return True
    if dst.endswith(" ' s") and src.endswith(" y") and dst[0:-4] == src[0:-2]:
        return True
    if src.endswith(" ' s") and src[0:-4] == dst:
        return True
    if dst.endswith(" ' s") and dst[0:-4] == src:
        return True
    if src.endswith(" s") and src[0:-2] == dst:
        return True
    if dst.endswith(" s") and dst[0:-2] == src:
        return True
    if src.endswith(" e d") and src[0:-4] == dst:
        return True
    if dst.endswith(" e d") and dst[0:-4] == src:
        return True

    if src.startswith("i n _ ") and src[6:] == dst:
        return True
    if dst.startswith("i n _ ") and dst[6:] == src:
        return True
    if src.startswith("o n _ ") and src[6:] == dst:
        return True
    if dst.startswith("o n _ ") and dst[6:] == src:
        return True
    if src.startswith("o f _ ") and src[6:] == dst:
        return True
    if dst.startswith("o f _ ") and dst[6:] == src:
        return True
    if src.startswith("a t _ ") and src[6:] == dst:
        return True
    if dst.startswith("a t _ ") and dst[6:] == src:
        return True

    if src.startswith("u n ") and src[4:] == dst:
        return True
    if dst.startswith("u n ") and dst[4:] == src:
        return True

    if src.startswith("r e ") and src[4:] == dst:
        return True
    if dst.startswith("r e ") and dst[4:] == src:
        return True

    if (
        dst == "t h r o u g h"
        or dst == "w i t h"
        or dst == "y o u ' v e"
        or dst == "w e ' v e"
        or dst == "a c t"
        or dst == "s e p t e m b e r"
        or dst == "n o v e m b e r"
        or dst == "o c t o b e r"
        or dst == "m a y"
        or dst == "j a n u a r y"
        or dst == "f e b r u a r y"
        or dst == "d e c e m b e r"
    ):
        return True

    if src != dst and (src.startswith(dst) or dst.startswith(src) or src.endswith(dst) or dst.endswith(src)):
        return True

    dummy_candidates = [
        "a g k t t r k n a p r t f",
        "v w w x y x u r t g p w q",
        "n t r y t q q r u p t l n t",
        "p b r t u r e t f v w x u p z",
        "p p o j j k l n b f q t",
        "j k y u i t d s e w s r e j h i p p",
        "q w r e s f c t d r q g g y",
    ]
    if dst in dummy_candidates:
        return True

    return False


final_corrections = defaultdict(str)
banned_count = 0
for name in os.listdir(args.spellchecker_results_folder):
    doc_id, _ = name.split(".")
    short2full_sent = defaultdict(list)
    full_sent2corrections = defaultdict(dict)
    try:
        with open(args.asr_hypotheses_folder + "/" + doc_id + ".txt", "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                short_sent, full_sent = s.split("\t")
                short_sent = " ".join(list(short_sent.replace(" ", "_")))
                full_sent = " ".join(list(full_sent.replace(" ", "_")))
                short2full_sent[short_sent].append(full_sent)
        print("len(short2full_sent)=", len(short2full_sent))
    except:
        continue

    with open(args.spellchecker_results_folder + "/" + doc_id + ".txt", "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s.startswith("REPLACE"):
                continue
            parts = s.split("\t")
            _, src, dst, short_sent = parts
            if short_sent not in short2full_sent:
                continue
            if check_banned_replacements(src, dst):
                print("!!!", src, " => ", dst)
                banned_count += 1
                continue
            for full_sent in short2full_sent[short_sent]:  # mostly there will be one-to-one correspondence
                if full_sent not in full_sent2corrections:
                    full_sent2corrections[full_sent] = {}
                if src not in full_sent2corrections[full_sent]:
                    full_sent2corrections[full_sent][src] = {}
                if dst not in full_sent2corrections[full_sent][src]:
                    full_sent2corrections[full_sent][src][dst] = 0
                full_sent2corrections[full_sent][src][dst] += 1

    print("len(full_sent2corrections)=", len(full_sent2corrections))

    for full_sent in full_sent2corrections:
        corrected_full_sent = full_sent
        for src in full_sent2corrections[full_sent]:
            for dst, freq in sorted(
                full_sent2corrections[full_sent][src].items(), key=lambda item: item[1], reverse=True
            ):
                corrected_full_sent = corrected_full_sent.replace(src, dst)
                # take only best variant
                break
        original_full_sent = "".join(full_sent.split()).replace(
            "_", " "
        )  # restore original format instead of separate letters
        corrected_full_sent = "".join(corrected_full_sent.split()).replace(
            "_", " "
        )  # restore original format instead of separate letters
        final_corrections[doc_id + "\t" + original_full_sent] = corrected_full_sent


print("len(final_corrections)=", len(final_corrections))

test_data = read_manifest(args.input_manifest)

# extract just the text corpus from the manifest
pred_text = [data['pred_text'] for data in test_data]
audio_filepath = [data['audio_filepath'] for data in test_data]
durations = [data['duration'] for data in test_data]

print("duration=", sum(durations))

for i in range(len(test_data)):
    sent, path = pred_text[i], audio_filepath[i]
    # example of path: ...clips/197_0000.wav   #doc_id=197
    path_parts = path.split("/")
    path_parts2 = path_parts[-1].split("_")
    doc_id = path_parts2[-2]
    k = doc_id + "\t" + sent
    if k in final_corrections:
        test_data[i]["before_spell_pred"] = test_data[i]["pred_text"]
        test_data[i]["pred_text"] = final_corrections[k]

with open(args.output_manifest, "w", encoding="utf-8") as out:
    for d in test_data:
        line = json.dumps(d)
        out.write(line + "\n")

print("banned count=", banned_count)
