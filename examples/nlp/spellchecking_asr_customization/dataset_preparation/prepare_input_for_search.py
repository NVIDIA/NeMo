import json
import random
from tqdm.auto import tqdm
from os.path import join
from argparse import ArgumentParser
from typing import Dict, Optional, TextIO, Tuple

parser = ArgumentParser(description="Prepare input for testing search: insert custom phrases into sample sentences")
parser.add_argument(
    "--input_manifest", required=True, type=str, help='Path to manifest file with sample sentences'
)
parser.add_argument("--output_name", type=str, required=True, help="Output file")
parser.add_argument("--input_name", type=str, required=True, help="Path to simulated custom vocabulary")

args = parser.parse_args()

def process_line(line: str) -> Optional[Tuple[str, str, str, int]]:
    """A helper function to read the file with alignment results"""

    parts = line.strip().split("\t")
    if len(parts) != 4:
        return None
    if parts[0] != "good:":
        return None

    src, dst, align = parts[1], parts[2], parts[3]

    return src, dst, align


def read_manifest(path):
    manifest = []
    with open(path, 'r') as f:
        for line in tqdm(f, desc="Reading manifest data"):
            line = line.replace("\n", "")
            data = json.loads(line)
            manifest.append(data)
    return manifest


def read_custom_vocab():
    refs = []
    hyps = []
    with open(args.input_name, "r", encoding="utf-8") as f:
        for line in f:
            t = process_line(line)
            if t is None:
                continue
            ref, hyp, _ = t
            refs.append(ref)
            hyps.append(hyp)
    return refs, hyps


refs, hyps = read_custom_vocab()
test_data = read_manifest(args.input_manifest)

# extract just the text corpus from the manifest
text = [data['text'] for data in test_data]
durations = [data['duration'] for data in test_data]


with open(args.output_name, "w", encoding="utf-8") as out:
    for i in range(len(text)):
        duration = durations[i]
        if duration > 7.0:
            continue
        sent = text[i]
        words = sent.split()

        # choose random position to insert custom phrase
        r = random.randrange(len(words))
        sent_begin = "_".join(words[0:r])
        sent_end = "_".join(words[r:])

        sent_begin_letters = " ".join(list(sent_begin))
        sent_end_letters = " ".join(list(sent_end))

        p = random.randrange(len(refs))
        final_sent = sent_begin_letters + " _ " + hyps[p] + " _ " + sent_end_letters
        hyp_len = len(hyps[p].split(" "))
        begin_len = len(list(sent_begin))

        out.write(final_sent + "\t" + refs[p] + "\t" + str(begin_len) + "\t" + str(hyp_len) + "\n")


