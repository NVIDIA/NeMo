import json
import os
import re
from collections import Counter
from tqdm.auto import tqdm

from argparse import ArgumentParser

parser = ArgumentParser(description="Analyze errors in ASR predictions")
parser.add_argument(
    "--manifest", required=True, type=str, help='Path to manifest file'
)
parser.add_argument(
    "--folder", required=True, type=str, help='Path to output folder'
)
args = parser.parse_args()


def read_manifest(path):
    manifest = []
    with open(path, 'r') as f:
        for line in tqdm(f, desc="Reading manifest data"):
            line = line.replace("\n", "")
            data = json.loads(line)
            manifest.append(data)
    return manifest


def save_hypotheses(hypotheses, doc_id):
    with open(args.folder + "/" + doc_id + ".txt", "w", encoding="utf-8") as out:
        for short, full in hypotheses:
            out.write(short + "\t" + full + "\n")


test_data = read_manifest(args.manifest)

# extract just the text corpus from the manifest
pred_text = [data['pred_text'] for data in test_data]
audio_filepath = [data['audio_filepath'] for data in test_data]

last_doc_id = ""
hypotheses = []
for sent, path in zip(pred_text, audio_filepath):
    # example of path: ...clips/197_0000.wav   #doc_id=197
    path_parts = path.split("/")
    path_parts2 = path_parts[-1].split("_")
    doc_id = path_parts2[-2]
    if last_doc_id != "" and doc_id != last_doc_id:
        save_hypotheses(hypotheses, last_doc_id)
        hypotheses = []
    last_doc_id = doc_id

    words = sent.split()
    for i in range(0, len(words), 2):
        short_sent = " ".join(words[i:i+10])
        if len(short_sent) > 8:
            hypotheses.append((short_sent, sent))

save_hypotheses(hypotheses, last_doc_id)


        
        


