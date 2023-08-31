# python materialize_text_data.py /media/data/datasets/LibriSpeech/en /tmp/t.manifest
import json
import random
import sys

en_data = open(sys.argv[1], 'r').readlines()
content = ""
for en in en_data:
    en = en.strip().lower()
    ens = en.split()
    if len(ens) < 5 or (ens[0] == ens[1] and ens[0] == ens[2]):
        continue
    ens = ens[:64]
    en = " ".join(ens)
    record = {}
    record['text'] = f"{en}"
    content += json.dumps(record)
    content += "\n"
final = open(sys.argv[2], "w")
final.write(content)
final.close()
