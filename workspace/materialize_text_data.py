# python materialize_text_data.py /media/data/datasets/LibriSpeech/en /tmp/t.manifest
import json
import random
import sys

en_data = open(sys.argv[1], 'r').readlines()
final = open(sys.argv[2], "w")
for en in en_data:
    en = en.strip()
    record = {}
    record['text'] = f"{en}"
    json.dump(record, final)
    final.write('\n')
final.close()
