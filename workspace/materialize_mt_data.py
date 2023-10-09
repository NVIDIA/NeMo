# python materialize_mt_data.py /media/data/datasets/LibriSpeech/en /media/data/datasets/LibriSpeech/zh /media/data/datasets/LibriSpeech/question_en_zh /tmp/t.manifest
import json
import random
import sys

en_data = open(sys.argv[1], 'r').readlines()
zh_data = open(sys.argv[2], 'r').readlines()
question_data = open(sys.argv[3], 'r').readlines()
content = ""
for en, zh in zip(en_data, zh_data):
    en = en.strip()
    zh = zh.strip()
    question = random.choice(question_data).strip()
    record = {}
    record['question'] = f"{question}: {en}\n"
    record['text'] = f"{zh}"
    content += json.dumps(record)
    content += "\n"
final = open(sys.argv[4], "w")
final.write(content)
final.close()
