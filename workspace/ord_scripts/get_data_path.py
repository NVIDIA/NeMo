import argparse
import json
import math
import os
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

from tqdm import tqdm

parser = argparse.ArgumentParser(description='Get data by language')

parser.add_argument('--input_file', type=str, help='Input file', default="./ssl_datasets.sh")
parser.add_argument('--output_file', type=str, help='Output file', default="./data_paths.sh")
parser.add_argument('--data_weight', type=str, help='Type of weights to use', default="default")
parser.add_argument('--scale', type=float, help='Scale factor for scaled data weights', default=0.5)
args = parser.parse_args()

input_file = args.input_file
output_file = args.output_file


def get_duration_hours(manifest_file):
    duration = 0
    min_dur = float('inf')
    max_dur = 0
    with open(manifest_file, 'r') as f:
        for line in f.readlines():
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            duration += data["duration"]
            min_dur = min(min_dur, data["duration"])
            max_dur = max(max_dur, data["duration"])
    return duration / 3600, min_dur, max_dur


def get_data_duration(manifest_line):
    manifest_line = manifest_line.replace("[", "").replace("]", "").replace(" ", "").replace("'", "").replace('"', "")
    manifests = manifest_line.split(",")
    duration = 0.0
    min_dur = float('inf')
    max_dur = 0
    for manifest in manifests:
        total_dur, _min_dur, _max_dur = get_duration_hours(manifest)
        duration += total_dur
        min_dur = min(min_dur, _min_dur)
        max_dur = max(max_dur, _max_dur)
    return duration, min_dur, max_dur


data_paths = {}
train_lang_hours = defaultdict(float)
train_data_hours = {}
train_data_weight = {}
train_data_min_dur = {}
train_data_max_dur = {}
train_manifest_files = defaultdict(list)
train_tar_files = defaultdict(list)
val_manifest_files = defaultdict(list)
val_lang_hours = defaultdict(float)

train_keys_by_lang = defaultdict(list)
val_keys_by_lang = defaultdict(list)
val_data_min_dur = {}
val_data_max_dur = {}
val_data_hours = {}

tmp_file = ".tmp.sh"
ftmp = open(tmp_file, "w")
ftmp.write(f"#!/bin/bash\n")
with open(input_file, 'r') as f:
    for line in f:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        ftmp.write("export " + line + "\n")
ftmp.close()
# Prepare the command to execute the script
command = f'bash -c "source {tmp_file} && env"'
env_dict = {}
try:
    # Run the command and capture the output
    result = subprocess.run(command, shell=True, capture_output=True, text=True, check=True)

    # Extract environment variables from the output
    env_vars = result.stdout.strip().split('\n')

    # Convert to dictionary
    for var in env_vars:
        key, _, value = var.partition('=')
        if key and value:
            env_dict[key] = value
except subprocess.CalledProcessError as e:
    print(f"Error executing script: {e}", file=sys.stderr)
os.system(f"rm {tmp_file}")

total = len(env_dict)
i = 0
for key, value in env_dict.items():
    i += 1
    if ".json" not in value and ".tar" not in value:
        continue

    print(f"[{i}/{total}] Processing {key}")

    if ".json" in value:
        if "_TRAIN_" in key:
            lang = key.split("TRAIN_", 1)[1].split("_")[0]
            hours, min_dur, max_dur = get_data_duration(value)
            train_lang_hours[lang] += hours
            train_data_hours[key] = hours
            train_data_min_dur[key] = min_dur
            train_data_max_dur[key] = max_dur
            train_manifest_files[lang].append(key)
        elif "_VAL_" in key:
            lang = key.split("VAL_", 1)[1].split("_")[0]
            hours, min_dur, max_dur = get_data_duration(value)
            val_lang_hours[lang] += hours
            val_manifest_files[lang].append(key)
            val_data_hours[key] = hours
            val_data_min_dur[key] = min_dur
            val_data_max_dur[key] = max_dur
    elif ".tar" in value:
        if "_TRAIN_" in key:
            lang = key.split("TRAIN_", 1)[1].split("_")[0]
            train_tar_files[lang].append(key)

total_train_hours = sum(train_lang_hours.values())

train_weight_by_lang = {}
for lang in train_lang_hours.keys():
    train_weight_by_lang[lang] = (train_lang_hours[lang] / total_train_hours) ** args.scale

for key in train_data_hours.keys():
    lang = key.split("TRAIN_", 1)[1].split("_")[0]
    train_data_weight[key] = train_data_hours[key] / train_lang_hours[lang] * train_weight_by_lang[lang]

total_weight = sum(train_data_weight.values())
for key in train_data_weight.keys():
    train_data_weight[key] /= total_weight


full_train_manifest = []
full_train_filepath = []
full_train_weight = []
full_val_manifest = []
fout = open(output_file, "w")
fout.write("#!/bin/bash\n")
fout.write(f"# sampling scale: {args.scale}\n")

# write total train/val hours
fout.write(
    f"# Total train hours: {total_train_hours:.2f}, min dur secs: {min(train_data_min_dur.values()):.2f}, max dur secs: {max(train_data_max_dur.values()):.2f}\n"
)
fout.write(
    f"# Total val hours: {sum(val_lang_hours.values()):.2f}, min dur secs: {min(val_data_min_dur.values()):.2f}, max dur secs: {max(val_data_max_dur.values()):.2f}\n\n"
)

fout.write("\n\n\n")

fout.write(f"# Training hours by dataset\n")
for key, hours in sorted(train_data_hours.items(), key=lambda x: x[0]):
    lang = key.split("TRAIN_", 1)[1].split("_")[0]
    fout.write(
        f"# {key}: {hours:.2f} hrs, weight={train_data_weight[key]:.4f}, min dur secs={train_data_min_dur[key]:.2f}, max dur secs={train_data_max_dur[key]:.2f}\n"
    )


fout.write("\n\n\n")
fout.write(f"# Validation hours by dataset\n")
for key, hours in sorted(val_data_hours.items(), key=lambda x: x[0]):
    lang = key.split("VAL_", 1)[1].split("_")[0]
    fout.write(
        f"# {key}: {hours:.2f} hrs, min dur secs={val_data_min_dur[key]:.2f}, max dur secs={val_data_max_dur[key]:.2f}\n"
    )


fout.write("\n\n\n")
fout.write(f"# Hours by language\n")
for lang, hours in sorted(train_lang_hours.items(), key=lambda x: x[1], reverse=True):
    fout.write(
        f"# {lang}: train={hours:.2f} hrs, val={val_lang_hours.get(lang, 0):.2f} hrs, weight={train_weight_by_lang.get(lang, 0):.4f}\n"
    )
fout.write("\n\n\n")


for lang, hours in sorted(train_lang_hours.items(), key=lambda x: x[1], reverse=True):
    data_keys = train_manifest_files[lang]
    lang_manifest = []
    lang_filepath = []
    lang_weight = []
    for key in data_keys:
        manifest_file = env_dict[key]
        tar_file = env_dict[key.replace("_MANIFEST", "_FILEPATH")]
        weight = train_data_weight[key]
        lang_manifest.append(manifest_file)
        lang_filepath.append(tar_file)
        lang_weight.append(str(weight))
    fout.write(f"\n\n##### {lang} Data Paths #####\n")
    fout.write(f"# {lang} TRAIN_HOURS={hours:.2f}\n")
    fout.write(f"TRAIN_{lang}_MANIFEST=\"{','.join(lang_manifest)}\"\n")
    fout.write(f"TRAIN_{lang}_FILEPATH=\"{','.join(lang_filepath)}\"\n")
    fout.write(f"TRAIN_{lang}_WEIGHT=\"{','.join(lang_weight)}\"\n\n")

    if lang in val_manifest_files:
        val_manifest = [env_dict[key] for key in val_manifest_files[lang]]
        fout.write(f"# {lang} VAL_HOURS={val_lang_hours[lang]:.2f}\n")
        fout.write(f"VAL_{lang}_MANIFEST=\"{','.join(val_manifest)}\"\n")
        full_val_manifest.extend(val_manifest)

    full_train_manifest.extend(lang_manifest)
    full_train_filepath.extend(lang_filepath)
    full_train_weight.extend(lang_weight)


full_train_manifest = ",".join(full_train_manifest)
full_train_filepath = ",".join(full_train_filepath)
full_train_weight = ",".join(full_train_weight)
full_val_manifest = ",".join(full_val_manifest)


fout.write("\n##### Full Train Data Path #####\n")
fout.write(f'TRAIN_MANIFEST=\"[{full_train_manifest}]\"\n\n')
fout.write(f'TRAIN_FILEPATH=\"[{full_train_filepath}]\"\n\n')
fout.write(f"TRAIN_WEIGHT=\"[{full_train_weight}]\"\n\n")
fout.write("\n\n##### Full Validation Data Path #####\n")
fout.write(f"VAL_MANIFEST=\"[{full_val_manifest}]\"\n\n")
fout.close()
