import json
import random

from datasets import load_dataset
from tqdm import tqdm

output_manifest = "./wizard_evolved_80k.json"

dataset = load_dataset("nickrosh/Evol-Instruct-Code-80k-v1", split="train")

new_samples = []
max_input = max_output = max_total = 0
for sample_idx, sample in tqdm(enumerate(dataset), desc="Loading samples..."):
    new_sample = {}
    new_sample["input"] = sample["instruction"]
    new_sample["output"] = sample["output"]
    new_samples.append(new_sample)
    input_len = len(new_sample["input"].split())
    output_len = len(new_sample["output"].split())
    total_len = input_len + output_len
    if input_len > max_input:
        max_input = input_len
    if output_len > max_output:
        max_output = output_len
    if total_len > max_total:
        max_total = total_len

print(max_input, max_output, max_total)

with open(output_manifest, 'w', encoding='utf-8') as outf:
    print("Writing the output file...")
    for sample in new_samples:
        outf.write(json.dumps(sample) + "\n")
