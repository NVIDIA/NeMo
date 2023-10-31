import random
from datasets import load_dataset
from tqdm import tqdm
import json

output_manifest = "./wizard_evolved_80k.json"

dataset = load_dataset("nickrosh/Evol-Instruct-Code-80k-v1", split="train")

new_samples = []
for sample_idx, sample in tqdm(enumerate(dataset), desc="Loading samples..."):
    new_sample = {}
    new_sample["input"] = sample["instruction"]
    new_sample["output"] = sample["output"]
    new_samples.append(new_sample)


with open(output_manifest, 'w', encoding='utf-8') as outf:
    print("Writing the output file...")
    for sample in new_samples:
        outf.write(json.dumps(sample) + "\n")
