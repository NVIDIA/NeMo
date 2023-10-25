import random
from datasets import load_dataset
from tqdm import tqdm
import json
from tts_normalization_utils import get_normalizer, normalize

random.seed(1402)

dataset_name = "squadv2"
split_name = "train"
output_file = f"{dataset_name}_{split_name}_not_normalized.json"
MSG_NOT_FOUND_ANSWER = "I could not find the answer in the audio."

do_shuffle = False
use_wellformed_answer = True
do_normalize = False
do_lowercase = False


dataset = load_dataset("squad_v2", split=split_name)
if do_normalize:
    normalizer = get_normalizer()
else:
    normalizer = None

positive_samples_num = 0
negative_samples_num = 0
new_samples = []
for sample_idx, sample in tqdm(enumerate(dataset), desc="Loading samples..."):
    new_sample = {}
    context = sample["context"]
    if normalizer:
        context = normalize(text=context, normalizer=normalizer, do_lowercase=do_lowercase)

    output = sample["answers"]["text"]
    if len(output) == 0:
        output = MSG_NOT_FOUND_ANSWER
        negative_samples_num += 1
    else:
        output = output[0]
        positive_samples_num += 1

    new_sample["sample_id"] = f"{dataset_name}_{sample['id']}"
    new_sample["instruction"] = sample["question"]
    new_sample["context"] = context
    new_sample["output"] = output
    new_samples.append(new_sample)

print("Total seed samples:", len(dataset))
print("Total #samples:", len(new_samples))
print("Positive #samples:", positive_samples_num)
print("Negative #samples:", negative_samples_num)
print("Last Sample", new_samples[-1])

if do_shuffle:
    random.shuffle(new_samples)
with open(output_file, 'w', encoding='utf-8') as outf:
    print("Writing the output file...")
    for sample in new_samples:
        outf.write(json.dumps(sample) + "\n")

