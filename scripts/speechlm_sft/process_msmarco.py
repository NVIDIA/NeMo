import random
from datasets import load_dataset
from tqdm import tqdm
import json
from tts_normalization_utils import get_normalizer, normalize

random.seed(1402)

dataset_name = "msmarco"
split_name = "validation"
split_name = "validation"
output_file = f"{dataset_name}_{split_name}_normalized.json"
MSG_NOT_FOUND_ANSWER = "I could not find the answer in the audio."

do_shuffle = False
use_wellformed_answer = True
do_normalize = True
do_lowercase = False


dataset = load_dataset("ms_marco", "v2.1", split=split_name)
if do_normalize:
    normalizer = get_normalizer()
else:
    normalizer = None

positive_samples_num = 0
negative_samples_num = 0
multi_answers_num = 0
new_samples = []
for sample_idx, sample in tqdm(enumerate(dataset), desc="Loading samples..."):
    is_selected = sample['passages']['is_selected']
    if len(sample["answers"]) > 1:
        multi_answers_num += 1
        continue
    new_sample = {}
    if is_selected.count(1) == 0:
        new_sample["output"] = MSG_NOT_FOUND_ANSWER
        negative_samples_num += 1
        context = random.choice(sample['passages']['passage_text'])
    else:
        if use_wellformed_answer and len(sample["wellFormedAnswers"]) > 0:
            new_sample["output"] = sample["wellFormedAnswers"][0]
        else:
            new_sample["output"] = sample["answers"][0]
        positive_samples_num += 1
        context = ""
        for idx, x in enumerate(is_selected):
            if x == 1:
                context += sample['passages']['passage_text'][idx]
    if normalizer:
        orig_context = context
        context = normalize(text=context, normalizer=normalizer, do_lowercase=do_lowercase)
    else:
        orig_context = None

    new_sample["sample_id"] = f"{dataset_name}_{sample['query_id']}"
    new_sample["instruction"] = sample["query"]
    new_sample["context"] = context
    if orig_context:
        new_sample["orig_context"] = orig_context

    new_samples.append(new_sample)

print("Samples dropped with multiple answers:", multi_answers_num)
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

