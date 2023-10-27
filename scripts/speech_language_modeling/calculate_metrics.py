# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
caculate different metrics for SpeechLM
from a given json file with pred and label
"""

from nemo.collections.asr.parts.utils.manifest_utils import read_manifest

from nemo.collections.common.metrics.classification_accuracy import ExactStringMatchMetric, TokenF1Score
from torchmetrics.text.wer import WordErrorRate


task2metric= {
    "asr": WordErrorRate,
    "speaker_verification": ExactStringMatchMetric,
    "speaker_attributed_asr": WordErrorRate,
}

LANGUAGES=[
    "English",
    "Spanish",
]

TASKS = ['asr', 'speaker_verification', 'speaker_attributed_asr']


pred_label_manifest_path='/home/kpuvvada/Downloads/val_predictions_validation_dev-blend_inputs_preds_labels.jsonl'


def get_speaker_specific_transcript(transcript):
    transcript = transcript.split()
    speaker2words = {}
    for i in range(len(transcript)):
        if transcript[i].startswith("<") and transcript[i].endswith(">"):
            speaker = transcript[i].strip("<>")
            if speaker not in speaker2words:
                speaker2words[speaker] = []
            speaker2words[speaker].append(transcript[i+1])

    for speaker in speaker2words:
        speaker2words[speaker] = " ".join(speaker2words[speaker])

    return speaker2words


def get_speaker_specific_preds_labels(pred, label):
    """
    sentences are of the form <speaker_name> word <speaker_name> word...
    accumulate words for each speaker
    """
    pred = get_speaker_specific_transcript(pred)
    label = get_speaker_specific_transcript(label)

    if not set(pred.keys()) == set(label.keys()):
        print(f"pred and label doesn't have same speakers \n pred: {pred.keys()} \n label: {label.keys()}")
        return None, None

    common_speakers = set(pred.keys()).intersection(set(label.keys()))

    pred_list = []
    label_list = []
    for speaker in common_speakers:
        pred_list.append(pred[speaker])
        label_list.append(label[speaker])

    return pred_list, label_list

    


def calculate_metric(manifest_items, task, language=None):
    metric = task2metric[task]()
    num_examples = 0
    if task == 'asr':
        if language is None:
            raise ValueError("language is required for asr task")
        
    for item in manifest_items:
        if not item['taskname'] == task:
            continue

        if task == 'asr':
            if not language.lower() in item['input'].lower().split():
                continue
        
        pred = item['pred']
        label = item['label']

        if task == 'speaker_attributed_asr':
            # get speaker specific sentences
            pred, label = get_speaker_specific_preds_labels(pred, label)
            if pred is None:
                continue

        #print(f"task: {task}, language: {language}, \n pred: {pred},\n label: {label}")
        #print("---------------------------------------------------")
        metric.update(pred, label)
        num_examples += 1

    metric_value = metric.compute()
    print(f"task: {task}, language: {language}, metric_value: {metric_value}, num_examples: {num_examples}")


if __name__=="__main__":
    manifest_items = read_manifest(pred_label_manifest_path)
    for task in TASKS:
        if task == 'asr':
            for language in LANGUAGES:
                calculate_metric(manifest_items, task, language)
        else:
            calculate_metric(manifest_items, task)








