# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

import argparse
import json
import random
from typing import Any, Dict, List, Tuple

from datasets import load_dataset

random.seed(42)


def process_mmlu_dataset() -> List[Dict[str, Any]]:
    """Process the MMLU dataset and return a list of formatted entries."""
    dataset = load_dataset("cais/mmlu", "all")
    output_data = []
    skipped = 0

    for split in dataset.keys():
        for item in dataset[split]:
            question = item['question']
            choices = item['choices']
            answer = item['answer']

            if not question or not choices:
                skipped += 1
                continue

            # Create a list of (choice, is_correct) tuples
            answers = []
            for i, choice in enumerate(choices):
                is_correct = chr(65 + i) == chr(65 + answer)
                answers.append((choice, is_correct))

            # Shuffle the answers
            random.shuffle(answers)

            # Create the choices dictionary
            choices_dict = {}
            correct_letter = None

            for i, (answer_text, is_correct) in enumerate(answers):
                letter = chr(65 + i)
                choices_dict[f'Choice {i+1}'] = answer_text
                if is_correct:
                    correct_letter = letter

            entry = {
                'Question': question,
                'Choice 1': choices_dict['Choice 1'],
                'Choice 2': choices_dict['Choice 2'],
                'Choice 3': choices_dict['Choice 3'],
                'Choice 4': choices_dict['Choice 4'],
                'Answer': correct_letter,
                'Subject': item['subject'],
            }

            output_data.append(entry)

    return output_data, skipped


def process_gpqa_dataset() -> List[Dict[str, Any]]:
    """Process the GPQA dataset and return a list of formatted entries."""
    dataset = load_dataset("Idavidrein/gpqa", "gpqa_main")
    output_data = []
    skipped = 0

    for split in dataset.keys():
        for item in dataset[split]:
            question = item['Question']
            correct_answer = item['Correct Answer']
            incorrect_1 = item['Incorrect Answer 1']
            incorrect_2 = item['Incorrect Answer 2']
            incorrect_3 = item['Incorrect Answer 3']

            if not question or not correct_answer or not incorrect_1 or not incorrect_2 or not incorrect_3:
                skipped += 1
                continue

            answers = [(correct_answer, True), (incorrect_1, False), (incorrect_2, False), (incorrect_3, False)]

            random.shuffle(answers)

            choices = {}
            correct_letter = None

            for i, (answer_text, is_correct) in enumerate(answers):
                letter = chr(65 + i)
                choices[f'Choice {i+1}'] = answer_text
                if is_correct:
                    correct_letter = letter

            entry = {
                'Question': question,
                'Choice 1': choices['Choice 1'],
                'Choice 2': choices['Choice 2'],
                'Choice 3': choices['Choice 3'],
                'Choice 4': choices['Choice 4'],
                'Answer': correct_letter,
            }

            output_data.append(entry)

    return output_data, skipped


def process_gpqa_diamond_dataset() -> List[Dict[str, Any]]:
    """Process the GPQA Diamond dataset and return a list of formatted entries."""
    dataset = load_dataset("Idavidrein/gpqa", "gpqa_diamond")
    output_data = []
    skipped = 0

    for split in dataset.keys():
        for item in dataset[split]:
            question = item['Question']
            correct_answer = item['Correct Answer']
            incorrect_1 = item['Incorrect Answer 1']
            incorrect_2 = item['Incorrect Answer 2']
            incorrect_3 = item['Incorrect Answer 3']

            if not question or not correct_answer or not incorrect_1 or not incorrect_2 or not incorrect_3:
                skipped += 1
                continue

            answers = [(correct_answer, True), (incorrect_1, False), (incorrect_2, False), (incorrect_3, False)]

            random.shuffle(answers)

            choices = {}
            correct_letter = None

            for i, (answer_text, is_correct) in enumerate(answers):
                letter = chr(65 + i)
                choices[f'Choice {i+1}'] = answer_text
                if is_correct:
                    correct_letter = letter

            entry = {
                'Question': question,
                'Choice 1': choices['Choice 1'],
                'Choice 2': choices['Choice 2'],
                'Choice 3': choices['Choice 3'],
                'Choice 4': choices['Choice 4'],
                'Answer': correct_letter,
            }

            output_data.append(entry)

    return output_data, skipped


def write_to_jsonl(data: List[Dict[str, Any]], filename: str) -> None:
    """Write the processed data to a JSONL file."""
    with open(filename, 'w') as f:
        for entry in data:
            f.write(json.dumps(entry) + '\n')


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Convert datasets to JSONL format')
    parser.add_argument(
        '--datasets',
        nargs='+',
        choices=['mmlu', 'gpqa', 'gpqa_diamond', 'all'],
        default=['all'],
        help='Datasets to process (default: all)',
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # If 'all' is selected, process all datasets
    if 'all' in args.datasets:
        args.datasets = ['mmlu', 'gpqa', 'gpqa_diamond']

    # Process selected datasets
    if 'mmlu' in args.datasets:
        mmlu_data, mmlu_skipped = process_mmlu_dataset()
        write_to_jsonl(mmlu_data, 'mmlu_dataset.jsonl')
        print(f"Converted {len(mmlu_data)} questions to mmlu_dataset.jsonl")
        if mmlu_skipped > 0:
            print(f"Skipped {mmlu_skipped} MMLU entries due to missing data")
        print("\nMMLU Sample entry:")
        print(json.dumps(mmlu_data[0], indent=2))

    if 'gpqa' in args.datasets:
        gpqa_data, gpqa_skipped = process_gpqa_dataset()
        write_to_jsonl(gpqa_data, 'gpqa_dataset.jsonl')
        print(f"\nConverted {len(gpqa_data)} questions to gpqa_dataset.jsonl")
        if gpqa_skipped > 0:
            print(f"Skipped {gpqa_skipped} GPQA entries due to missing data")
        print("\nGPQA Sample entry:")
        print(json.dumps(gpqa_data[0], indent=2))

    if 'gpqa_diamond' in args.datasets:
        gpqa_diamond_data, gpqa_diamond_skipped = process_gpqa_diamond_dataset()
        write_to_jsonl(gpqa_diamond_data, 'gpqa_diamond_dataset.jsonl')
        print(f"\nConverted {len(gpqa_diamond_data)} questions to gpqa_diamond_dataset.jsonl")
        if gpqa_diamond_skipped > 0:
            print(f"Skipped {gpqa_diamond_skipped} GPQA Diamond entries due to missing data")
        print("\nGPQA Diamond Sample entry:")
        print(json.dumps(gpqa_diamond_data[0], indent=2))


if __name__ == "__main__":
    main()
