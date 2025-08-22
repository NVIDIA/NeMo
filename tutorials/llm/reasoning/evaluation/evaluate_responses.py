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
import csv
import json
import re

import pandas as pd


def extract_model_answer(response):
    if not response or "Internal Server Error" in response:
        return "Internal Server Error"

    # Look for the pattern "The final answer is <letter>"
    match = re.search(r"The final answer is ([A-D])", response)
    if match:
        return match.group(1)
    return ""


def process_answers(input_file, output_file):
    # Read the JSONL file
    data = []
    with open(input_file, 'r') as f:
        for line in f:
            if line.strip():  # Skip empty lines
                data.append(json.loads(line))

    # Prepare CSV headers
    headers = [
        'Question',
        'Choice A',
        'Choice B',
        'Choice C',
        'Choice D',
        'Expected Answer',
        'Model Response',
        'Extracted Model Answer',
    ]

    # Write to CSV
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)

        # Process each question
        for question_data in data:
            question = question_data.get('question', '')
            choices = question_data.get('choices', {})
            expected_answer = question_data.get('expected_answer', '')
            model_response = question_data.get('model_response', '')

            # Extract model answer
            extracted_answer = extract_model_answer(model_response)

            # Write row
            row = [
                question,
                choices.get('A', ''),
                choices.get('B', ''),
                choices.get('C', ''),
                choices.get('D', ''),
                expected_answer,
                model_response,
                extracted_answer,
            ]
            writer.writerow(row)

    return output_file


def evaluate_results(csv_file, model_name):
    # Read the CSV file
    df = pd.read_csv(csv_file)

    # Calculate metrics
    total = len(df)
    correct = len(df[df['Extracted Model Answer'] == df['Expected Answer']])
    refusals = len(df[df['Extracted Model Answer'].str.contains('Internal Server Error', case=False, na=False)])

    # Print results
    print(f"\nModel: {model_name}")
    print(f"Total problems: {total}")
    print(f"Correct answers: {correct}")
    print(f"Refusals: {refusals}")
    print(f"Accuracy: {correct/total*100:.1f}% ({correct}/{total})")


def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Process and evaluate model responses')
    parser.add_argument(
        '--input_file', type=str, required=True, help='Path to the input JSONL file containing model responses'
    )
    parser.add_argument('--output_file', type=str, required=True, help='Path to the output CSV file')
    parser.add_argument('--model_name', type=str, required=True, help='Name of the model for reporting results')

    args = parser.parse_args()

    # Process answers and generate CSV
    print(f"Processing answers from {args.input_file}...")
    csv_file = process_answers(args.input_file, args.output_file)
    print(f"CSV file has been generated: {csv_file}")

    # Evaluate results
    print("\nEvaluating results...")
    evaluate_results(csv_file, args.model_name)


if __name__ == "__main__":
    main()
