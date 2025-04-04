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
from dataclasses import dataclass
from pathlib import Path
from pprint import pprint
from typing import List

import numpy as np

parser = argparse.ArgumentParser(description="Evaluate end of utterance predictions against reference labels.")
parser.add_argument(
    "-p",
    "--predictions",
    type=str,
    required=True,
    help="Path to the JSON file containing the predictions.",
)
parser.add_argument(
    "-r",
    "--references",
    type=str,
    required=True,
    help="Path to the JSON file containing the reference labels.",
)
parser.add_argument(
    "-t",
    "--threshold",
    type=float,
    default=0.5,
    help="Threshold for considering a prediction as EOU.",
)
parser.add_argument(
    "--drop_prefix",
    default="",
    type=str,
    help="Prefix to drop from the audio_filepath in the JSON file.",
)
parser.add_argument(
    "-c",
    "--collar",
    type=float,
    default=0.1,
    help="Collar time in seconds for matching predictions to references.",
)
parser.add_argument(
    "-o",
    "--output_dir",
    type=str,
    default="eou_results/",
    help="Output directory to save the evaluation results.",
)


@dataclass
class EOUResult:
    latency: list
    early_cutoff: list
    true_positives: int
    false_negatives: int
    false_positives: int
    num_utterances: int
    num_predictions: int


def load_json(file_path: str, drop_prefix: str = "") -> List[dict]:
    """Load a JSON file, then clean the audio_filepath."""
    with open(file_path, "r") as f:
        data = json.load(f)

    cleaned_data = []
    for item in data:
        audio_filepath = item["audio_filepath"]
        if drop_prefix and audio_filepath.startswith(drop_prefix):
            audio_filepath = audio_filepath[len(drop_prefix) :]
        elif audio_filepath.startswith("./"):
            audio_filepath = audio_filepath[2:]
        item["audio_filepath"] = audio_filepath

        cleaned_data.append(item)
    return cleaned_data


def evaluate_eou(prediction: List[dict], reference: List[dict], threshold: float, collar: float) -> EOUResult:
    """
    Evaluate end of utterance predictions against reference labels.
    Each item in predicition/reference is a dictionary containing:
    {
        "session_id": str,
        "start_time": float,  # start time in seconds
        "end_time": float,  # end time in seconds
        "words": str,  # transcription of the utterance
        "audio_filepath": str,  # only in prediction
        "eou_prob": float, # only in prediction, probability of EOU in range [0.1]
        "eou_pred": bool, # only in prediction
        "full_text": str, # only in prediction, which is the full transcription up to the end_time
    }

    Args:
        predictions (List[dict]): List of dictionaries containing predictions.
        references (List[dict]): List of dictionaries containing reference labels.
        threshold (float): Threshold for considering a prediction as EOU.
        collar (float): Collar time in seconds for matching predictions to references.
    """

    latency = []
    early_cutoff = []
    true_positives = 0
    false_negatives = 0
    false_positives = 0
    num_utterances = len(reference)
    num_predictions = len(prediction)

    predicted_eou = [p for p in prediction if p["eou_pred"] > threshold]
    predicted_eou = sorted(predicted_eou, key=lambda x: x["start_time"])
    reference = sorted(reference, key=lambda x: x["start_time"])

    p_idx = 0
    r_idx = 0
    for p_idx in range(len(predicted_eou)):
        p = predicted_eou[p_idx]
        p_start = p["start_time"]
        p_end = p["end_time"]

        while r_idx < len(reference) and reference[r_idx]["end_time"] < p_start:
            # Current reference ends before the current predicted utterance starts, find the next reference
            r_idx += 1

        if r_idx >= len(reference):
            # No more references to compare against
            false_positives += 1
            continue

        r = reference[r_idx]
        r_start = r["start_time"]
        r_end = r["end_time"]

        if np.abs(p_end - r_end) <= collar:
            # Correctly predicted EOU
            true_positives += 1
            latency.append(p_end - r_end)
            r_idx += 1
        elif r_start <= p_end < r_end - collar:
            # Early cutoff
            # current predicted EOU is within the current reference utterance
            false_positives += 1
            early_cutoff.append(r_end - p_end)
        elif r_end + collar < p_end:
            # Late EOU
            # Current predicted EOU is after the current reference ends
            false_negatives += 1
            latency.append(p_end - r_end)
        else:
            # p_end <= r_start
            # Current predicted EOU is before the current reference starts
            false_positives += 1

    if r_idx < len(reference):
        # There are remaining references that were not matched
        false_negatives += len(reference) - r_idx

    return EOUResult(
        latency=latency,
        early_cutoff=early_cutoff,
        true_positives=true_positives,
        false_negatives=false_negatives,
        false_positives=false_positives,
        num_utterances=num_utterances,
        num_predictions=num_predictions,
    )


def main():
    args = parser.parse_args()

    predictions = load_json(args.predictions, args.drop_prefix)
    references = load_json(args.references, args.drop_prefix)
    results = evaluate_eou(
        predictions,
        references,
        threshold=args.threshold,
        collar=args.collar,
    )

    f1_score = (
        (2 * results.true_positives / (2 * results.true_positives + results.false_negatives + results.false_positives))
        if (results.true_positives + results.false_negatives + results.false_positives) > 0
        else 0
    )

    avg_cutoffs = len(results.early_cutoff) / len(results.num_utterances) if len(results.num_utterances) > 0 else 0

    p80_latency = np.percentile(results.latency, 80) if len(results.latency) > 0 else 0
    p90_latency = np.percentile(results.latency, 90) if len(results.latency) > 0 else 0
    p95_latency = np.percentile(results.latency, 95) if len(results.latency) > 0 else 0
    p99_latency = np.percentile(results.latency, 99) if len(results.latency) > 0 else 0
    # Print the results
    print("Evaluation Results:")
    print(f"Number of utterances: {results.num_utterances}")
    print(f"Number of predictions: {results.num_predictions}")
    print(f"F1 Score: {f1_score:.4f}")
    print(f"Early cutoff rate: {avg_cutoffs:.4f}")
    print(f"P80 Latency: {p80_latency:.4f} seconds")
    print(f"P90 Latency: {p90_latency:.4f} seconds")
    print(f"P95 Latency: {p95_latency:.4f} seconds")
    print(f"P99 Latency: {p99_latency:.4f} seconds")


if __name__ == "__main__":
    main()
