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

from dataclasses import dataclass
from typing import List


@dataclass
class EOUResult:
    latency: list
    early_cutoff: list
    true_positives: int
    false_negatives: int
    false_positives: int
    num_utterances: int
    num_predictions: int


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
