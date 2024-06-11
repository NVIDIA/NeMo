# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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
#

"""
This script is used for evaluating RTL (Reasoning Temporal Localization) task.
It accepts two JSON files, one for the ground truth and another one for the predictions.
The two JSON files should have the following structure:
[
    {
    "video": "rY7eLyJF31M_6.mp4",
    "question_id": "rY7eLyJF31M_6_0",
    "question": "When is \"Apply mascara , false lashes on the lashes \" depicted in the video? Convey your answer using start and end timestamps exclusively.",
    "answer": "<0> <53> Apply mascara , false lashes on the lashes ",
    "duration": 102.002002002002
    },
  {
    "video": "rY7eLyJF31M_6.mp4",
    "question_id": "rY7eLyJF31M_6_1",
    "question": "When is \"Apply foundation on the face with a brush\" depicted in the video? Provide a response using only start and end timestamps.",
    "answer": "<56> <97> Apply foundation on the face with a brush",
    "duration": 102.002002002002
  },
]

The `answer` field should contain the start and end timestamps such as `<56>` and `<97>` of the event along with the sentence.
If not, the [0, duration] will be used as the predicted timestamps.

USAGE:
python eval_rtl.py --pred_file <path_to_predictions.json> \
    --ref_file <path_to_ground_truth.json> \
    --output_dir <path_to_output_dir> \
    --save_mid_result
"""
import json
import re
import os
import argparse
from collections import defaultdict


def iou(seg1, seg2):
    """Compute the intersection over union (IoU) between two segments.

    Args:
        seg1 (list): [start, end]
        seg2 (list): [start, end]

    Returns:
        float: IoU value
    """
    assert seg1[1] >= seg1[0] and seg2[1] >= seg2[0]
    
    x1 = max(seg1[0], seg2[0])
    x2 = min(seg1[1], seg2[1])
    inter = max(x2 - x1, 0)
    
    len1 = max(seg1[1] - seg1[0], 0)
    len2 = max(seg2[1] - seg2[0], 0)
    
    union = len1 + len2 - inter
    
    if union == 0:
        return 0.0
    else:
        return inter/union
    

def precision_func(thres):
    """calculate the precision based on the threshold.
    If the IoU value is greater than or equal to the threshold, \
    the precision is 1.0, otherwise 0.0.

    Args:
        thres (float): threshold value [0.0, 1.0]
    """
    def precision(seg1, seg2):
        return float(iou(seg1, seg2) >= thres)
    return precision


def parse_start_end_timestamps(outputs, duration, strict=False):
    timestamp_pattern = '\<(?: (?: \d* \.? \d+ ) | (?: \d+ \.? ) )\>'
    rx = re.compile(timestamp_pattern, re.VERBOSE)
    matches = list(rx.finditer(outputs))
    if strict:
        assert len(list(matches)) >= 2, "cannot find timestamps"
    elif len(list(matches)) < 2:
        return outputs, [0, duration]
    
    prev_end = 0
    sentence = ""
    timestamps = []
    for i in range(2):
        m = matches[i]
        start = m.start(0)
        end = m.end(0)
        timestamp = float(m.group(0)[1:-1])
        timestamp = min(max(timestamp, 0), duration)
        timestamps.append(timestamp)
        sentence += outputs[prev_end:start]
        prev_end = end
    sentence += outputs[prev_end:]
    sentence = sentence.strip()
    
    return sentence, [min(timestamps), max(timestamps)]


def eval(pred_file, ref_file, output_dir, save_mid_result=True):
    """Evaluate the predictions against the ground truth.

    Args:
        pred_file (str): path to the predictions JSON file
        ref_file (str): path to the ground truth JSON file
        output_dir (str): path to the output directory, 
            where the `answers.json` and `metrics.json` result will be saved.
    """
    metric_func = {
        'iou': iou,
        'precision@0.5': precision_func(0.5)
    }
    metrics = {}
    for metric in metric_func:
        metrics[metric] = defaultdict(list)
    

    with open(pred_file, 'r') as f:
        pred_data = json.load(f)
    
    with open(ref_file, 'r') as f:
        ref_data = json.load(f)
    
    assert len(pred_data) == len(ref_data)
    
    out_list = []
    for pred, ref in zip(pred_data, ref_data):
        assert pred['video'] == ref['video']
        assert pred['question_id'] == ref['question_id']
        duration = ref['duration']
        pred_answer, pred_timestamps = parse_start_end_timestamps(pred['answer'], duration, strict=True)
        ref_answer, ref_timestamps = parse_start_end_timestamps(ref['answer'], duration, strict=True)
        
        for metric in metric_func:
            metrics[metric][pred['video']].append(metric_func[metric](pred_timestamps, ref_timestamps))
        
        out_list.append({
            'video': pred['video'],
            'question_id': pred['question_id'],
            'question': pred['question'],
            'pred_answer': pred_answer,
            'ref_answer': ref_answer,
            'pred_timestamps': pred_timestamps,
            'ref_timestamps': ref_timestamps
        })
    
    # save result
    os.makedirs(output_dir, exist_ok=True)
    if save_mid_result:
        output_file = os.path.join(output_dir, 'answers.json')
        with open(output_file, 'w') as f:
            json.dump(out_list, f, indent=2)
    
    final_result = {}
    for metric in metrics:
        values = []
        for vid in metrics[metric]:
            values.extend(metrics[metric][vid])
        # get average value
        values = sum(values) / len(values)
        final_result[metric] = values
    
    
    output_file = os.path.join(output_dir, 'metrics.json')
    with open(output_file, 'w') as f:
        json.dump(final_result, f, indent=2)
        

def main():
    parser = argparse.ArgumentParser(description="Evaluate the predictions against the ground truth")
    parser.add_argument("--pred_file", help="Path to the predictions JSON file", required=True)
    parser.add_argument("--ref_file", help="Path to the ground truth JSON file", required=True)
    parser.add_argument("--output_dir", help="Path to the output directory", required=True)
    parser.add_argument("--save_mid_result", action="store_true", help="Save intermediate result")
    args = parser.parse_args()
    
    eval(args.pred_file, args.ref_file, args.output_dir, args.save_mid_result)

if __name__ == "__main__":
    main()       