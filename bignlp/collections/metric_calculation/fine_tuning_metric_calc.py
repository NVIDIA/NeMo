# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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
import json
import argparse
import tempfile
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_file", type=str, required=True, help="json file with preds, inputs and targets.")
    parser.add_argument("--target_file", type=str, required=True, help="jsonl file that contains the squad dev set with multiple correct answers.")
    parser.add_argument("--squad_eval_script_path", type=str, required=True, help="path to the squad evaluation script.")
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as tmp:
        json_outputs = json.load(open(args.pred_file))
        with open(f"{tmp}/preds.text", "w") as f:
            for pred in json_outputs['preds']:
                pred = pred.strip().replace('\n', ' ')
                f.write(pred + "\n")
        os.system(f"python {args.squad_eval_script_path} --ground-truth {args.target_file} --preds {tmp}/preds.text")