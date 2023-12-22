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

from pathlib import Path
import argparse
from nemo.export import TensorRTLLM
from tests.infer_data_path import download_nemo_checkpoint
import torch
import shutil
import json

def get_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=f"Deploy nemo models to Triton and benchmark the models",
    )
    parser.add_argument(
        "--model_name",
        type=str,
    )
    parser.add_argument(
        "--model_type",
        type=str,
    )
    parser.add_argument(
        "--min_gpus",
        type=int,
    )
    parser.add_argument(
        "--max_gpus",
        type=int,
    )
    parser.add_argument(
        "--location",
        type=str,
    )
    parser.add_argument(
        "--trt_llm_model_dir",
        type=str,
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="/tmp/nemo_checkpoint/",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
    )
    parser.add_argument(
        "--checkpoint_link",
        type=str,
    )
    parser.add_argument(
        "--prompt_template",
        type=str,
    )
    parser.add_argument(
        "--max_output_token",
        type=int,
    )
    parser.add_argument(
        "--max_batch_size",
        type=int,
    )
    parser.add_argument(
        "--p_tuning_checkpoint",
        type=str,
    )
    args = parser.parse_args()

    return args


def get_accuracy_with_lambada(model):
        # lambada dataset based accuracy test, which includes more than 5000 sentences.
        # Use generated last token with original text's last token for accuracy comparison.
        # If the generated last token start with the original token, trtllm_correct make an increment.
        # It generates a CSV file for text comparison detail.

        trtllm_correct = 0
        trtllm_correct_relaxed = 0
        all_expected_outputs = []
        all_trtllm_outputs = []

        with open('/opt/NeMo/tests/deploy/lambada.json', 'r') as file:
            records = json.load(file)

            for record in records:
                prompt = record["text_before_last_word"]
                expected_output = record["last_word"].strip().lower()
                trtllm_output = model.forward(input_texts=[prompt], max_output_token=1, top_k=1, top_p=0, temperature=0.1)
                trtllm_output = trtllm_output[0][0].strip().lower()

                all_expected_outputs.append(expected_output)
                all_trtllm_outputs.append(trtllm_output)

                if expected_output == trtllm_output:
                    trtllm_correct += 1

                if expected_output == trtllm_output or trtllm_output.startswith(expected_output) or expected_output.startswith(trtllm_output):
                    if len(trtllm_output) == 1 and len(expected_output) > 1:
                        continue
                    trtllm_correct_relaxed += 1

                # print("-- expected_output: {0} and trtllm_output: {1}".format(expected_output, trtllm_output))
                
        trtllm_accuracy = trtllm_correct / len(all_expected_outputs)
        trtllm_accuracy_relaxed = trtllm_correct_relaxed / len(all_expected_outputs)
        return trtllm_accuracy, trtllm_accuracy_relaxed, all_trtllm_outputs, all_expected_outputs


def run_trt_llm_export(args, prompt_template, expected_keyword, n_gpus, tp_size=None, pp_size=None):

    if args.location == "HF":
        download_nemo_checkpoint(
            args.checkpoint_link, args.checkpoint_dir, args.checkpoint
        )

    print(
        "Path: {0} and model: {1} is next and test will start if the nemo checkpoint exists ...".format(
            args.checkpoint, args.model_name
        )
    )
    if Path(args.checkpoint).exists():
        if n_gpus > torch.cuda.device_count():
            print(
                "Path: {0} and model: {1} with {2} gpus won't be tested since available # of gpus = {3}".format(
                    args.checkpoint, args.model_name, n_gpus, torch.cuda.device_count()
                )
            )
            return

        Path(args.trt_llm_model_dir).mkdir(parents=True, exist_ok=True)

        print(
            "Path: {0} and model: {1} with {2} gpus will be tested".format(
                args.checkpoint, args.model_name, n_gpus
            )
        )

        prompt_embeddings_checkpoint_path = None
        if args.p_tuning_checkpoint:
            print("---- PTuning enabled.")
            prompt_embeddings_checkpoint_path=args.p_tuning_checkpoint

        trt_llm_exporter = TensorRTLLM(model_dir=args.trt_llm_model_dir)
        trt_llm_exporter.export(
            nemo_checkpoint_path=args.checkpoint,
            model_type=args.model_type,
            n_gpus=n_gpus,
            max_input_token=1024,
            max_output_token=128,
            max_batch_size=args.max_batch_size,
        )
        output = trt_llm_exporter.forward(
            input_texts=prompt_template,
            max_output_token=args.max_output_token,
            top_k=1,
            top_p=0.0,
            temperature=1.0,
            prompt_embeddings_checkpoint_path=prompt_embeddings_checkpoint_path,
        )

        print("")
        print("--- Prompt: ", prompt_template)
        print("")
        print("--- Output: ", output)
        print("")
        
        print("Start model accuracy testing ...")
        trtllm_accuracy, trtllm_accuracy_relaxed, all_trtllm_outputs, all_expected_outputs = get_accuracy_with_lambada(trt_llm_exporter)

        trt_llm_exporter = None
        shutil.rmtree(args.trt_llm_model_dir)

        return trtllm_accuracy, trtllm_accuracy_relaxed


def main(args):
    prompt_template=["The capital of France is", "Largest animal in the sea is"]
    expected_keyword=["Paris", "Whale", "Cheetah"]

    n_gpus = args.min_gpus
    result_dic = {}
    while n_gpus <= args.max_gpus:
        trtllm_accuracy, trtllm_accuracy_relaxed = run_trt_llm_export(args, prompt_template, expected_keyword, n_gpus)
        result_dic[n_gpus] = (trtllm_accuracy, trtllm_accuracy_relaxed)
        n_gpus = n_gpus * 2

    test_result = "PASS"
    print("======================================= Test Summary =======================================")
    for i, results in result_dic.items():
        print("Number of GPUS: {0}, Model Accuracy: {1}, Relaxed Model Accuracy: {2}".format(i, results[0], results[1]))
        if results[1] < 0.5:
            test_result = "FAIL"
    print("=============================================================================================")
    print ("TEST: " + test_result)
    if test_result == "FAIL":
        raise Exception("Model accuracy is below 0.5")


if __name__ == '__main__':
    args = get_args()
    main(args)
