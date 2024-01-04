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
from nemo.export import TensorRTLLM
from tests.infer_data_path import get_infer_test_data, download_nemo_checkpoint
import torch
import shutil
import json
import argparse


def get_accuracy_with_lambada(model, prompt_embeddings_checkpoint_path, task_ids):
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
                trtllm_output = model.forward(
                    input_texts=[prompt],
                    max_output_token=1,
                    top_k=1,
                    top_p=0,
                    temperature=0.1,
                    task_ids=task_ids,
                )
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


def run_trt_llm_inference(
        model_name,
        model_type,
        prompt,
        checkpoint_path,
        trt_llm_model_dir,
        n_gpu=1,
        max_batch_size=8,
        max_input_token=128,
        max_output_token=128,
        ptuning=False,
        p_tuning_checkpoint=None,
        tp_size=None,
        pp_size=None,
        top_k=1,
        top_p=0.0,
        temperature=1.0,
        run_accuracy=True,
        debug=True,
):

    if Path(checkpoint_path).exists():
        if n_gpu > torch.cuda.device_count():
            print(
                "Path: {0} and model: {1} with {2} gpus won't be tested since available # of gpus = {3}".format(
                    model_info["checkpoint"], model_name, n_gpu, torch.cuda.device_count()
                )
            )
            return None, None

        Path(trt_llm_model_dir).mkdir(parents=True, exist_ok=True)

        if debug:
            print("")
            print("")
            print("################################################## NEW TEST ##################################################")
            print("")

            print(
                "Path: {0} and model: {1} with {2} gpus will be tested".format(
                    checkpoint_path, model_name, n_gpu
                )
            )

        prompt_embeddings_checkpoint_path = None
        task_ids = None
        max_prompt_embedding_table_size = 0

        if ptuning:
            if Path(p_tuning_checkpoint).exists():
                prompt_embeddings_checkpoint_path=p_tuning_checkpoint
                max_prompt_embedding_table_size = 8192
                task_ids = ["0"]
                if debug:
                    print("---- PTuning enabled.")
            else:
                print("---- PTuning could not be enabled and skipping the test.")
                return None, None

        trt_llm_exporter = TensorRTLLM(trt_llm_model_dir)
        trt_llm_exporter.export(
            nemo_checkpoint_path=checkpoint_path,
            model_type=model_type,
            n_gpus=n_gpu,
            tensor_parallel_size=tp_size,
            pipeline_parallel_size=pp_size,
            max_input_token=max_input_token,
            max_output_token=max_output_token,
            max_batch_size=max_batch_size,
            max_prompt_embedding_table_size=max_prompt_embedding_table_size,
        )

        if ptuning:
            trt_llm_exporter.add_prompt_table(
                task_name="0",
                prompt_embeddings_checkpoint_path=prompt_embeddings_checkpoint_path,
            )

        output = trt_llm_exporter.forward(
            input_texts=prompt,
            max_output_token=max_output_token,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            task_ids=task_ids,
        )

        if debug:
            print("")
            print("--- Prompt: ", prompt)
            print("")
            print("--- Output: ", output)
            print("")
            print("Start model accuracy testing ...")

        if run_accuracy:
            trtllm_accuracy, trtllm_accuracy_relaxed, all_trtllm_outputs, all_expected_outputs = get_accuracy_with_lambada(
                trt_llm_exporter,
                prompt_embeddings_checkpoint_path,
                task_ids,
            )

            return trtllm_accuracy, trtllm_accuracy_relaxed
        else:
            return None, None

        shutil.rmtree(trt_llm_model_dir)
    else:
        raise Exception("Checkpoint {0} could not be found.".format(checkpoint_path))


def run_existing_checkpoints(model_name, n_gpus, tp_size=None, pp_size=None, ptuning=False):

    if n_gpus > torch.cuda.device_count():
        print("Skipping the test due to not enough number of GPUs")
        return None, None

    test_data = get_infer_test_data()
    if not model_name in test_data.keys():
        raise Exception("Model {0} is not supported.".format(model_name))

    model_info = test_data[model_name]

    if n_gpus < model_info["min_gpus"]:
        print("Min n_gpus for this model is {0}".format(n_gpus))
        return None, None

    p_tuning_checkpoint = None
    if ptuning:
        if "p_tuning_checkpoint" in model_info.keys():
            p_tuning_checkpoint = model_info["p_tuning_checkpoint"]
        else:
            raise Exception("There is not ptuning checkpoint path defined.")

    return run_trt_llm_inference(
        model_name=model_name,
        model_type=model_info["model_type"],
        prompt=model_info["prompt_template"],
        checkpoint_path=model_info["checkpoint"],
        trt_llm_model_dir=model_info["trt_llm_model_dir"],
        n_gpu=n_gpus,
        max_batch_size=model_info["max_batch_size"],
        max_input_token=512,
        max_output_token=model_info["max_output_token"],
        ptuning=ptuning,
        p_tuning_checkpoint=p_tuning_checkpoint,
        tp_size=tp_size,
        pp_size=pp_size,
        top_k=1,
        top_p=0.0,
        temperature=1.0,
        run_accuracy=True,
        debug=True,
    )


def get_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=f"Deploy nemo models to Triton and benchmark the models",
    )

    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--existing_test_models",
        type=str,
        required=True,
        default="False",
    )
    parser.add_argument(
        "--model_type",
        type=str,
    )
    parser.add_argument(
        "--min_gpus",
        type=int,
        default=1,
        required=True,
    )
    parser.add_argument(
        "--max_gpus",
        type=int,
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="/tmp/nemo_checkpoint/",
    )
    parser.add_argument(
        "--trt_llm_model_dir",
        type=str,
    )
    parser.add_argument(
        "--max_batch_size",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--max_input_token",
        type=int,
        default=256,
    )
    parser.add_argument(
        "--max_output_token",
        type=int,
        default=128,
    )
    parser.add_argument(
        "--p_tuning_checkpoint",
        type=str,
    )
    parser.add_argument(
        "--ptuning",
        type=str,
        default="False",
    )
    parser.add_argument(
        "--ptuning_checkpoint_dir",
        type=str,
        default="False",
    )
    parser.add_argument(
        "--tp_size",
        type=int,
    )
    parser.add_argument(
        "--pp_size",
        type=int,
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--run_accuracy",
        type=str,
        default="True",
    )
    parser.add_argument(
        "--debug",
        type=str,
        default="True",
    )

    args = parser.parse_args()

    if args.debug == "True":
        args.debug = True
    else:
        args.debug = False

    if args.run_accuracy == "True":
        args.run_accuracy = True
    else:
        args.run_accuracy = False

    if args.ptuning == "True":
        args.ptuning = True
    else:
        args.ptuning = False

    if args.existing_test_models == "True":
        args.existing_test_models = True
    else:
        args.existing_test_models = False

    return args


def run_inference_tests(args):

    result_dic = {}

    if args.existing_test_models:
        if args.max_gpus is None:
            trtllm_accuracy, trtllm_accuracy_relaxed = run_existing_checkpoints(
                model_name=args.model_name,
                n_gpus=args.min_gpus,
                ptuning=args.ptuning,
                tp_size=args.tp_size,
                pp_size=args.pp_size,
            )
            result_dic[args.min_gpus] = (trtllm_accuracy, trtllm_accuracy_relaxed)
        else:
            n_gpus = args.min_gpus
            while n_gpus <= args.max_gpus:
                trtllm_accuracy, trtllm_accuracy_relaxed = run_existing_checkpoints(
                    model_name=args.model_name,
                    n_gpus=n_gpus,
                    ptuning=args.ptuning,
                    tp_size=args.tp_size,
                    pp_size=args.pp_size,
                )
                result_dic[n_gpus] = (trtllm_accuracy, trtllm_accuracy_relaxed)
                n_gpus = n_gpus * 2
    else:
        prompt_template=["The capital of France is", "Largest animal in the sea is"]
        n_gpus = args.min_gpus

        while n_gpus <= args.max_gpus:
            trtllm_accuracy, trtllm_accuracy_relaxed = run_trt_llm_inference(
                model_name=args.model_name,
                model_type=args.model_type,
                prompt=prompt_template,
                checkpoint_path=args.checkpoint_dir,
                trt_llm_model_dir=args.trt_llm_model_dir,
                n_gpu=n_gpus,
                max_batch_size=args.max_batch_size,
                max_input_token=args.max_input_token,
                max_output_token=args.max_output_token,
                ptuning=args.ptuning,
                p_tuning_checkpoint=args.ptuning_checkpoint_dir,
                tp_size=args.tp_size,
                pp_size=args.pp_size,
                top_k=args.top_k,
                top_p=args.top_p,
                temperature=args.temperature,
                run_accuracy=args.run_accuracy,
                debug=args.debug,
            )
            result_dic[n_gpus] = (trtllm_accuracy, trtllm_accuracy_relaxed)
            n_gpus = n_gpus * 2

    test_result = "PASS"
    print("======================================= Test Summary =======================================")
    for i, results in result_dic.items():
        if not results[0] is None and not results[1] is None:
            print("Number of GPUS: {0}, Model Accuracy: {1}, Relaxed Model Accuracy: {2}".format(i, results[0], results[1]))
            if results[1] < 0.5:
                test_result = "FAIL"

    print("=============================================================================================")
    print ("TEST: " + test_result)
    if test_result == "FAIL":
        raise Exception("Model accuracy is below 0.5")


if __name__ == '__main__':
    args = get_args()
    run_inference_tests(args)