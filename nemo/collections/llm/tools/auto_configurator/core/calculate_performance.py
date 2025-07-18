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

import os
import re
from typing import Optional

import pandas as pd
from tensorboard.backend.event_processing import event_accumulator

GPT_BASED_MODELS = [
    "gpt3",
    "llama",
    "qwen",
    "mixtral",
    "mistral",
    "gemma",
    "nemotron",
    "starcoder",
]


def get_results(
    base_config=None,
    train_config=None,
    path_to_save: str = None,
    output_top_n: Optional[int] = 10,
    log_file_prefix: Optional[str] = 'log-',
):
    """Generates performance results.

    Args:
        base_config (Partial): model base config.
        train_config (AutoConfigurator): Auto Configurator runner config.
        path_to_save (str): path where to save performance results.
        output_top_n (Optional[int]): Number of configs to be printed out as best configs.
        log_file_prefix: (Optional[str]): prefix of log files.
    """

    # Define needed variables
    model_name = train_config.model_type
    model_size = train_config.model_size_in_b
    seq_length = base_config.data.seq_length

    vocab_size = train_config.vocab_size
    num_nodes = train_config.num_nodes
    gpus_per_node = train_config.num_gpus

    layers = base_config.model.config.num_layers
    hs = base_config.model.config.hidden_size
    ffn_hs = base_config.model.config.ffn_hidden_size

    training_logs = path_to_save
    final_result_logs = path_to_save

    result_columns = [
        "Model Name",
        "Model Size",
        "Seq Length",
        "TP",
        "PP",
        "CP",
        "EP",
        "MBS",
        "VP",
        "Num Layers",
        "Hidden Size",
        "FFN Hidden Size",
        "GBS",
        "Nodes",
        "GPUs per Node",
        "Time per Step",
        "Samples per Second",
        "Model TFLOPS / GPU",
        "Model TFLOPS Aggregate",
        "Full Configuration Name",
    ]
    error_columns = [
        "Model Name",
        "Model Size",
        "Seq Length",
        "TP",
        "PP",
        "CP",
        "EP",
        "MBS",
        "VP",
        "Num Layers",
        "Hidden Size",
        "FFN Hidden Size",
        "GBS",
        "Nodes",
        "GPUs per Node",
        "Error Message",
    ]
    result = []
    errors = []

    performance_dict = {}

    training_logs = os.path.abspath(training_logs)
    error_files = find_tb_logs(training_logs, log_file_prefix)
    tb_files = find_tb_logs(training_logs, "events")
    dirs = [f.path for f in os.scandir(training_logs) if f.is_dir()]

    for error_file, tb_file, candidate_dir in zip(error_files, tb_files, dirs):
        try:
            tp, pp, cp, ep, mbs, vp, gbs = get_config(candidate_dir)
        except Exception as e:
            print(f"Skipping {candidate_dir}: {e}")
            continue

        error = find_error(error_file)
        if error:
            errors.append(
                [
                    model_name,
                    model_size,
                    seq_length,
                    tp,
                    pp,
                    cp,
                    ep,
                    mbs,
                    vp,
                    layers,
                    hs,
                    ffn_hs,
                    gbs,
                    num_nodes,
                    gpus_per_node,
                    error,
                ]
            )

        ea = event_accumulator.EventAccumulator(tb_file)
        ea.Reload()
        try:
            timing_list = ea.Scalars("train_step_timing in s")
            if len(timing_list) < 10:
                continue
            timing_list = [x.value for x in timing_list[1:]]
            avg_global_step_time = round(sum(timing_list) / len(timing_list), 2)
            samples_per_s = round(gbs / avg_global_step_time, 2)
            m_tflops, m_tflops_gpu = calculate_tflops(
                model_name=model_name,
                gbs=gbs,
                enc_seq_len=seq_length,
                dec_seq_len=seq_length,
                hs=hs,
                ffn_hs=ffn_hs,
                layers=layers,
                vocab=vocab_size,
                nodes=num_nodes,
                gpus_per_node=gpus_per_node,
                time_per_step=avg_global_step_time,
            )

            descriptive_name = create_descriptive_model_name(
                model_name=model_name,
                model_size=model_size,
                num_nodes=num_nodes,
                tp=tp,
                pp=pp,
                cp=cp,
                ep=ep,
                mbs=mbs,
                vp=vp,
                seq_length=seq_length,
                global_batch_size=gbs,
            )

            result.append(
                [
                    model_name,
                    model_size,
                    seq_length,
                    tp,
                    pp,
                    cp,
                    ep,
                    mbs,
                    vp,
                    layers,
                    hs,
                    ffn_hs,
                    gbs,  # Use extracted GBS
                    num_nodes,
                    gpus_per_node,
                    avg_global_step_time,
                    samples_per_s,
                    m_tflops_gpu,
                    m_tflops,
                    descriptive_name,
                ]
            )

            performance_dict[descriptive_name] = {
                "m_tflops_gpu": m_tflops_gpu,
                "time_per_global_step": avg_global_step_time,
                "samples_per_s": samples_per_s,
                "m_tflops": m_tflops,
            }

        finally:
            continue

    if result:
        result.sort(key=lambda x: x[15])
        print(f"Top {min(output_top_n, len(result))} configs sorted from fastest to slowest:")
        for i, res in enumerate(result):
            print(f"Config #{i+1} - {res[-1]}: {res[-3]} TFLOPS per GPU with {res[15]:.4f}s per global step.")
            if i + 1 == output_top_n:
                break

        top_config = (
            f"{model_name}_{model_size}b_{num_nodes}nodes_tp_"
            f"{result[0][3]}_pp_{result[0][4]}_cp_"
            f"{result[0][5]}_ep_{result[0][6]}_mbs_"
            f"{result[0][7]}_vp_{result[0][8]}"
        )
        print("\n==================================================")
        print(f"Optimal config: {result[0][-1]} with {result[0][15]:.4f}s per global step.")
        print("==================================================\n")

        # Save results as a CSV file.
        os.makedirs(final_result_logs, exist_ok=True)
        result_df = pd.DataFrame(result, columns=result_columns)
        result_df.to_csv(os.path.join(final_result_logs, f"final_summary_{num_nodes}nodes.csv"), index=False)

        error_df = pd.DataFrame(errors, columns=error_columns)
        error_df.to_csv(os.path.join(final_result_logs, f"failed_jobs_{num_nodes}nodes.csv"), index=False)
    else:
        print("No valid results found to process.")

    return performance_dict


def calculate_tflops(
    model_name,
    gbs,
    enc_seq_len,
    dec_seq_len,
    hs,
    ffn_hs,
    layers,
    vocab,
    nodes,
    gpus_per_node,
    time_per_step,
):
    """Calculates model and hardware TFLOPS for each model.

    GPT-3 Formulas:
        Model FLOPs = (24𝐵𝑠ℎ^2 + 4𝐵��^2ℎ) x (3 x num_layers) + 6𝐵𝑠ℎ
    T5/mT5 Formula:
        Model FLOPs =
    Bert Formula:
        Model FLOPs = 72BLsh^2 * ( 1 + (s/6h) + (v/12hL))
    """

    if model_name in GPT_BASED_MODELS:
        # Model FLOPS calculation
        model_flops = (
            (24 * gbs * enc_seq_len * hs * hs + 4 * gbs * enc_seq_len * enc_seq_len * hs) * (3 * layers)
            + (6 * gbs * enc_seq_len * hs * vocab)
        ) / time_per_step
        model_flops_per_gpu = model_flops / (nodes * gpus_per_node)

        model_tflops = model_flops / 1e12
        model_tflops_per_gpu = model_flops_per_gpu / 1e12

    elif model_name == "bert":
        model_flops = (
            72 * gbs * layers * enc_seq_len * hs * hs * (1 + (enc_seq_len / (6 * hs)) + (vocab / (12 * hs * layers)))
        ) / time_per_step
        model_flops_per_gpu = model_flops / (nodes * gpus_per_node)
        model_tflops = model_flops / 1e12
        model_tflops_per_gpu = model_flops_per_gpu / 1e12

    elif model_name in ["t5", "mt5"]:
        # Encoder Layer FLOPS: include self attention + MLP
        flops_self_attn_enc = 8 * gbs * enc_seq_len * hs * hs + 4 * gbs * enc_seq_len * enc_seq_len * hs
        flops_mlp_enc = 6 * gbs * enc_seq_len * hs * ffn_hs  # geglu needs two gemms for h -> ffn_h
        flops_enc_layer = flops_self_attn_enc + flops_mlp_enc

        # Decoder Layer FLOPS: inlcude self_attn + cross_attn + MLP
        flops_self_attn_dec = 8 * gbs * dec_seq_len * hs * hs + 4 * gbs * dec_seq_len * dec_seq_len * hs
        flops_cross_attn_dec = (
            4 * gbs * enc_seq_len * hs * hs
            + 4 * gbs * dec_seq_len * hs * hs
            + 4 * gbs * enc_seq_len * dec_seq_len * hs
        )
        flops_mlp_dec = 6 * gbs * dec_seq_len * hs * ffn_hs  # geglu needs two gemms for h -> ffn_h
        flops_dec_layer = flops_self_attn_dec + flops_cross_attn_dec + flops_mlp_dec

        # FLOPs of logits layer in the head
        flops_logits = 2 * gbs * dec_seq_len * hs * vocab

        # FLOPs of fprop
        flops_fprop = (flops_enc_layer + flops_dec_layer) * (layers // 2) + flops_logits

        # FLOPs of each train step (FLOPs of bprop is 2*fprop)
        model_flops = 3 * flops_fprop / time_per_step
        model_flops_per_gpu = model_flops / (nodes * gpus_per_node)
        model_tflops = model_flops / 1e12
        model_tflops_per_gpu = model_flops_per_gpu / 1e12

    return round(model_tflops, 2), round(model_tflops_per_gpu, 2)


def find_error(error_file: str, errors: list = ["CUDA out of memory"]):
    """Function that finds the error among job output.

    Args:
        errors (list): list of "popular" errors.
        error_file (str): path to the job output.

    Returns:
        str: serror message if job has been failed because of one of listed errors or None if not.
    """

    error = None
    with open(error_file, "r") as f:
        output = f.read()
    for e in errors:
        if e in output:
            error = e
    return error


def get_config(run_name: str) -> tuple:
    """Function that extract model parallelism parameters including GBS

    Args:
        run_name (str): name of the run.

    Returns:
        tuple: model parallelism parameters (tp, pp, cp, ep, mbs, vp, gbs).
    """

    # Updated pattern to include gbs
    pattern = r'_(tp|pp|cp|ep|mbs|vp|gbs)_([^_]+)'

    # Find all matches in the input string
    matches = re.findall(pattern, run_name)

    # Convert matches to a dictionary
    params = {param: value for param, value in matches}

    # Convert string values to appropriate types
    try:
        tp = int(params.get("tp"))
        pp = int(params.get("pp"))
        cp = int(params.get("cp"))
        ep = int(params.get("ep"))
        mbs = int(params.get("mbs"))
        vp = int(params["vp"]) if params.get("vp") not in [None, 'None'] else None
        gbs = int(params.get("gbs"))
    except (ValueError, KeyError, TypeError) as e:
        raise ValueError(
            f"Missing or invalid configuration parameters in '{run_name}': {e}. Expected integer values for all parallelism settings."
        )

    return (tp, pp, cp, ep, mbs, vp, gbs)


def find_tb_logs(logs_dir: str, tb_prefix: str) -> list:
    """Function that finds tensorboard logs

    Args:
        logs_dir (str): results directory.

    Returns:
        list: list of tensorboard files.
    """

    tb_files = []
    # Walk through all directories and subdirectories
    for root, dirs, files in os.walk(logs_dir):
        for file in files:
            # Check if the file starts with the tb prefix
            if file.startswith(tb_prefix):
                absolute_path = os.path.abspath(os.path.join(root, file))
                tb_files.append(absolute_path)

    return tb_files


def create_descriptive_model_name(
    model_name: str,
    model_size: float,
    num_nodes: int,
    tp: int,
    pp: int,
    cp: int,
    ep: int,
    mbs: int,
    vp: str,
    seq_length: int,
    global_batch_size: int,
) -> str:
    """
    Create a descriptive model name from configuration parameters.

    Args:
        model_name: Name of the model
        model_size: Model size in billions
        num_nodes: Number of nodes
        tp, pp, cp, ep, mbs: Parallelism parameters (integers)
        vp: Virtual pipeline parameter (string)
        seq_length: Sequence length
        global_batch_size: Global batch size

    Returns:
        str: Descriptive model name
    """
    # Handle vp parameter formatting
    try:
        vp_str = int(vp) if vp.lower() != 'none' else 'None'
    except (ValueError, AttributeError):
        vp_str = vp

    descriptive_name = (
        f"{model_name}_"
        f"{model_size}b_"
        f"{num_nodes}nodes_"
        f"tp_{tp}_pp_{pp}_cp_{cp}_ep_{ep}_"
        f"mbs_{mbs}_"
        f"vp_{vp_str}_"
        f"seq_{seq_length}_"
        f"gbs_{global_batch_size}"
    )

    return descriptive_name
