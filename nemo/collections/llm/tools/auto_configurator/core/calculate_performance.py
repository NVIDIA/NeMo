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

    print(f"=== DEBUG: get_results called ===")
    print(f"path_to_save: {path_to_save}")
    print(f"log_file_prefix: {log_file_prefix}")
    print(f"output_top_n: {output_top_n}")

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

    print(f"=== DEBUG: Model config ===")
    print(f"model_name: {model_name}")
    print(f"model_size: {model_size}")
    print(f"seq_length: {seq_length}")
    print(f"num_nodes: {num_nodes}")
    print(f"gpus_per_node: {gpus_per_node}")

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
    print(f"=== DEBUG: Scanning directory ===")
    print(f"training_logs (absolute): {training_logs}")

    error_files = find_tb_logs(training_logs, log_file_prefix)
    tb_files = find_tb_logs(training_logs, "events")
    dirs = [f.path for f in os.scandir(training_logs) if f.is_dir()]

    print(f"Found {len(error_files)} error files")
    print(f"Found {len(tb_files)} tensorboard files")
    print(f"Found {len(dirs)} directories")

    print(f"=== DEBUG: Directory contents ===")
    for i, d in enumerate(dirs):
        print(f"  Dir {i+1}: {os.path.basename(d)}")

    print(f"=== DEBUG: Error files ===")
    for i, f in enumerate(error_files):
        print(f"  Error file {i+1}: {os.path.basename(f)}")

    print(f"=== DEBUG: Tensorboard files ===")
    for i, f in enumerate(tb_files):
        print(f"  TB file {i+1}: {os.path.basename(f)}")

    print(f"=== DEBUG: Processing directories ===")
    for i, (error_file, tb_file, candidate_dir) in enumerate(zip(error_files, tb_files, dirs)):
        print(f"\n--- Processing directory {i+1}/{len(dirs)} ---")
        print(f"  Directory: {os.path.basename(candidate_dir)}")
        print(f"  Error file: {os.path.basename(error_file) if error_file else 'None'}")
        print(f"  TB file: {os.path.basename(tb_file) if tb_file else 'None'}")

        try:
            print(f"  Attempting to extract config from: {candidate_dir}")
            tp, pp, cp, ep, mbs, vp, gbs = get_config(candidate_dir)
            print(f"  SUCCESS: tp={tp}, pp={pp}, cp={cp}, ep={ep}, mbs={mbs}, vp={vp}, gbs={gbs}")
        except Exception as e:
            print(f"  FAILED to extract config: {e}")
            print(f"  Skipping {candidate_dir}")
            continue

        error = find_error(error_file) if error_file else None
        if error:
            print(f"  Found error: {error}")
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
        else:
            print(f"  No error found")

        if not tb_file:
            print(f"  No tensorboard file found, skipping")
            continue

        print(f"  Loading tensorboard file: {tb_file}")
        ea = event_accumulator.EventAccumulator(tb_file)
        ea.Reload()

        try:
            print(f"  Looking for 'train_step_timing in s' scalar")
            timing_list = ea.Scalars("train_step_timing in s")
            print(f"  Found {len(timing_list)} timing entries")

            if len(timing_list) < 10:
                print(f"  Not enough timing entries ({len(timing_list)} < 10), skipping")
                continue

            timing_list = [x.value for x in timing_list[1:]]
            avg_global_step_time = round(sum(timing_list) / len(timing_list), 2)
            samples_per_s = round(gbs / avg_global_step_time, 2)

            print(f"  Calculating TFLOPS...")
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
            print(f"  TFLOPS calculated: {m_tflops_gpu} per GPU, {m_tflops} total")

            vp_str = int(vp) if vp and str(vp).lower() != 'none' else 'None'
            descriptive_name = (
                f"{model_name}_"
                f"{model_size}b_"
                f"{num_nodes}nodes_"
                f"tp_{tp}_pp_{pp}_cp_{cp}_ep_{ep}_"
                f"mbs_{mbs}_"
                f"vp_{vp_str}_"
                f"seq_{seq_length}_"
                f"gbs_{gbs}"
            )
            print(f"  Descriptive name: {descriptive_name}")

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
            print(f"  SUCCESS: Added result #{len(result)}")

            performance_dict[descriptive_name] = {
                "m_tflops_gpu": m_tflops_gpu,
                "time_per_global_step": avg_global_step_time,
                "samples_per_s": samples_per_s,
                "m_tflops": m_tflops,
            }

        except Exception as e:
            print(f"  ERROR processing tensorboard file: {e}")
            import traceback

            traceback.print_exc()
        finally:
            print(f"  Continuing to next directory...")
            continue

    print(f"\n=== DEBUG: Final results summary ===")
    print(f"Total results collected: {len(result)}")
    print(f"Total errors collected: {len(errors)}")

    if result:
        print(f"Processing {len(result)} results...")
        result.sort(key=lambda x: x[15])
        print(f"Top {min(output_top_n, len(result))} configs sorted from fastest to slowest:")
        for i, res in enumerate(result):
            print(f"Config #{i+1} - {res[-1]}: {res[-3]} TFLOPS per GPU with {res[15]:.4f}s per global step.")
            if i + 1 == output_top_n:
                break

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
        print("=== DEBUG: This means either:")
        print("  1. No directories were found")
        print("  2. All config extractions failed")
        print("  3. All tensorboard files failed to process")
        print("  4. No timing data was found")

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

    print(f"    DEBUG get_config: Processing run_name: {run_name}")

    # Updated pattern to include gbs
    pattern = r'_(tp|pp|cp|ep|mbs|vp|gbs)_([^_]+)'
    print(f"    DEBUG get_config: Using regex pattern: {pattern}")

    # Find all matches in the input string
    matches = re.findall(pattern, run_name)
    print(f"    DEBUG get_config: Found matches: {matches}")

    # Convert matches to a dictionary
    params = {param: value for param, value in matches}
    print(f"    DEBUG get_config: Extracted params: {params}")

    # Convert string values to appropriate types
    try:
        tp = int(params.get("tp"))
        pp = int(params.get("pp"))
        cp = int(params.get("cp"))
        ep = int(params.get("ep"))
        mbs = int(params.get("mbs"))
        vp = int(params["vp"]) if params.get("vp") not in [None, 'None'] else None
        gbs = int(params.get("gbs"))

        print(f"    DEBUG get_config: SUCCESS - tp={tp}, pp={pp}, cp={cp}, ep={ep}, mbs={mbs}, vp={vp}, gbs={gbs}")

    except (ValueError, KeyError, TypeError) as e:
        print(f"    DEBUG get_config: FAILED to convert params: {e}")
        print(f"    DEBUG get_config: Available params: {list(params.keys())}")
        print(f"    DEBUG get_config: Required params: ['tp', 'pp', 'cp', 'ep', 'mbs', 'vp', 'gbs']")
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

    print(f"    DEBUG find_tb_logs: Searching in {logs_dir} for files starting with '{tb_prefix}'")

    tb_files = []
    # Walk through all directories and subdirectories
    for root, dirs, files in os.walk(logs_dir):
        for file in files:
            # Check if the file starts with the tb prefix
            if file.startswith(tb_prefix):
                absolute_path = os.path.abspath(os.path.join(root, file))
                tb_files.append(absolute_path)
                print(f"    DEBUG find_tb_logs: Found file: {file} -> {absolute_path}")

    print(f"    DEBUG find_tb_logs: Total files found: {len(tb_files)}")
    return tb_files
