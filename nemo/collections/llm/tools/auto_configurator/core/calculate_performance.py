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

import logging
import os
import re
from dataclasses import dataclass
from typing import List, Optional

import pandas as pd
from tensorboard.backend.event_processing import event_accumulator

# Set up logging
logger = logging.getLogger(__name__)

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


@dataclass
class BaseConfigResult:
    """Base class for configuration results containing common parallelism and model parameters."""
    
    model_name: str
    model_size: int
    seq_length: int
    tp: int
    pp: int
    cp: int
    ep: int
    mbs: int
    vp: Optional[int]
    layers: int
    hidden_size: int
    ffn_hidden_size: int
    gbs: int
    num_nodes: int
    gpus_per_node: int

    @classmethod
    def from_common_params(cls, model_name: str, model_size: int, seq_length: int, 
                          tp: int, pp: int, cp: int, ep: int, mbs: int, vp: Optional[int],
                          layers: int, hidden_size: int, ffn_hidden_size: int, gbs: int,
                          num_nodes: int, gpus_per_node: int, **kwargs):
        """Create a base config result from common parameters."""
        return cls(
            model_name=model_name,
            model_size=model_size,
            seq_length=seq_length,
            tp=tp,
            pp=pp,
            cp=cp,
            ep=ep,
            mbs=mbs,
            vp=vp,
            layers=layers,
            hidden_size=hidden_size,
            ffn_hidden_size=ffn_hidden_size,
            gbs=gbs,
            num_nodes=num_nodes,
            gpus_per_node=gpus_per_node,
            **kwargs
        )

    def to_list(self) -> List:
        """Convert base configuration to list format for backward compatibility with CSV export.
        
        This method handles the common fields. Child classes should override this method
        to add their specific fields at the end.
        """
        return [
            self.model_name,
            self.model_size,
            self.seq_length,
            self.tp,
            self.pp,
            self.cp,
            self.ep,
            self.mbs,
            self.vp,
            self.layers,
            self.hidden_size,
            self.ffn_hidden_size,
            self.gbs,
            self.num_nodes,
            self.gpus_per_node,
        ]

    @classmethod
    def get_csv_columns(cls) -> List[str]:
        """Get the CSV column headers for the base configuration."""
        return [
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
        ]


@dataclass
class PerformanceResult(BaseConfigResult):
    """Structured class for performance results to replace confusing list indices."""
    
    time_per_step: float
    samples_per_second: float
    tflops_per_gpu: float
    tflops_total: float
    descriptive_name: str

    def to_list(self) -> List:
        """Convert to list format for backward compatibility with CSV export."""
        base_list = super().to_list()
        return base_list + [
            self.time_per_step,
            self.samples_per_second,
            self.tflops_per_gpu,
            self.tflops_total,
            self.descriptive_name,
        ]

    @classmethod
    def get_csv_columns(cls) -> List[str]:
        """Get the CSV column headers for performance results."""
        base_columns = BaseConfigResult.get_csv_columns()
        return base_columns + [
            "Time per Step",
            "Samples per Second",
            "Model TFLOPS / GPU",
            "Model TFLOPS Aggregate",
            "Full Configuration Name",
        ]


@dataclass
class ErrorResult(BaseConfigResult):
    """Structured class for error results."""
    
    error_message: str

    def to_list(self) -> List:
        """Convert to list format for backward compatibility with CSV export."""
        base_list = super().to_list()
        return base_list + [self.error_message]

    @classmethod
    def get_csv_columns(cls) -> List[str]:
        """Get the CSV column headers for error results."""
        base_columns = BaseConfigResult.get_csv_columns()
        return base_columns + ["Error Message"]


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

    logger.info(f"Starting performance analysis for model: {train_config.model_type}")
    logger.info(f"Logs directory: {path_to_save}")
    logger.info(f"Log file prefix: {log_file_prefix}")

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

    # Use generic column methods instead of hardcoded lists
    result_columns = PerformanceResult.get_csv_columns()
    error_columns = ErrorResult.get_csv_columns()
    
    result = []
    errors = []

    performance_dict = {}

    training_logs = os.path.abspath(training_logs)
    error_files = find_tb_logs(training_logs, log_file_prefix)
    tb_files = find_tb_logs(training_logs, "events")
    dirs = [f.path for f in os.scandir(training_logs) if f.is_dir()]

    logger.info(f"Found {len(dirs)} directories, {len(error_files)} error files, {len(tb_files)} tensorboard files")

    # Track tensorboard processing errors
    tb_errors = 0
    tb_error_types = {}

    for error_file, tb_file, candidate_dir in zip(error_files, tb_files, dirs):
        try:
            tp, pp, cp, ep, mbs, vp, gbs = get_config(candidate_dir)
        except Exception as e:
            logger.warning(f"Skipping {candidate_dir}: {e}")
            continue

        error = find_error(error_file) if error_file else None
        if error:
            errors.append(
                ErrorResult.from_common_params(
                    model_name=model_name,
                    model_size=model_size,
                    seq_length=seq_length,
                    tp=tp,
                    pp=pp,
                    cp=cp,
                    ep=ep,
                    mbs=mbs,
                    vp=vp,
                    layers=layers,
                    hidden_size=hs,
                    ffn_hidden_size=ffn_hs,
                    gbs=gbs,
                    num_nodes=num_nodes,
                    gpus_per_node=gpus_per_node,
                    error_message=error,
                )
            )

        if not tb_file:
            continue
            
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

            result.append(
                PerformanceResult.from_common_params(
                    model_name=model_name,
                    model_size=model_size,
                    seq_length=seq_length,
                    tp=tp,
                    pp=pp,
                    cp=cp,
                    ep=ep,
                    mbs=mbs,
                    vp=vp,
                    layers=layers,
                    hidden_size=hs,
                    ffn_hidden_size=ffn_hs,
                    gbs=gbs,  # Use extracted GBS
                    num_nodes=num_nodes,
                    gpus_per_node=gpus_per_node,
                    time_per_step=avg_global_step_time,
                    samples_per_second=samples_per_s,
                    tflops_per_gpu=m_tflops_gpu,
                    tflops_total=m_tflops,
                    descriptive_name=descriptive_name,
                )
            )

            performance_dict[descriptive_name] = {
                "m_tflops_gpu": m_tflops_gpu,
                "time_per_global_step": avg_global_step_time,
                "samples_per_s": samples_per_s,
                "m_tflops": m_tflops,
            }

        except Exception as e:
            tb_errors += 1
            error_type = str(e)
            tb_error_types[error_type] = tb_error_types.get(error_type, 0) + 1
            
            # Only log the first few errors to avoid spam
            if tb_errors <= 1:
                logger.warning(f"Error processing at least one tensorboard file: {e}")
        finally:
            continue

    # Log summary of tensorboard errors if any occurred
    if tb_errors > 0:
        logger.warning(f"Tensorboard processing completed with {tb_errors} errors:")
        for error_type, count in tb_error_types.items():
            logger.warning(f"  {error_type}: {count} occurrences")

    logger.info(f"Processing complete. Found {len(result)} successful results and {len(errors)} errors")

    if result:
        result.sort(key=lambda x: x.time_per_step)
        logger.info(f"Top {min(output_top_n, len(result))} configs sorted from fastest to slowest:")
        for i, res in enumerate(result):
            logger.info(f"Config #{i+1} - {res.descriptive_name}: {res.tflops_per_gpu:.4f} TFLOPS per GPU with {res.time_per_step:.4f}s per global step.")
            if i + 1 == output_top_n:
                break

        logger.info("\n==================================================")
        logger.info(f"Optimal config: {result[0].descriptive_name} with {result[0].time_per_step:.4f}s per global step.")
        logger.info("==================================================\n")

        # Save results as a CSV file.
        os.makedirs(final_result_logs, exist_ok=True)
        result_df = pd.DataFrame([r.to_list() for r in result], columns=result_columns)
        result_df.to_csv(os.path.join(final_result_logs, f"final_summary_{num_nodes}nodes.csv"), index=False)

        error_df = pd.DataFrame([e.to_list() for e in errors], columns=error_columns)
        error_df.to_csv(os.path.join(final_result_logs, f"failed_jobs_{num_nodes}nodes.csv"), index=False)
    else:
        logger.warning("No valid results found to process.")

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
