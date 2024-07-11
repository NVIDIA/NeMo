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

import csv
import itertools
import json
import os
import random
import subprocess

from autoconfig import utils

NEMO_LAUNCHER_DEBUG = os.getenv("NEMO_LAUNCHER_DEBUG", "False").lower() in (
    "true",
    "t",
    "1",
)


def nodes_necessary(gpus_per_node, tp, pp):
    if tp > gpus_per_node:
        if tp % gpus_per_node != 0:
            return 0
        else:
            return max(pp, pp * tp // gpus_per_node)
    else:
        return pp


def get_vocabulary_size(base_cfg, cfg):
    vocab_path_cfg = base_cfg["model"]["tokenizer"]["vocab_file"]
    vocab_path = vocab_path_cfg.format(data_dir=cfg.data_dir)[1:]
    try:
        with open(vocab_path) as f:
            data = json.load(f)
            vocabulary_size = len(data)
            print(f"Vocabulary loaded from {vocab_path} with size {vocabulary_size}")
            divider = base_cfg["model"]["make_vocab_size_divisible_by"]
            if divider > 1:
                new_vocabulary_size = divider * (vocabulary_size // divider + 1)
                if new_vocabulary_size != vocabulary_size:
                    print(
                        f"make_vocab_size_divisible_by set so vocabulary rounded "
                        f"to {new_vocabulary_size}"
                    )
                    return new_vocabulary_size
                else:
                    return vocabulary_size
            else:
                return vocabulary_size
    except IOError as io:
        print("Vocabulary open error", io)
        print("FAILED TO LOAD VOCABULARY FOR TOKENIZER - set to default 51200")
        return 51200


def filter_configuration(base_cfg, cfg, tp, pp, gpus_per_node):
    attention_heads = base_cfg["model"]["num_attention_heads"]
    num_layers = base_cfg["model"]["num_layers"]
    if attention_heads % tp != 0:
        print(
            f"FasterTransformer invalid configuration "
            f"TENSOR_PARALLEL={tp} "
            f"PIPELINE_PARALLEL={pp} ignored due to "
            f"base_cfg[model][num_attention_heads]={attention_heads}."
        )
        return False
    elif num_layers % pp != 0:
        print(
            f"FasterTransformer invalid configuration "
            f"TENSOR_PARALLEL={tp} "
            f"PIPELINE_PARALLEL={pp} ignored due to "
            f"base_cfg[model][num_layers]={num_layers}."
        )
        return False
    elif pp == 1 or (pp > 1 and tp >= gpus_per_node):
        return nodes_necessary(gpus_per_node, tp, pp) > 0
    print(
        f"FasterTransformer partial node configuration "
        f"TENSOR_PARALLEL={tp} "
        f"PIPELINE_PARALLEL={pp} ignored."
    )
    return False


def configure_fastertransformer(base_cfg, cfg, tp, pp, bs, destination):
    max_seq_len = (
        cfg.search_config.inference_settings.benchmark.input_len
        + cfg.search_config.inference_settings.benchmark.output_len
    )
    inter_size = base_cfg["model"]["hidden_size"] * 4
    size_per_head = (
        base_cfg["model"]["hidden_size"] // base_cfg["model"]["num_attention_heads"]
    )
    vocabulary_size = get_vocabulary_size(base_cfg, cfg)

    command = [
        f"python3",
        f"{cfg.fastertransformer_path}/examples/pytorch/gpt/utils/generate_gpt_config.py",
        f"--max_seq_len {max_seq_len}",
        f"--beam_width {cfg.search_config.inference_settings.benchmark.beam_width}",
        f"--head_num {base_cfg['model']['num_attention_heads']}",
        f"--size_per_head {size_per_head}",
        f"--inter_size {inter_size}",
        f"--num_layer {base_cfg['model']['num_layers']}",
        f"--vocab_size {vocabulary_size}",
        f"--data_type {cfg.search_config.inference_settings.run.data_type}",
        f"-topk {cfg.search_config.inference_settings.benchmark.topk}",
        f"-topp {cfg.search_config.inference_settings.benchmark.topp}",
        f"--tensor_para_size {tp}",
        f"--pipeline_para_size {pp}",
        f"--request_batch_size {bs}",
        f"--request_output_len {cfg.search_config.inference_settings.benchmark.output_len}",
        f"--destination {destination}",
    ]
    print(f"Generated config for FasterTransformer to: {destination} ")
    result = os.system(" ".join(command))
    if result != 0:
        raise Exception("generate_gpt_config.py failed")


def generate_start_ids(base_cfg, cfg, bs, destination):
    command = [
        f"python3",
        f"{cfg.fastertransformer_path}/examples/pytorch/gpt/utils/generate_start_ids.py",
        f"-max_batch_size {bs}",
        f"-max_input_length {cfg.search_config.inference_settings.benchmark.input_len}",
        f"--destination {destination}",
    ]
    print(f"Generated start_ids for FasterTransformer to: {destination}")
    result = os.system(" ".join(command))
    if result != 0:
        raise Exception("generate_start_ids.py failed")


def generate_submission(
    base_cfg,
    cfg,
    job_name,
    nodes,
    tasks_per_node,
    ini,
    csv,
    submission_file,
    mounts_str,
):
    cluster_job_name = f"{cfg.cluster.job_name_prefix}{job_name}"
    gpus_per_task = cfg.cluster.gpus_per_task
    gpus_per_node = cfg.cluster.gpus_per_node
    path_list = submission_file.split("/")
    path_list[-1] = "log_job_%j.out"
    output = "/".join(path_list)
    path_list[-1] = "log_job_%j.err"
    error = "/".join(path_list)

    bash_commands = [
        f"export NCCL_LAUNCH_MODE=GROUP",
        "echo ${SLURM_PROCID}.${SLURM_LOCALID}@$(hostname)",
        f"/opt/FasterTransformer/build/bin/multi_gpu_gpt_example {ini} {csv}",
    ]
    bash_command = [" && ".join(bash_commands)]

    flags = [
        "--mpi pmix",
        f"--output {output}",
        f"--error {error}",
        f"--container-image {cfg.training_container}",
        f"--container-mounts {mounts_str}",
        f"--unbuffered",
    ]
    flags_str = " ".join(flags)

    utils.create_slurm_file(
        new_script_path=submission_file,
        cmds=bash_command,
        job_name=cluster_job_name,
        flags=flags_str,
        time=cfg.search_config.inference_settings.run.time_limit,
        nodes=nodes,
        ntasks_per_node=tasks_per_node,
        gpus_per_task=gpus_per_task,
        gpus_per_node=gpus_per_node,
        partition=cfg.cluster.partition,
        account=cfg.cluster.account,
        output=output,
        comment=f"'FasterTransformer {job_name}'",
    )


def submit_job(submission_file, results_dir):
    if os.getenv("NEMO_LAUNCHER_CI"):
        job_id = subprocess.check_output(
            [f'sbatch {submission_file} | tee "{results_dir}/../launcher.log" '],
            shell=True,
        )
    else:
        if not NEMO_LAUNCHER_DEBUG:
            job_id = subprocess.check_output(
                [f"sbatch --parsable {submission_file}"], shell=True
            )
        else:
            job_id = str(random.randint(10000, 99999)).encode("utf-8")
    dependency = job_id.decode("utf-8").split()[-1]
    return dependency


def search_inference_config(base_cfg, cfg):
    """
    Main function to launch a inference sweep job, with the config given in cfg.
    """
    # Prepare global folders
    inference_results_dir = os.path.join(
        cfg.search_config.inference_settings.run.results_dir, "inference"
    )
    os.makedirs(inference_results_dir, exist_ok=True)

    # Process container-mounts.
    auto_configurator_path = cfg.get("auto_configurator_path")
    base_results_dir = cfg.get("base_results_dir")
    container_mounts = cfg.get("container_mounts")
    mounts_str = f"{auto_configurator_path}:{auto_configurator_path},{base_results_dir}:{base_results_dir}"
    mounts_str += utils.add_container_mounts(container_mounts)

    assert (
        cfg.search_config.inference_settings.run.model_type == "gpt3"
    ), "Only GPT-3 models are currently supported for the inference HP search."
    cluster_gpus_per_task = cfg.cluster.gpus_per_task
    cluster_gpus_per_node = cfg.cluster.gpus_per_node

    all_configurations = itertools.product(
        cfg.search_config.inference_settings.run.tensor_parallel_sizes,
        cfg.search_config.inference_settings.run.pipeline_parallel_sizes,
        cfg.search_config.inference_settings.benchmark.batch_sizes,
    )

    gpus_per_node = cfg.search_config.inference_settings.run.gpus_per_node

    configurations = list(
        [
            (tp, pp, bs)
            for tp, pp, bs in all_configurations
            if filter_configuration(base_cfg, cfg, tp, pp, gpus_per_node)
        ]
    )

    if len(configurations) == 0:
        print("ALL FasterTransformer CONFIGURATIONS NOT VALID FOR BENCHMARK")
        return

    job_ids = []

    for tp, pp, bs in configurations:
        benchmark_model_name = f"{cfg.search_config.inference_settings.run.model_train_name}_tp{tp}_pp{pp}_bs{bs}"
        model_dir = os.path.join(inference_results_dir, benchmark_model_name)
        os.makedirs(model_dir, exist_ok=True)

        # Generate .ini file for FasterTransformer.
        config_ini_file = os.path.join(model_dir, "config.ini")
        configure_fastertransformer(base_cfg, cfg, tp, pp, bs, config_ini_file)

        # Generate start ids for this model.
        config_start_ids_file = os.path.join(model_dir, "start_ids.csv")
        generate_start_ids(base_cfg, cfg, bs, config_start_ids_file)

        # Generate the submission Slurm job.
        submission_file = os.path.join(model_dir, "submission_script.sh")
        job_name = f"benchmark_FT_{benchmark_model_name}"
        num_nodes = nodes_necessary(gpus_per_node, tp, pp)
        tasks_per_node = min(gpus_per_node, tp)
        generate_submission(
            base_cfg=base_cfg,
            cfg=cfg,
            job_name=job_name,
            nodes=num_nodes,
            tasks_per_node=tasks_per_node,
            ini=config_ini_file,
            csv=config_start_ids_file,
            submission_file=submission_file,
            mounts_str=mounts_str,
        )
        dependency = submit_job(submission_file, inference_results_dir)
        job_ids.append(dependency)
        print()

    # Prepare final job config files.
    results_dir = os.path.join(inference_results_dir, "final_summary")
    os.makedirs(results_dir, exist_ok=True)
    cfg_fields = ["TP", "PP", "BS"]
    configurations_file_name = os.path.join(results_dir, "inference_sweep_configs.csv")
    with open(configurations_file_name, "w") as configs_file:
        cfg_writer = csv.writer(configs_file)
        cfg_writer.writerow(cfg_fields)
        cfg_writer.writerows(configurations)

    # Prepare final summary job.
    dependency_string = ":".join(job_ids)
    summary_submission_file = os.path.join(results_dir, "job_submission.sh")
    summary_job_output = os.path.join(results_dir, f"log_final_summary_job_%j.out")
    summary_job_error = os.path.join(results_dir, f"log_final_summary_job_%j.err")
    summary_job_result = os.path.join(results_dir, "final_output.csv")
    summary_name = (
        f"{cfg.search_config.inference_settings.run.model_train_name}_summary"
    )
    summary_job_name = f"{cfg.cluster.job_name_prefix}{summary_name}_job"
    summary_script_path = (
        f"{cfg.auto_configurator_path}/autoconfig/inference_summary.py"
    )

    summary_command_elem = [
        f"python3 {summary_script_path}",
        f"--model-prefix {cfg.search_config.inference_settings.run.model_train_name}",
        f"--configs-csv {configurations_file_name}",
        f"--workspace {inference_results_dir}",
        f"--output {summary_job_result}",
    ]
    echo_command_elem = f"cat {summary_job_result}"
    bash_command = [" ".join(summary_command_elem) + " && " + echo_command_elem]

    summary_flags = [
        f"--output {summary_job_output}",
        f"--error {summary_job_error}",
        f"--container-image {cfg.training_container}",
        f"--container-mounts {mounts_str}",
        f"--unbuffered",
    ]
    summary_flags_str = " ".join(summary_flags)

    utils.create_slurm_file(
        new_script_path=summary_submission_file,
        cmds=bash_command,
        job_name=summary_job_name,
        flags=summary_flags_str,
        time=cfg.search_config.inference_settings.run.time_limit,
        nodes=1,
        ntasks_per_node=1,
        gpus_per_task=cluster_gpus_per_task,
        gpus_per_node=cluster_gpus_per_node,
        partition=cfg.cluster.partition,
        account=cfg.cluster.account,
        output=summary_job_output,
        comment=f"'FasterTransformer {summary_job_name}'",
        dependency=dependency_string,
    )
    submit_job(summary_submission_file, inference_results_dir)
    print("Submitted job to generate the final summary.")
