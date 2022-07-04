# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
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
import math
import os
import pathlib
import subprocess
import typing

from bignlp.bignlp_utils import add_container_mounts

FT_PATH = pathlib.Path("/opt/bignlp/FasterTransformer")
FT_BACKEND_PATH = pathlib.Path("/opt/bignlp/fastertransformer_backend")


def create_slurm_file(
    new_script_path,
    steps_cmds,
    job_name,
    flags="",
    dependency=None,
    time="04:00:00",
    exclusive=True,
    mem: typing.Optional[int] = None,
    overcommit=True,
    nodes=None,
    ntasks=None,
    ntasks_per_node=None,
    gpus_per_task=None,
    gpus_per_node=None,
    partition="batch",
    account=None,
    exclude=None,
):
    """
    Creates a slurm file to launch an export job.
    """
    with open(new_script_path, "w") as f:
        f.writelines("#!/usr/bin/env bash\n")
        if nodes is not None:
            f.writelines(f"#SBATCH --nodes={nodes}\n")
        if ntasks is not None:
            f.writelines(f"#SBATCH --ntasks={ntasks}\n")
        if ntasks_per_node is not None:
            f.writelines(f"#SBATCH --ntasks-per-node={ntasks_per_node}\n")
        if gpus_per_task is not None:
            f.writelines(f"#SBATCH --gpus-per-task={gpus_per_task}\n")
        if gpus_per_node is not None:
            f.writelines(f"#SBATCH --gpus-per-node={gpus_per_node}\n")
        if dependency is not None:
            dependency = dependency.strip()
            if dependency != "singleton":
                dependency = f"afterany:{dependency}"
            f.writelines(f"#SBATCH --dependency={dependency}\n")
        f.writelines(f"#SBATCH -p {partition}\n")
        if account is not None:
            f.writelines(f"#SBATCH -A {account}\n")
        f.writelines(f"#SBATCH --job-name={job_name}\n")
        if mem is not None:
            f.writelines(f"#SBATCH --mem={mem}\n")
        if exclusive:
            f.writelines("#SBATCH --exclusive\n")
        if overcommit:
            f.writelines("#SBATCH --overcommit\n")
        if exclude:
            f.writelines(f"#SBATCH --exclude={','.join(exclude)}\n")
        f.writelines(f"#SBATCH --time={time}\n\n")
        for cmd in steps_cmds:
            assert "'" not in cmd
            f.writelines(f"srun {flags} sh -c '{cmd}'\n\n")
        f.writelines("set +x\n")


def create_bcp_file(
    *,
    cmds,
    num_nodes=None,
    num_tasks=None,
    ntasks_per_node=None,
    gpus_per_task=None,
    gpus_per_node=None,
    log_file,
    new_script_path,
    env_exports=None,
):
    bcprun_args = []
    if num_nodes is not None:
        bcprun_args.extend(["-n", str(num_nodes)])
    if num_tasks is not None:
        bcprun_args.extend(["--replicas", str(num_tasks)])
    if ntasks_per_node is not None:
        bcprun_args.extend(["-p", str(ntasks_per_node)])
    if env_exports is not None:
        bcprun_args.extend(["--env", str(env_exports)])
    bcprun_args = " ".join(bcprun_args)

    with open(new_script_path, "w") as f:
        for cmd in cmds:
            f.writelines(f'bcprun {bcprun_args} -c "{cmd}" >> {log_file} 2>&1 \n')
        f.writelines("\n")
        f.writelines("set +x \n")
    os.chmod(new_script_path, 0o755)


def run_export(cfg, dependency=None):
    train_cfg = cfg.training
    model_cfg = train_cfg.model
    cluster_cfg = cfg.cluster

    accuracy_cfg = cfg.export.accuracy
    convert_cfg = cfg.export.conversion
    deploy_cfg = cfg.export.triton_deployment
    run_cfg = cfg.export.run

    checkpoint_path = convert_cfg.checkpoint_path
    triton_model_dir = run_cfg.triton_model_dir

    try:
        convert_cmds_fn = {"gpt": _get_gpt_conversion_cmds}[model_cfg.model_type]
    except KeyError:
        print(f"{model_cfg.model_type} model_type is not supported yet in export stage")
        return

    convert_cmds = convert_cmds_fn(cfg, checkpoint_path, triton_model_dir)

    results_dir = pathlib.Path(run_cfg.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)  # slurm requires to have directory where logs are saved existent

    job_base_name = run_cfg.name
    job_name = f"{job_base_name}_convert"
    new_script_path = pathlib.Path(cfg.bignlp_path) / "bignlp/export_scripts" / f"{job_name}.sh"

    cluster_type = cfg.cluster_type
    _get_submission_cmd_fn = {"bcm": _get_bcm_submission_cmd, "bcp": _get_bcp_submission_command}[cluster_type]

    submit_cmd = _get_submission_cmd_fn(
        cfg=cfg,
        dependency=dependency,
        job_name=job_name,
        cmds=convert_cmds,
        # assume that tokenizer files are in data dir and checkpoints are in base results dir
        dirs_to_mount=[cfg.bignlp_path, cfg.data_dir, cfg.base_results_dir],
        submission_script_path=new_script_path,
        ntasks=1,
        logs_dir=results_dir,
        time_limit=run_cfg.time_limit,
    )
    job_id = subprocess.check_output([submit_cmd], shell=True)
    conversion_job_id = job_id.decode("utf-8")
    print(f"Triton model store preparation job submitted with command: \n{submit_cmd}")
    print(f"Submitted Triton model store preparation script with job id: {conversion_job_id}")

    accuracy_cmds_fn = {"gpt": _get_gpt_accuracy_cmds}[model_cfg.model_type]
    accuracy_cmds = accuracy_cmds_fn(cfg)

    job_name = f"{job_base_name}_accuracy"
    new_script_path = pathlib.Path(cfg.bignlp_path) / "bignlp/export_scripts" / f"{job_name}.sh"

    gpus_required = convert_cfg.tensor_model_parallel_size * deploy_cfg.pipeline_model_parallel_size
    num_tasks = gpus_required
    num_nodes = int(math.ceil(num_tasks / accuracy_cfg.ntasks_per_node))
    submit_cmd = _get_submission_cmd_fn(
        cfg=cfg,
        dependency=conversion_job_id,
        job_name=job_name,
        cmds=accuracy_cmds,
        # assume that checkpoints are in base results dir
        dirs_to_mount=[cfg.bignlp_path, cfg.base_results_dir],
        submission_script_path=new_script_path,
        nodes=num_nodes,
        ntasks=num_tasks,
        ntasks_per_node=accuracy_cfg.ntasks_per_node,
        logs_dir=results_dir,
        time_limit=run_cfg.time_limit,
    )
    job_id = subprocess.check_output([submit_cmd], shell=True)
    accuracy_job_id = job_id.decode("utf-8")

    print(f"Accuracy for FT checkpoint job submitted with command: \n{submit_cmd}")
    print(f"Submitted accuracy for FT checkpoint script with job id: {accuracy_job_id}")

    dependency = ":".join([accuracy_job_id])

    return dependency


def _get_bcp_submission_command(
    *,
    cfg,
    dependency,
    job_name,
    cmds,
    dirs_to_mount,
    submission_script_path,
    nodes=None,
    ntasks=None,
    ntasks_per_node=None,
    logs_dir,
    time_limit,
):
    create_bcp_file(
        new_script_path=submission_script_path,
        cmds=cmds,
        num_nodes=nodes,
        num_tasks=ntasks,
        ntasks_per_node=ntasks_per_node,
        log_file=f"{logs_dir}/eval_log.txt",
    )
    submit_cmd = f"NGC_TASKS_PER_NODE={ntasks_per_node} {submission_script_path}"
    return submit_cmd


def _get_bcm_submission_cmd(
    *,
    cfg,
    dependency,
    job_name,
    cmds,
    dirs_to_mount,
    submission_script_path,
    nodes=None,
    ntasks=None,
    ntasks_per_node=None,
    gpus_per_task=None,
    gpus_per_node=None,
    logs_dir,
    time_limit,
):
    run_cfg = cfg.export.run
    cluster_cfg = cfg.cluster

    # Process container-mounts.
    mounts_str = ",".join([f"{dir_path}:{dir_path}" for dir_path in dirs_to_mount])
    mounts_str += add_container_mounts(cfg.container_mounts)
    ci_test = cfg.get("ci_test")
    container = cfg.container
    if ci_test:  # Whether this job is running in CI or not.
        flags = (
            f"--container-image {container} --container-mounts {mounts_str} "
            f"--no-container-mount-home "
            f"-o {logs_dir}/slurm_%J.log"
        )
    else:
        flags = (
            f"--container-image {container} --container-mounts {mounts_str} "
            f"--no-container-mount-home "
            f"-o {logs_dir}/{job_name}-%J.log -e {logs_dir}/{job_name}-%J.error"
        )
    if cluster_cfg.get("srun_flags"):
        flags += f" {cluster_cfg.get('srun_flags')}"
    create_slurm_file(
        new_script_path=submission_script_path,
        steps_cmds=cmds,
        job_name=f"{cluster_cfg.job_name_prefix}{job_name}",
        flags=flags,
        dependency=dependency,
        exclusive=cluster_cfg.exclusive,
        mem=None,
        overcommit=False,
        time=time_limit,
        nodes=nodes,
        ntasks=ntasks,
        ntasks_per_node=ntasks_per_node,
        gpus_per_task=gpus_per_task or cluster_cfg.gpus_per_task,
        gpus_per_node=gpus_per_node or cluster_cfg.gpus_per_node,
        partition=cluster_cfg.partition,
        account=cluster_cfg.account,
        exclude=cluster_cfg.get("exclude"),
    )
    if ci_test:
        submit_cmd = f'sbatch {submission_script_path} | tee "{logs_dir}/launcher.log" '
    else:
        submit_cmd = f"sbatch --parsable {submission_script_path}"
    return submit_cmd


def _get_gpt_conversion_cmds(cfg, checkpoint_path, triton_model_dir):
    convert_cfg = cfg.export.conversion
    deploy_cfg = cfg.export.triton_deployment
    model_cfg = cfg.training.model

    bignlp_scripts_path = pathlib.Path(cfg.bignlp_path)
    converter_path = FT_PATH / "examples/pytorch/gpt/utils/nemo_ckpt_convert.py"
    prepare_model_config_script_path = bignlp_scripts_path / "bignlp/export_scripts/prepare_triton_model_config.py"
    template_path = FT_BACKEND_PATH / "all_models/gpt/fastertransformer/config.pbtxt"

    triton_model_version_dir = f"{triton_model_dir}/1"

    # TODO: obtain parameter for --fused-qkv
    convert_cmd = (
        f"python -u {converter_path} \\\n"
        f" --in-file {checkpoint_path} \\\n"
        f" --saved-dir {triton_model_version_dir} \\\n"
        f" --infer-gpu-num {convert_cfg.tensor_model_parallel_size} \\\n"
        f" --weight-data-type {convert_cfg.weight_data_type} \\\n"
        f" --vocab-path {model_cfg.tokenizer.vocab_file} \\\n"
        f" --merges-path {model_cfg.tokenizer.merge_file} \\\n"
        f" --processes {convert_cfg.processes} \\\n"
        f" --load-checkpoints-to-cpu {int(convert_cfg.load_checkpoints_to_cpu)}"
    )
    triton_prepare_model_config_cmd = (
        f"python -u {prepare_model_config_script_path} \\\n"
        f" --template-path {template_path} \\\n"
        f" --ft-checkpoint {triton_model_version_dir}/{convert_cfg.tensor_model_parallel_size}-gpu \\\n"
        f" --config-path {triton_model_dir}/config.pbtxt \\\n"
        f" --max-batch-size {deploy_cfg.max_batch_size} \\\n"
        f" --pipeline-model-parallel-size {deploy_cfg.pipeline_model_parallel_size} \\\n"
        f" --data-type {deploy_cfg.data_type}"
    )
    if deploy_cfg.int8_mode:
        triton_prepare_model_config_cmd += " \\\n --int8-mode"
    if deploy_cfg.enable_custom_all_reduce:
        triton_prepare_model_config_cmd += " \\\n --enable-custom-all-reduce"
    return [
        (
            f"export PYTHONPATH={FT_PATH}:${{PYTHONPATH}} && \\\n"
            f"rm -rf {triton_model_dir} && \\\n" + convert_cmd + " && \\\n" + triton_prepare_model_config_cmd
        )
    ]


def _get_gpt_accuracy_cmds(cfg):
    run_cfg = cfg.export.run
    convert_cfg = cfg.export.conversion
    triton_cfg = cfg.export.triton_deployment
    accuracy_cfg = cfg.export.accuracy

    checkpoint_path = f"{run_cfg.triton_model_dir}/1/{convert_cfg.tensor_model_parallel_size}-gpu"

    lambada_script_path = FT_PATH / "examples/pytorch/gpt/lambada_task_example.py"
    update_config_script_path = FT_PATH / "examples/pytorch/gpt/utils/update_gpt_config.py"
    lib_path = FT_PATH / "build/lib/libth_parallel_gpt.so"

    create_config_ini_cmd = (
        f"mkdir -p $(dirname {accuracy_cfg.runtime_config_ini_path}) && \\\n"
        f"cp {checkpoint_path}/config.ini {accuracy_cfg.runtime_config_ini_path} && \\\n"
        f"python -u {update_config_script_path} \\\n"
        f" --model-dir {checkpoint_path} \\\n"
        f" --config-ini-path {accuracy_cfg.runtime_config_ini_path} \\\n"
        f" --pipeline-para-size {triton_cfg.pipeline_model_parallel_size} \\\n"
        # TODO: probably this parameter could be removed
        f" --tensor-para-size {convert_cfg.tensor_model_parallel_size} \\\n"
        f" --max-seq-len {accuracy_cfg.runtime.max_seq_len} \\\n"
        f" --beam-width {accuracy_cfg.runtime.beam_width} \\\n"
        f" --sampling-top-k {accuracy_cfg.runtime.sampling_top_k} \\\n"
        f" --sampling-top-p {accuracy_cfg.runtime.sampling_top_p} \\\n"
        f" --data-type {accuracy_cfg.runtime.data_type} && \\\n"
        f"mkdir -p $(dirname {accuracy_cfg.lambada_path}) && \\\n"
        f"wget https://github.com/cybertronai/bflm/raw/master/lambada_test.jsonl -O {accuracy_cfg.lambada_path}"
    )

    lambada_cmd = (
        f"python -u {lambada_script_path} \\\n"
        f" --checkpoint-path {checkpoint_path} \\\n"
        f" --lib-path {lib_path} \\\n"
        f" --config-ini-path {accuracy_cfg.runtime_config_ini_path} \\\n"
        f" --lambada-path {accuracy_cfg.lambada_path} \\\n"
        f" --output-path {accuracy_cfg.output_path} \\\n"
        f" --batch-size {accuracy_cfg.batch_size}"
    )

    # separate into 2 tasks to not start lambada cmd before configurations and data files are prepared
    # LOCAL_RANK is set with an enroot hook for Pytorch containers
    # SLURM_LOCALID is set by Slurm
    # OMPI_COMM_WORLD_LOCAL_RANK is set by mpirun
    return [
        (
            f"export PYTHONPATH={FT_PATH}:${{PYTHONPATH}} && \\\n"
            'export MY_LOCAL_RANK="${LOCAL_RANK:=${SLURM_LOCALID:=${OMPI_COMM_WORLD_LOCAL_RANK:-}}}" && \\\n'
            'if [ ${MY_LOCAL_RANK} = "0" ]; then ' + create_config_ini_cmd + "; fi"
        ),
        f"export PYTHONPATH={FT_PATH}:${{PYTHONPATH}} && \\\n" + lambada_cmd,
    ]
