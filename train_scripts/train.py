import sys
import os
import subprocess

import hydra
import omegaconf

def create_bcp_submit_cmd(
    job_name,
    container,
    workspace_common,
    workspace_results,
    bignlp_path,
    bcp_script,
    instance,
    num_nodes=1,
    ntasks_per_node=8,
    array_type="PYTORCH",
    total_runtime="10H"
):
    base_cmd = f"cd {bignlp_path}; NGC_NTASKS_PER_NODE=8 {bcp_script}"
    submit_cmd = f"ngc batch run --name \"{job_name}\" --image \"{container}\" \
    --commandline \"{base_cmd}\" --workspace {workspace_common}:/workspace-common \
    --workspace {workspace_results}:/workspace-results --result /result \
    --preempt RUNONCE --instance {instance} --replicas {num_nodes} \
    --array-type {array_type} --total-runtime {total_runtime}"
    
    print(f"\n Submit command: {submit_cmd}")
    print(f"\n Script file: {bcp_script}")

def create_bcp_file(
    bignlp_path,
    train_cmd,
    log_file,
    err_file,
    new_script_path="train_scripts/bcp_5b_script.sh"
):
    with open(new_script_path, "w+") as f:
        f.writelines(f'{bignlp_path}/bcprun2 -c "{train_cmd}" >> {log_file} 2>>{err_file} \n\n')
        f.writelines("set +x\n") 
    os.chmod(new_script_path, 0o755)

def run_training(cfg, hydra_args="", dependency=None):
    # Read config
    bignlp_path = cfg.get("bignlp_path")
    container = cfg.get("container")
    train_cfg = cfg.get("training")
    run_cfg = train_cfg.get("run")
    megatron_cfg = train_cfg.get("megatron")

    # Run parameters
    name = run_cfg.get("name")
    results_dir = run_cfg.get("results_dir")
    log_dir = run_cfg.get("log_dir")
    
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        
    new_script_path = os.path.join(bignlp_path, f"train_scripts/{name}.sh")
    code_path = os.path.join(bignlp_path, "train_scripts/pretrain_gpt.py")
    train_cmd = f"python3 -u {code_path} {hydra_args}"

    create_bcp_file(
        bignlp_path=bignlp_path,
        new_script_path=new_script_path,
        train_cmd=train_cmd,
        log_file=f"{log_dir}/log.txt",
        err_file=f"{log_dir}/err.txt"
    )
    
    # BCP submit command
    bcp_cfg = train_cfg.get("bcp")
    nodes = bcp_cfg.get("nodes")
    ntasks_per_node = bcp_cfg.get("ntasks_per_node")
    gpus_per_task = bcp_cfg.get("gpus_per_task")
    instance = bcp_cfg.get("instance")
    time_limit = bcp_cfg.get("time_limit")

    create_bcp_submit_cmd(
        job_name=bcp_cfg.get("job_name"),
        container=container,
        workspace_common=bcp_cfg.get("workspace_common"),
        workspace_results=bcp_cfg.get("workspace_results"),
        bignlp_path=bignlp_path,
        bcp_script=new_script_path,
        instance=instance,
        num_nodes=nodes,
        ntasks_per_node=ntasks_per_node,
        array_type="PYTORCH",
        total_runtime=time_limit
    )
