import sys
import os
import subprocess

import hydra
import omegaconf

def create_bcp_submit_cmd(
    job_name,
    container,
    workspace_common,
    workspace_scripts,
    bignlp_path,
    bcp_script,
    instance,
    num_nodes,
    array_type="PYTORCH",
    total_runtime="10H"
):
    base_cmd = f"cd {bignlp_path}; {bcp_script}"
    if (num_nodes == 1):
        num_nodes = 2  # bcprun needs at least 2 nodes
    submit_cmd = f"ngc batch run --name \"{job_name}\" --image \"{container}\" \
    --commandline \"{base_cmd}\" --workspace {workspace_common}:/workspace-common \
    --workspace {workspace_scripts}:/workspace-scripts --result /result \
    --preempt RUNONCE --instance {instance} --replicas {num_nodes} \
    --array-type {array_type} --total-runtime {total_runtime}"
    
    return submit_cmd        

def create_bcp_file(
    bignlp_path,
    new_script_path,
    eval_cmd1,
    eval_cmd2,
    num_nodes,
    ntasks_per_node,
    log_file,
    err_file
):        
    with open(new_script_path, "w") as f:
        # Replace {bignlp_path}/bcprun2 below by bcprun once new bcprun is deployed
        f.writelines(f'{bignlp_path}/bcprun2 -n {num_nodes} --npernode 1 -c "{eval_cmd1}" >> {log_file} 2>>{err_file} \n\n')
        f.writelines(f'{bignlp_path}/bcprun2 -n {num-nodes} --npernode {ntasks_per_node} -c "{eval_cmd2}" >> {log_file} 2>>{err_file} \n\n')
        f.writelines("set +x\n") 
    os.chmod(new_script_path, 0o755)
    
def run_evaluation(cfg, dependency=None):
    # Read config
    bignlp_path = cfg.get("bignlp_path")
    container = cfg.get("container")
    eval_cfg = cfg.get("evaluation")
    run_cfg = eval_cfg.get("run")
    model_cfg = eval_cfg.get("model")

    # Model parameters
    model_type = model_cfg.get("type")
    checkpoint = model_cfg.get("checkpoint_path")
    tensor_model_parallel_size = model_cfg.get("tensor_model_parallel_size")
    batch_size = model_cfg.get("eval_batch_size")
    vocab_file = model_cfg.get("vocab_file")
    merge_file = model_cfg.get("merge_file")

    # BCP parameters
    bcp_cfg = eval_cfg.get("bcp")
    nodes = bcp_cfg.get("nodes")
    ntasks_per_node = bcp_cfg.get("ntasks_per_node")
    gpus_per_task = bcp_cfg.get("gpus_per_task")
    instance = bcp_cfg.get("instance")
    time_limit = bcp_cfg.get("time_limit")
  
    # Run parameters
    name = run_cfg.get("name")
    tasks = run_cfg.get("tasks")
    
    output_path = run_cfg.get("output_path")
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    log_dir = output_path
        
    cache_dir = os.path.join(output_path, "data_cache")
    code_path1 = os.path.join(bignlp_path, "eval_scripts/eval_harness/download.py")
    eval_cmd1 = f"python {code_path1} --tasks {tasks} --cache_dir {cache_dir} " \

    new_script_path = os.path.join(bignlp_path, f"eval_scripts/{name}.sh")
    code_path2 = os.path.join(bignlp_path, "eval_scripts/eval_harness/evaluate.py")
    eval_cmd2 = f"python -u {code_path2} " \
                f"--name {name} " \
                f"--model {model_type} " \
                f"--tasks {tasks} " \
                f"--cache_dir {cache_dir} " \
                f"--batch_size {batch_size} " \
                f"--output_path {log_dir} " \
                f"--model_args nemo_model={checkpoint},tensor_model_parallel_size={tensor_model_parallel_size},vocab_file={vocab_file},merges_file={merge_file} "

    create_bcp_file(
        bignlp_path=bignlp_path,
        new_script_path=new_script_path,
        eval_cmd1=eval_cmd1,
        eval_cmd2=eval_cmd2,
        num_nodes=nodes,
        ntasks_per_node=ntasks_per_node,
        log_file=f"{log_dir}/log.txt",
        err_file=f"{log_dir}/err.txt"
    )

    submit_cmd = create_bcp_submit_cmd(
        job_name=bcp_cfg.get("job_name"),
        container=container,
        workspace_common=bcp_cfg.get("workspace_common"),
        workspace_scripts=bcp_cfg.get("workspace_scripts"),
        bignlp_path=bignlp_path,
        bcp_script=new_script_path,
        instance=instance,
        num_nodes=nodes,
        array_type="PYTORCH",
        total_runtime=time_limit
    )
    
    print(f"\n Submit command after conversion done:\n {submit_cmd}")
    print(f"\n Script file: {new_script_path}")
