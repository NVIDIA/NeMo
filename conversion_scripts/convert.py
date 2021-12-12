import sys
import os
import subprocess
import glob

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
    cmd_str,
    num_nodes,
    ntasks_per_node,
    log_file,
    err_file,
    new_script_path
):
    with open(new_script_path, "w") as f:
        # Replace {bignlp_path}/bcprun2 below by bcprun once new bcprun is deployed
        f.writelines(f'{bignlp_path}/bcprun2 -n {num_nodes} -p {ntasks_per_node} -c "{cmd_str}" >> {log_file} 2>>{err_file} \n\n')
        f.writelines("set +x\n") 
    os.chmod(new_script_path, 0o755)
        
def convert_ckpt(cfg, hydra_args="", dependency=None):
    # Read config
    bignlp_path = cfg.get("bignlp_path")
    container = cfg.get("container")
    container_mounts = cfg.get("container_mounts")
    convert_cfg = cfg.get("conversion")
    run_cfg = convert_cfg.get("run")
    model_cfg = convert_cfg.get("model")

    # BCP parameters
    bcp_cfg = eval_cfg.get("bcp")
    nodes = bcp_cfg.get("nodes")
    ntasks_per_node = bcp_cfg.get("ntasks_per_node")
    gpus_per_task = bcp_cfg.get("gpus_per_task")
    instance = bcp_cfg.get("instance")
    time_limit = bcp_cfg.get("time_limit")
    
    # Run parameters
    name = run_cfg.get("name")
    nemo_file_name = run_cfg.get("nemo_file_name")

    output_path = run_cfg.get("output_path")
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    log_dir = output_path

    new_script_path = os.path.join(bignlp_path, f"conversion_scripts/{name}.sh")

    code_path = os.path.join(bignlp_path, "conversion_scripts/convert_ckpt.py")
    cmd_str = f"python3 -u {code_path} {hydra_args}"

    create_bcp_file(
        bignlp_path=bignlp_path,
        new_script_path=new_script_path,
        cmd_str=cmd_str,
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
    
    print(f"\n Submit command after training is done:\n {submit_cmd}")
    print(f"\n Script file: {new_script_path}")
    