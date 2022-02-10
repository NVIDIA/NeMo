import sys
import os
import subprocess

import hydra
import omegaconf


def create_slurm_file(
        new_script_path,
        convert_cmd,
        job_name,
        flags="",
        dependency=None,
        time="04:00:00",
        exclusive=True,
        mem=0,
        overcommit=True,
        nodes=1,
        ntasks_per_node=1,
        gpus_per_task=1,
        partition="batch",
        account=None,
):
    with open(new_script_path, "w") as f:
        f.writelines("#!/bin/bash\n")
        f.writelines(f"#SBATCH --nodes={nodes}\n")
        f.writelines(f"#SBATCH --ntasks-per-node={ntasks_per_node}\n")
        if gpus_per_task is not None:
            f.writelines(f"#SBATCH --gpus-per-task={gpus_per_task}\n")
        if dependency is not None:
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
        f.writelines(f"#SBATCH --time={time}\n\n")
        f.writelines(f'srun {flags} --ntasks={ntasks_per_node} sh -c "{convert_cmd}"\n\n')
        f.writelines("set +x\n")


def create_bcp_file(
    cmd_str,
    num_nodes,
    ntasks_per_node,
    log_file,
    new_script_path
):
    with open(new_script_path, "w") as f:
        f.writelines(f'bcprun -n {num_nodes} -p {ntasks_per_node} -c "{cmd_str}" >> {log_file} 2>&1\n\n')
        f.writelines("set +x\n")
    os.chmod(new_script_path, 0o755)


        
def convert_ckpt(cfg, hydra_args="", dependency=None):
    # Read config
    bignlp_path = cfg.bignlp_path
    container = cfg.container
    container_mounts = cfg.container_mounts
    convert_cfg = cfg.conversion
    data_dir = cfg.data_dir
    base_results_dir = cfg.base_results_dir
    run_cfg = convert_cfg.run
    model_cfg = convert_cfg.model

    # Run parameters
    job_name = run_cfg.job_name
    nodes = run_cfg.nodes
    time_limit = run_cfg.time_limit
    ntasks_per_node = run_cfg.ntasks_per_node
    gpus_per_task = run_cfg.gpus_per_task
    convert_name = run_cfg.convert_name
    model_train_name = run_cfg.model_train_name
    results_dir = run_cfg.results_dir
    output_path = run_cfg.output_path
    nemo_file_name = run_cfg.nemo_file_name
    
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    new_script_path = os.path.join(bignlp_path, f"conversion_scripts/{model_train_name}.sh")
    code_path = os.path.join(bignlp_path, "conversion_scripts/convert_ckpt.py")
    cmd_str = f"python3 -u {code_path} {hydra_args}"
    
    if cfg.cluster_type == "bcm":
        # BCM parameters
        partition = cfg.cluster.partition
        account = cfg.cluster.account
        exclusive = cfg.cluster.exclusive
        job_name_prefix = cfg.cluster.job_name_prefix

        # Process container-mounts.
        mounts_str = f"{bignlp_path}:{bignlp_path},{data_dir}:{data_dir},{base_results_dir}:{base_results_dir}"
        if container_mounts is not None:
            assert isinstance(container_mounts, omegaconf.listconfig.ListConfig), "container_mounts must be a list."
            for mount in container_mounts:
                if mount is not None and isinstance(mount, str):
                    mounts_str += f",{mount}:{mount}"
        flags = (
            f"--no-container-mount-home "
            f"--container-image {container} "
            f"--container-mounts {mounts_str} "
            f"-o {base_results_dir}/{model_train_name}/{convert_name}/convert-%j.log "
            f"-e {base_results_dir}/{model_train_name}/{convert_name}/convert-%j.error "
        )
        create_slurm_file(
            new_script_path=new_script_path,
            convert_cmd=cmd_str,
            job_name=job_name_prefix+job_name,
            flags=flags,
            dependency=dependency,
            exclusive=exclusive,
            mem=None,
            overcommit=None,
            time=time_limit,
            nodes=nodes,
            ntasks_per_node=ntasks_per_node,
            gpus_per_task=gpus_per_task,
            partition=partition,
            account=account,
        )
        job_id = subprocess.check_output(
            [f"sbatch --parsable {new_script_path}"], shell=True
        )
        dependency = job_id.decode("utf-8")
        print(f"Submitted Conversion script with job id: {dependency}")
        return dependency

    elif cfg.cluster_type == "bcp":
        create_bcp_file(
            new_script_path=new_script_path,
            cmd_str=cmd_str,
            num_nodes=nodes,
            ntasks_per_node=ntasks_per_node,
            log_file=f"{base_results_dir}/{model_train_name}/{convert_name}/convert_log.txt",
        )

        submit_cmd = f"NGC_TASKS_PER_NODE={ntasks_per_node} {new_script_path}"
        job_id = subprocess.check_output([f"{submit_cmd}"], shell=True)
        print(f"Conversion job submitted with command: \n{submit_cmd}")
        return job_id
