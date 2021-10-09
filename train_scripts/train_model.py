import sys
import os
import subprocess

import hydra
from omegaconf import OmegaConf


def create_slurm_file(
    new_script_path,
    train_cmd,
    job_name,
    blend_path,
    flags="",
    dependency=None,
    time="04:00:00",
    exclusive=True,
    mem=0,
    overcommit=True,
    nodes=1,
    ntasks_per_node=8,
    gpus_per_task=1,
    partition="batch",
):
    with open(new_script_path, "w") as f:
        f.writelines("#!/bin/bash\n")
        f.writelines(f"#SBATCH --nodes={nodes}\n")
        f.writelines(f"#SBATCH --ntasks-per-node={ntasks_per_node}\n")
        f.writelines(f"#SBATCH --gpus-per-task={gpus_per_task}\n")
        if dependency is not None:
            f.writelines(f"#SBATCH --dependency={dependency}\n")
        f.writelines(f"#SBATCH -p {partition}\n")
        f.writelines(f"#SBATCH --job-name={job_name}\n")
        f.writelines(f"#SBATCH --mem={mem}\n")
        if exclusive:
            f.writelines("#SBATCH --exclusive\n")
        if overcommit:
            f.writelines("#SBATCH --overcommit\n")
        f.writelines(f"#SBATCH --time={time}\n\n")
        f.writelines(f". {blend_path}\n\n")
        f.writelines(f'srun {flags} sh -c "{train_cmd}"\n\n')
        f.writelines("set +x\n")


@hydra.main(config_path="../conf", config_name="config")
def main(cfg):
    # Read config
    bignlp_path = cfg.get("bignlp_path")
    container = cfg.get("container")
    train_cfg = cfg.get("training")
    run_cfg = train_cfg.get("run")
    megatron_cfg = train_cfg.get("megatron")

    # SLURM parameters
    slurm_cfg = train_cfg.get("slurm")
    partition = slurm_cfg.get("partition")
    account = slurm_cfg.get("account")
    time_limit = slurm_cfg.get("time_limit")
    nodes = slurm_cfg.get("nodes")
    exclusive = slurm_cfg.get("exclusive")
    mem = slurm_cfg.get("mem")
    overcommit = slurm_cfg.get("overcommit")
    ntasks_per_node = slurm_cfg.get("ntasks_per_node")
    dependency = slurm_cfg.get("dependency")
    job_name = slurm_cfg.get("job_name")

    # Run parameters
    name = run_cfg.get("name")
    blend_path = run_cfg.get("blend_path")
    full_blend_path = os.path.join(bignlp_path, blend_path)
    log_dir = os.path.join(bignlp_path, run_cfg.get("log_dir"))
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    flags = (
        f"--container-image {container} "
        f"--container-mounts {bignlp_path}:{bignlp_path} "
        f"-o {log_dir}/{name}-%j.log "
        f"-e {log_dir}/{name}-%j.error "
    )
    new_script_path = os.path.join(bignlp_path, "train_scripts/train_script.sh")
    code_path = os.path.join(bignlp_path, "train_scripts/pretrain_gpt.py")
    train_cmd = f"python3 -u {code_path}"
    create_slurm_file(
        new_script_path=new_script_path,
        train_cmd=train_cmd,
        blend_path=full_blend_path,
        job_name=f"bignlp:{name}",
        flags=flags,
        dependency=dependency,
        exclusive=exclusive,
        mem=mem,
        overcommit=overcommit,
        time=time_limit,
        nodes=nodes,
        partition=partition,
    )
    #job_id = subprocess.check_output(
    #    [f"sbatch --parsable {new_script_path}"], shell=True
    #)
    #job_id = job_id.decode("utf-8")
    #print(f"Submitted Training script with job id: {job_id}")


if __name__ == "__main__":
    main()
