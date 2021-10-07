import sys
import os
import subprocess

import hydra


def create_slurm_file(
    new_file_name,
    train_cmd,
    job_name,
    blend_path,
    flags="",
    depend=None,
    time="04:00:00",
    exclusive=True,
    nodes=1,
    partition="A100",
):
    path_to_file = os.path.join(os.environ.get("PWD"), new_file_name)
    with open(path_to_file, "w") as f:
        f.writelines("#!/bin/bash\n")
        f.writelines(f"#SBATCH --nodes={nodes}\n")
        if depend is not None:
            f.writelines(f"#SBATCH --depend={depend}\n")
        f.writelines(f"#SBATCH -p {partition}\n")
        f.writelines(f"#SBATCH --job-name={job_name}\n")
        if exclusive:
            f.writelines("#SBATCH --exclusive\n")
        f.writelines(f"#SBATCH --time={time}\n\n")

        f.writelines(f". {blend_path}\n\n")

        f.writelines(f'srun {flags} sh -c "{train_cmd}"\n\n')
        f.writelines("set +x\n")
    return path_to_file


@hydra.main(config_path="../conf", config_name="config")
def main(cfg):
    # Read config
    bignlp_path = cfg.get("bignlp_path")
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
    container = run_cfg.get("container")
    blend_path = os.path.join(bignlp_path, run_cfg.get("blend_path"))
    log_dir = os.path.join(bignlp_path, run_cfg.get("log_dir"))
    if not os.path.exists(log_dir):
        print(log_dir)
        os.makedirs(log_dir)

    flags = (
        f"--container-image {container} "
        f"--container-mounts {bignlp_path}:/workspace/bignlp-scripts "
        f"-o {log_dir}/{name}-%j.log "
        f"-e {log_dir}/{name}-%j.error "
        )
    train_file_name = "train_script.sh"
    train_cmd = f"python3 -u /workspace/bignlp-scripts/train_scripts/pretrain_gpt.py"
    path_to_train_file = create_slurm_file(
        new_file_name=train_file_name,
        train_cmd=train_cmd,
        blend_path=blend_path,
        job_name="bignlp:gpt3-126m",
        flags=flags,
        depend=dependency,
        time=time_limit,
        nodes=nodes,
        partition=partition,
    )
    job_id = subprocess.check_output(
        [f"sbatch --parsable {path_to_train_file}"], shell=True
    )
    job_id = job_id.decode("utf-8")
    print(f"Submitted Training script with job id: {job_id}")


if __name__ == "__main__":
    main()
