import os
import time
import subprocess
import omegaconf

from bignlp.data_preparation.pile_dataprep_scripts import utils
from bignlp.bignlp_utils import add_container_mounts


def create_slurm_file(
    new_script_path,
    code_path,
    log_dir="./",
    flags="",
    args="",
    dependency=None,
    time="04:00:00",
    exclusive=True,
    requeue=True,
    nodes=1,
    partition="batch",
    account=None,
    mem=0,
    overcommit=False,
    job_name="",
):
    if "spm_train" in code_path:
        task = "tokenizer_train"
    else:
        task = code_path.split("/")[-1].split(".")[0]
    node_array = f"0-{nodes-1}"
    with open(new_script_path, "w") as f:
        f.writelines("#!/bin/bash\n")
        f.writelines("#SBATCH --nodes=1\n")
        if dependency is not None:
            f.writelines(f"#SBATCH --dependency=aftercorr:{dependency}\n")
        f.writelines(f"#SBATCH -p {partition}\n")
        if job_name is not None and job_name != "":
            f.writelines(f"#SBATCH --job-name={job_name}\n")
        if account is not None:
            f.writelines(f"#SBATCH -A {account}\n")
        if requeue:
            f.writelines("#SBATCH --requeue\n")
        if exclusive:
            f.writelines("#SBATCH --exclusive\n")
        f.writelines(f"#SBATCH --time={time}\n")
        if mem:
            f.writelines(f"#SBATCH --mem={mem}\n")
        if overcommit:
            f.writelines(f"#SBATCH --overcommit\n")
        f.writelines(f"#SBATCH --array={node_array}%{nodes}\n")
        f.writelines(f"#SBATCH -o {log_dir}/log-{task}-%j_%a.out\n")
        args = args.replace(" ", " \\\n  ")
        if "spm_train" not in code_path:
            f.writelines(f"srun {flags} \\\n python3 {code_path} \\\n  {args} &\n")
        else:
            f.writelines(f"srun {flags} \\\n {code_path} \\\n  {args} &\n")
        f.writelines("wait\n")

def run_data_preparation(cfg, hydra_args="", dependency=None):
    # Read config
    bignlp_path = cfg.get("bignlp_path")
    container = cfg.get("container")
    container_mounts = cfg.get("container_mounts")
    data_dir = cfg.get("data_dir")
    base_results_dir = cfg.get("base_results_dir")
    data_cfg = cfg.get("data_preparation")

    # Data preparation config
    custom_dataset_dir = data_cfg.get("custom_dataset_dir")
    train_tokenizer = data_cfg.get("train_tokenizer")
    preprocess_data = data_cfg.get("preprocess_data")
    train_tokenizer_args = data_cfg.get("train_tokenizer_args")
    bpe_save_dir = data_cfg.get("bpe_save_dir")
    raw_dataset_files = data_cfg.get("raw_dataset_files")
    tokenizer_model = data_cfg.get("tokenizer_model")
    preprocess_worker_mapping = data_cfg.get("preprocess_worker_mapping")
    preprocessed_dir = data_cfg.get("preprocessed_dir")
    log_dir = data_cfg.get("log_dir")
    nodes = data_cfg.get("nodes")
    time_limit = data_cfg.get("time_limit")
    cpus_per_node = data_cfg.get("cpus_per_node")
    workers_per_node = data_cfg.get("workers_per_node")

    os.makedirs(custom_dataset_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(bpe_save_dir, exist_ok=True)

    # Setup preprocess data
    if preprocess_data:
        # Sort list of files in directory by size
        sorted_files = sorted(raw_dataset_files, key=lambda x: os.stat(x).st_size)
        file_sizes = [os.stat(x).st_size for x in sorted_files]

        avail_workers = nodes * workers_per_node
        distributed_files = [[] for _ in range(avail_workers)]
        distributed_size = [0] * avail_workers
        for f, file_size in zip(sorted_files, file_sizes):
            min_ind = distributed_size.index(min(distributed_size))
            distributed_files[min_ind].append(f)
            distributed_size[min_ind] += file_size

        output = [",".join(distributed_files[i]) for i in range(avail_workers)]
        output = "\n".join(output)
        with open(preprocess_worker_mapping, "w") as file:
            file.write(output)
        print(f" ****** Workers mapping saved to {preprocess_worker_mapping} ...")
        for i in range(avail_workers):
            print(
                f"{i + 1:>4d} "
                f"{distributed_size[i]:>7.2f}GB  "
                f"{','.join([os.path.basename(file) for file in distributed_files[i]]):s}"
            )

    # Define running commands
    train_tokenizer_cmd = f"cd {bpe_save_dir} && spm_train "
    train_tokenizer_args = " ".join([f"--{k}={v}" for k, v in train_tokenizer_args.items()])

    preprocess_code_path = os.path.join(
        bignlp_path, "bignlp/data_preparation/custom_dataprep_scripts/preprocess.py"
    )
    preprocess_args = (
        f"--worker-mapping-file={preprocess_worker_mapping} "
        f"--workers-per-node={workers_per_node} "
        f"--output-path={preprocessed_dir} "
        f"--tokenizer-library=sentencepiece "
        f"--tokenizer-model={tokenizer_model} "
        f"--dataset-impl=mmap "
        f"--workers={cpus_per_node // workers_per_node} "
        f"--log-interval=2000 "
        f"--apply-ftfy"
    )

    # BCM config
    cluster_cfg = cfg.get("cluster")
    if cfg.get("cluster_type") == "bcm" and cluster_cfg is not None:
        partition = cluster_cfg.get("partition")
        account = cluster_cfg.get("account")
        exclusive = cluster_cfg.get("exclusive")
        mem = cluster_cfg.get("mem")
        overcommit = cluster_cfg.get("overcommit")
        job_name_prefix = cluster_cfg.get("job_name_prefix")

        # Process container-mounts.
        mounts_str = f"{bignlp_path}:{bignlp_path},{data_dir}:{data_dir},{base_results_dir}:{base_results_dir}"
        mounts_str += add_container_mounts(container_mounts)

        flags = f"--container-image {container} --container-mounts {mounts_str}"

        assert isinstance(train_tokenizer, bool), "train_tokenizer must be bool."
        if train_tokenizer:
            # Prepare custom tokenizer
            train_tokenizer_script_path = os.path.join(
                bignlp_path, "bignlp/data_preparation/train_custom_tokenizer_script.sh"
            )
            create_slurm_file(
                new_script_path=train_tokenizer_script_path,
                code_path=train_tokenizer_cmd,
                log_dir=log_dir,
                flags=flags,
                args=train_tokenizer_args,
                dependency=dependency,
                time=time_limit,
                nodes=1,
                partition=partition,
                account=account,
                mem=mem,
                exclusive=exclusive,
                overcommit=overcommit,
                job_name=f"{job_name_prefix}train_tokenizer",
            )
            job_id = subprocess.check_output(
                [f"sbatch --parsable {train_tokenizer_script_path}"], shell=True
            )
            dependency = job_id.decode("utf-8")
            print(f"Submitted custom tokenizer train script with job id: {dependency}")

        assert isinstance(preprocess_data, bool), "preprocess_data must be bool."
        if preprocess_data:
            # Preprocess the dataset
            preprocess_script_path = os.path.join(
                bignlp_path, "bignlp/data_preparation/preprocess_custom_dataset_script.sh"
            )
            preprocess_flags = (
                f"{flags}"
                f" --ntasks-per-node={workers_per_node} "
                f" --cpus-per-task={cpus_per_node // workers_per_node}"
            )
            create_slurm_file(
                new_script_path=preprocess_script_path,
                code_path=preprocess_code_path,
                log_dir=log_dir,
                flags=preprocess_flags,
                args=preprocess_args,
                dependency=dependency,
                time=time_limit,
                nodes=nodes,
                partition=partition,
                account=account,
                mem=mem,
                overcommit=overcommit,
                job_name=f"{job_name_prefix}preprocess",
            )
            job_id = subprocess.check_output(
                [f"sbatch --parsable {preprocess_script_path}"], shell=True
            )
            dependency = job_id.decode("utf-8")
            print(f"Submitted custom dataset Preprocessing script with job id: {dependency}")
        return dependency

    if cfg.get("cluster_type") == "bcp":
        def get_launcher(nnodes, npernode, cmd):
            if utils.is_tool("bcprun"):
                launcher = (
                    "NGC_ARRAY_TYPE=MPIJob "
                    f"bcprun --nnodes {nnodes} --npernode {npernode} "
                    f"--launcher 'mpirun --allow-run-as-root' --cmd \"{cmd}\""
                )
            else:
                launcher = (
                    f"mpirun --allow-run-as-root "
                    f"-np {nnodes * npernode} -npernode {npernode} {cmd}"
                )
            return launcher

        joblog = os.path.join(log_dir, "data_joblog.log")
        nnodes = int(os.environ.get("NGC_ARRAY_SIZE", 1))

        assert isinstance(train_tokenizer, bool), "train_tokenizer must be bool."
        if train_tokenizer:
            # Train tokenizer
            launchcmd = f"{train_tokenizer_cmd} {train_tokenizer_args}"
            proc = subprocess.Popen(
                launchcmd, shell=True, stdout=subprocess.PIPE, universal_newlines=True
            )
            print(f"\nSubmitted train custom tokenizer script with job pid: {proc.pid}")
            with open(joblog, "a", encoding="utf-8") as jlog:
                print(f"train custom tokenizer CMD:\n{launchcmd}", file=jlog)
                for line in proc.stdout:
                    print(line)
                    jlog.write(line)

            proc.wait()
            print(f"Finished train custom tokenizer script returncode: {proc.returncode}")

        assert isinstance(preprocess_data, bool), "preprocess_data must be bool."
        if preprocess_data:
            # Preprocess the dataset
            preprocess_args += " --bcp "
            cmd = f"python3 {preprocess_code_path} {preprocess_args}"
            launchcmd = get_launcher(nnodes, workers_per_node, cmd)
            proc = subprocess.Popen(
                launchcmd, shell=True, stdout=subprocess.PIPE, universal_newlines=True
            )
            print(f"\nSubmitted custom dataset Preprocess script with job pid: {proc.pid}")
            with open(joblog, "a", encoding="utf-8") as jlog:
                print(f"custom dataset Preprocess CMD:\n{launchcmd}", file=jlog)
                for line in proc.stdout:
                    print(line)
                    jlog.write(line)

            proc.wait()
            print(f"Finished custom dataset Preprocess script returncode: {proc.returncode}")
    return None
