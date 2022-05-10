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
    task = code_path.split("/")[-1].split(".")[0]
    node_array = f"0-{nodes-1}"
    with open(new_script_path, "w") as f:
        f.writelines("#!/bin/bash\n")
        f.writelines("#SBATCH --nodes=1\n")
        if dependency is not None:
            f.writelines(f"#SBATCH --dependency=aftercorr:{dependency}\n")
        f.writelines(f"#SBATCH -p {partition}\n")
        if job_name is None or job_name == "":
            job_name = "slurm_job"
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
        f.writelines(f"srun {flags} \\\n python3 {code_path} \\\n  {args} &\n")
        f.writelines("wait\n")


def download_single_file(url, save_dir, file_name):
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, file_name)
    if os.path.exists(save_path):
        print(f"File {save_path} already exists, skipping download.")
        return save_path
    subprocess.call(f"wget {url} -O {save_path}", shell=True)


def run_data_preparation(cfg, hydra_args="", dependency=None):
    # Read config
    bignlp_path = cfg.get("bignlp_path")
    container = cfg.get("container")
    container_mounts = cfg.get("container_mounts")
    data_dir = cfg.get("data_dir")
    base_results_dir = cfg.get("base_results_dir")
    data_cfg = cfg.get("data_preparation")

    # Data preparation config
    download_mc4 = data_cfg.get("download_mc4")
    preprocess_data = data_cfg.get("preprocess_data")
    mc4_dir = data_cfg.get("mc4_dir")
    preprocessed_dir = data_cfg.get("preprocessed_dir")
    git_lfs_dir = data_cfg.get("git_lfs_dir")
    download_vocab_url = data_cfg.get("download_vocab_url")
    download_tokenizer_url = data_cfg.get("download_tokenizer_url")
    vocab_save_dir = data_cfg.get("vocab_save_dir")
    tokenizer_save_dir = data_cfg.get("tokenizer_save_dir")
    tokenizer_model = data_cfg.get("tokenizer_model")
    languages = data_cfg.get("languages")
    use_cleaned_english = data_cfg.get("use_cleaned_english")
    softlinks_dir = data_cfg.get("softlinks_dir")
    download_worker_mapping = data_cfg.get("download_worker_mapping")
    preprocess_worker_mapping = data_cfg.get("preprocess_worker_mapping")
    rm_downloaded = data_cfg.get("rm_downloaded")
    log_dir = data_cfg.get("log_dir")
    nodes = data_cfg.get("nodes")
    time_limit = data_cfg.get("time_limit")
    cpus_per_node = data_cfg.get("cpus_per_node")
    workers_per_node = data_cfg.get("workers_per_node")
    max_split_size = 200  # TODO: maybe add this to config

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Download vocab
    if download_vocab_url is not None:
        assert vocab_save_dir is not None, "vocab_save_dir must be a valid path."
        download_single_file(url=download_vocab_url, save_dir=vocab_save_dir, file_name="vocab.txt")

    if download_tokenizer_url is not None:
        assert tokenizer_save_dir is not None, "vocab_save_dir must be a valid path."
        download_single_file(
            url=download_tokenizer_url, save_dir=tokenizer_save_dir, file_name="mt5_tokenizer.model"
        )

    # Define running commands
    prepare_code_path = os.path.join(
        bignlp_path, "bignlp/data_preparation/mc4_dataprep_scripts/prepare.py"
    )
    cleaned_en = "--cleaned-en " if use_cleaned_english else ""
    prepare_args = (
        f"--data-path={mc4_dir} "
        f"--git-lfs-path={git_lfs_dir} "
        f"--languages={languages} "
        f"{cleaned_en}"
        f"--node-array-size={nodes} "
        f"--worker-mapping-file={download_worker_mapping}"
    )

    download_code_path = os.path.join(
        bignlp_path, "bignlp/data_preparation/mc4_dataprep_scripts/download.py"
    )
    download_args = (
        f"--c4-path={os.path.join(mc4_dir, 'c4')} "
        f"--git-lfs-path={git_lfs_dir} "
        f"--worker-mapping-file={download_worker_mapping}"
    )

    setup_preprocess_code_path = os.path.join(
        bignlp_path, "bignlp/data_preparation/mc4_dataprep_scripts/setup_preprocess.py"
    )
    setup_preprocess_args = (
        f"--c4-path={os.path.join(mc4_dir, 'c4')} "
        f"--soft-link-path={softlinks_dir} "
        f"--languages={languages} "
        f"{cleaned_en}"
        f"--node-array-size={nodes} "
        f"--workers-per-node={workers_per_node} "
        f"--max-split-size={max_split_size} "
        f"--worker-mapping-file={preprocess_worker_mapping}"
    )

    preprocess_code_path = os.path.join(
        bignlp_path, "bignlp/data_preparation/mc4_dataprep_scripts/preprocess.py"
    )
    rm_arg = "--rm-downloaded" if rm_downloaded else ""
    preprocess_args = (
        f"{rm_arg} "
        f"--worker-mapping-file={preprocess_worker_mapping} "
        f"--workers-per-node={workers_per_node} "
        f"--output-path={preprocessed_dir} "
        f"--tokenizer-library=sentencepiece "
        f"--tokenizer-model={tokenizer_model} "
        f"--dataset-impl=mmap "
        f"--workers={cpus_per_node // workers_per_node} "
        f"--log-interval=2000 "
        f"--preproc-folder "
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

        assert isinstance(download_mc4, bool), "download_mc4 must be bool."
        if download_mc4:
            # Prepare mC4 dataset repo
            prepare_script_path = os.path.join(
                bignlp_path, "bignlp/data_preparation/prepare_mc4_script.sh"
            )
            create_slurm_file(
                new_script_path=prepare_script_path,
                code_path=prepare_code_path,
                log_dir=log_dir,
                flags=flags,
                args=prepare_args,
                dependency=dependency,
                time=time_limit,
                nodes=1,
                partition=partition,
                account=account,
                mem=mem,
                exclusive=exclusive,
                overcommit=overcommit,
                job_name=f"{job_name_prefix}mc4_prepare",
            )
            job_id = subprocess.check_output(
                [f"sbatch --parsable {prepare_script_path}"], shell=True
            )
            dependency = job_id.decode("utf-8")
            time.sleep(0.5)
            print(f"Submitted mC4 Prepare script with job id: {dependency}")

            # Download mC4 dataset files
            download_script_path = os.path.join(
                bignlp_path, "bignlp/data_preparation/download_mc4_script.sh"
            )
            create_slurm_file(
                new_script_path=download_script_path,
                code_path=download_code_path,
                log_dir=log_dir,
                flags=flags,
                args=download_args,
                dependency=dependency,
                time=time_limit,
                nodes=nodes,
                partition=partition,
                account=account,
                mem=mem,
                exclusive=exclusive,
                overcommit=overcommit,
                job_name=f"{job_name_prefix}mc4_download",
            )
            job_id = subprocess.check_output(
                [f"sbatch --parsable {download_script_path}"], shell=True
            )
            dependency = job_id.decode("utf-8")
            time.sleep(0.5)
            print(f"Submitted mC4 Download script with job id: {dependency}")

        assert isinstance(preprocess_data, bool), "preprocess_data must be bool."
        if preprocess_data:
            # Setup preprocess for mC4 dataset
            setup_preprocess_script_path = os.path.join(
                bignlp_path, "bignlp/data_preparation/setup_preprocess_mc4_script.sh"
            )
            create_slurm_file(
                new_script_path=setup_preprocess_script_path,
                code_path=setup_preprocess_code_path,
                log_dir=log_dir,
                flags=flags,
                args=setup_preprocess_args,
                dependency=dependency,
                time=time_limit,
                nodes=1,
                partition=partition,
                account=account,
                mem=mem,
                overcommit=overcommit,
                job_name=f"{job_name_prefix}setup_preprocess",
            )
            job_id = subprocess.check_output(
                [f"sbatch --parsable {setup_preprocess_script_path}"], shell=True
            )
            dependency = job_id.decode("utf-8")
            time.sleep(0.5)
            print(f"Submitted mC4 Setup Preprocessing script with job id: {dependency}")

            # Preprocess the dataset
            preprocess_script_path = os.path.join(
                bignlp_path, "bignlp/data_preparation/preprocess_mc4_script.sh"
            )
            preprocess_flags = (
                flags + f" --ntasks-per-node={workers_per_node} "
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
            time.sleep(0.5)
            print(f"Submitted mC4 Preprocessing script with job id: {dependency}")
        return dependency

    if cfg.get("cluster_type") == "bcp":

        def get_launcher(nnodes, npernode, cmd):
            if utils.is_tool("bcprun"):
                launcher = (
                    "NGC_ARRAY_TYPE=MPIJob "
                    + f"bcprun --nnodes {nnodes} --npernode {npernode} "
                    + f"--launcher 'mpirun --allow-run-as-root' --cmd \"{cmd}\""
                )
            else:
                launcher = (
                    f"mpirun --allow-run-as-root "
                    + f"-np {nnodes * npernode} -npernode {npernode} {cmd}"
                )
            return launcher

        joblog = os.path.join(log_dir, "data_joblog.log")
        nnodes = int(os.environ.get("NGC_ARRAY_SIZE", 1))

        assert isinstance(download_mc4, bool), "download_mc4 must be bool."
        if download_mc4:
            # Prepare mC4 dataset repo
            cmd = f"python3 {prepare_code_path} {prepare_args}"
            launchcmd = cmd  # prepare env do not need mpirun
            proc = subprocess.Popen(
                launchcmd, shell=True, stdout=subprocess.PIPE, universal_newlines=True
            )
            print(f"\nSubmitted mC4 Prepare script with job pid: {proc.pid}")
            with open(joblog, "a", encoding="utf-8") as jlog:
                print(f"mC4 Prepare CMD:\n{launchcmd}", file=jlog)
                for line in proc.stdout:
                    print(line)
                    jlog.write(line)

            proc.wait()
            print(f"Finished mC4 Prepare script returncode: {proc.returncode}")

            # Download mC4 dataset files
            download_args += " --bcp "
            cmd = f"python3 {download_code_path} {download_args}"
            launchcmd = get_launcher(nnodes, 1, cmd)
            proc = subprocess.Popen(
                launchcmd, shell=True, stdout=subprocess.PIPE, universal_newlines=True
            )
            print(f"\nSubmitted mC4 Download script with job pid: {proc.pid}")
            with open(joblog, "a", encoding="utf-8") as jlog:
                print(f"mC4 Download CMD:\n{launchcmd}", file=jlog)
                for line in proc.stdout:
                    print(line)
                    jlog.write(line)

            proc.wait()
            print(f"Finished mC4 Download script returncode: {proc.returncode}")

        assert isinstance(preprocess_data, bool), "preprocess_data must be bool."
        if preprocess_data:
            # Setup preprocess for mC4 dataset
            cmd = f"python3 {setup_preprocess_code_path} {setup_preprocess_args}"
            launchcmd = cmd  # prepare env do not need mpirun
            proc = subprocess.Popen(
                launchcmd, shell=True, stdout=subprocess.PIPE, universal_newlines=True
            )
            print(f"\nSubmitted mC4 Setup Preprocessing script with job pid: {proc.pid}")
            with open(joblog, "a", encoding="utf-8") as jlog:
                print(f"mC4 Setup Preprocessing CMD:\n{launchcmd}", file=jlog)
                for line in proc.stdout:
                    print(line)
                    jlog.write(line)

            proc.wait()
            print(f"Finished mC4 Setup Preprocessing script returncode: {proc.returncode}")

            # Preprocess the dataset
            preprocess_args += " --bcp "
            cmd = f"python3 {preprocess_code_path} {preprocess_args}"
            launchcmd = get_launcher(nnodes, workers_per_node, cmd)
            proc = subprocess.Popen(
                launchcmd, shell=True, stdout=subprocess.PIPE, universal_newlines=True
            )
            print(f"\nSubmitted mC4 Preprocess script with job pid: {proc.pid}")
            with open(joblog, "a", encoding="utf-8") as jlog:
                print(f"mC4 Preprocess CMD:\n{launchcmd}", file=jlog)
                for line in proc.stdout:
                    print(line)
                    jlog.write(line)

            proc.wait()
            print(f"Finished mC4 Preprocess script returncode: {proc.returncode}")
    return None
