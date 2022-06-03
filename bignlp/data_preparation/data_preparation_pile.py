import os
import subprocess

import omegaconf

from bignlp.data_preparation.pile_dataprep_scripts import utils
from bignlp.bignlp_utils import add_container_mounts


def create_slurm_file(
    new_script_path,
    code_path,
    log_dir="./",
    flags="",
    hydra_args="",
    dependency=None,
    time="04:00:00",
    exclusive=True,
    requeue=True,
    file_numbers="0",
    nodes=1,
    partition="batch",
    account=None,
    mem=0,
    overcommit=False,
    job_name="",
):
    task = code_path.split("/")[-1].split(".")[0]
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
        f.writelines(f"#SBATCH --array={file_numbers}%{nodes}\n")
        f.writelines(f"#SBATCH -o {log_dir}/log-{task}-%j_%a.out\n")
        f.writelines(f"srun {flags} python3 {code_path} {hydra_args} &\n")
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
    download_the_pile = data_cfg.get("download_the_pile")
    file_numbers = data_cfg.get("file_numbers")
    preprocess_data = data_cfg.get("preprocess_data")
    download_vocab_url = data_cfg.get("download_vocab_url")
    download_merges_url = data_cfg.get("download_merges_url")
    vocab_save_dir = data_cfg.get("vocab_save_dir")
    merges_save_dir = data_cfg.get("merges_save_dir")
    log_dir = data_cfg.get("log_dir")
    nodes = data_cfg.get("nodes")
    time_limit = data_cfg.get("time_limit")

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Download vocab
    if download_vocab_url is not None:
        assert vocab_save_dir is not None, "vocab_save_dir must be a valid path."
        utils.download_single_file(
            url=download_vocab_url,
            save_dir=vocab_save_dir,
            file_name="vocab.json" if download_vocab_url.endswith("json") else "vocab.txt",
        )

    # Download merges
    if download_merges_url is not None:
        assert merges_save_dir is not None, "merges_save_dir must be a valid path."
        utils.download_single_file(
            url=download_merges_url,
            save_dir=merges_save_dir,
            file_name="merges.txt",
        )

    download_code_path = os.path.join(
        bignlp_path, "bignlp/data_preparation/pile_dataprep_scripts/download.py"
    )
    extract_code_path = os.path.join(
        bignlp_path, "bignlp/data_preparation/pile_dataprep_scripts/extract.py"
    )
    preprocess_code_path = os.path.join(
        bignlp_path, "bignlp/data_preparation/pile_dataprep_scripts/preprocess.py"
    )

    # BCM config
    cluster_cfg = cfg.get("cluster")
    if cfg.get("cluster_type") == "bcm" and cluster_cfg is not None:
        partition = cfg.cluster.partition
        account = cfg.cluster.account
        exclusive = cfg.cluster.exclusive
        mem = cfg.cluster.mem
        overcommit = cfg.cluster.overcommit
        job_name_prefix = cfg.cluster.job_name_prefix

        # Process container-mounts.
        mounts_str = f"{bignlp_path}:{bignlp_path},{data_dir}:{data_dir},{base_results_dir}:{base_results_dir}"
        mounts_str += add_container_mounts(container_mounts)

        flags = f"--container-image {container} --container-mounts {mounts_str}"

        assert isinstance(download_the_pile, bool), "download_the_pile must be bool."
        if download_the_pile:
            # Download The Pile dataset files
            download_script_path = os.path.join(
                bignlp_path, "bignlp/data_preparation/download_script.sh"
            )
            create_slurm_file(
                new_script_path=download_script_path,
                code_path=download_code_path,
                log_dir=log_dir,
                flags=flags,
                hydra_args=hydra_args,
                dependency=dependency,
                time=time_limit,
                file_numbers=file_numbers,
                nodes=nodes,
                partition=partition,
                account=account,
                mem=mem,
                overcommit=overcommit,
                job_name=f"{job_name_prefix}download",
            )
            if cfg.get("ci_test"):
                job_id_1 = subprocess.check_output([f'sbatch {download_script_path} | tee "{log_dir}/launcher.log" '], shell=True)
            else:
                job_id_1 = subprocess.check_output([f"sbatch --parsable {download_script_path}"], shell=True)
            dependency = job_id_1.decode("utf-8")
            print(f"Submitted Download script with job id: {dependency}")

            # Extract The Pile dataset files
            extract_script_path = os.path.join(
                bignlp_path, "bignlp/data_preparation/extract_script.sh"
            )
            create_slurm_file(
                new_script_path=extract_script_path,
                code_path=extract_code_path,
                log_dir=log_dir,
                flags=flags,
                hydra_args=hydra_args,
                dependency=dependency,
                time=time_limit,
                file_numbers=file_numbers,
                nodes=nodes,
                partition=partition,
                account=account,
                mem=mem,
                overcommit=overcommit,
                job_name=f"{job_name_prefix}extract",
            )
            if cfg.get("ci_test"):
                job_id_2 = subprocess.check_output([f'sbatch {extract_script_path} | tee "{log_dir}/launcher.log" '], shell=True)
            else:
                job_id_2 = subprocess.check_output([f"sbatch --parsable {extract_script_path}"], shell=True)
            dependency = job_id_2.decode("utf-8")
            print(f"Submitted Extract script with job id: {dependency}")

        assert isinstance(preprocess_data, bool), "preprocess_data must be bool."
        if preprocess_data:
            # Preprocess the dataset
            preprocess_script_path = os.path.join(
                bignlp_path, "bignlp/data_preparation/preprocess_script.sh"
            )
            create_slurm_file(
                new_script_path=preprocess_script_path,
                code_path=preprocess_code_path,
                log_dir=log_dir,
                flags=flags,
                hydra_args=hydra_args,
                dependency=dependency,
                time=time_limit,
                file_numbers=file_numbers,
                nodes=nodes,
                partition=partition,
                account=account,
                mem=mem,
                overcommit=overcommit,
                job_name=f"{job_name_prefix}preprocess",
            )
            if cfg.get("ci_test"):
                job_id_3 = subprocess.check_output([f'sbatch {preprocess_script_path} | tee "{log_dir}/launcher.log" '], shell=True)
            else:
                job_id_3 = subprocess.check_output([f"sbatch --parsable {preprocess_script_path}"], shell=True)
            dependency = job_id_3.decode("utf-8")
            print(f"Submitted Preprocessing script with job id: {dependency}")
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

        assert isinstance(download_the_pile, bool), "download_the_pile must be bool."
        if download_the_pile:
            # Downloading the files
            cmd = f"python3 {download_code_path} {hydra_args}"
            launchcmd = get_launcher(nnodes, 1, cmd)
            proc = subprocess.Popen(
                launchcmd, shell=True, stdout=subprocess.PIPE, universal_newlines=True
            )
            print(f"\nSubmitted Download script with job pid: {proc.pid}")
            with open(joblog, "a", encoding="utf-8") as jlog:
                print(f"Download CMD:\n{launchcmd}", file=jlog)
                for line in proc.stdout:
                    print(line)
                    jlog.write(line)

            proc.wait()
            print(f"Finished Download script returncode: {proc.returncode}")

            # Extract The Pile dataset files
            cmd = f"python3 {extract_code_path} {hydra_args}"
            launchcmd = get_launcher(nnodes, 1, cmd)
            proc = subprocess.Popen(
                launchcmd, shell=True, stdout=subprocess.PIPE, universal_newlines=True
            )
            print(f"\nSubmitted extract script with job pid: {proc.pid}")
            with open(joblog, "a", encoding="utf-8") as jlog:
                print(f"Extract CMD:\n{launchcmd}", file=jlog)
                for line in proc.stdout:
                    print(line)
                    jlog.write(line)

            proc.wait()
            print(f"Finished extract script returncode: {proc.returncode}")

        assert isinstance(preprocess_data, bool), "preprocess_data must be bool."
        if preprocess_data:
            # Preprocess the dataset
            megatron_dir = "/opt/bignlp/NeMo/nemo/collections/nlp/data/language_modeling/megatron"
            # Remove compiled helpers lib to avoid race condition
            compiled_helpers_lib = os.path.join(megatron_dir, "compiled_helpers_lib")
            clean = (
                f'bash -c \'[ ! -e "{compiled_helpers_lib}" ] || '
                + f'rm "{compiled_helpers_lib}" \''
            )
            cleancmd = get_launcher(nnodes, 1, clean)
            os.system(cleancmd)

            preproc_npernode = int(data_cfg.get("bcp_preproc_npernode"))
            cmd = f"python3 {preprocess_code_path} {hydra_args}"
            launchcmd = get_launcher(nnodes, preproc_npernode, cmd)
            proc = subprocess.Popen(
                launchcmd, shell=True, stdout=subprocess.PIPE, universal_newlines=True
            )
            print(f"\nSubmitted preprocess script with job pid: {proc.pid}")
            with open(joblog, "a", encoding="utf-8") as jlog:
                print(f"Preprocess CMD:\n{launchcmd}", file=jlog)
                for line in proc.stdout:
                    print(line)
                    jlog.write(line)

            proc.wait()
            print(f"Finished preprocess script returncode: {proc.returncode}")
    return None
