import sys
import os
import subprocess

import hydra
import omegaconf

from bignlp.bignlp_utils import convert_to_cli, add_container_mounts, create_slurm_file, create_bcp_file
from bignlp.finetune_scripts.data import download_glue
from bignlp.data_preparation.pile_dataprep_scripts.utils import download_single_file


def run_prompt_learning(cfg, hydra_args="", dependency=None):
    """
    Main function to launch a training job, with the config given in cfg.
    """
    # Read config
    bignlp_path = cfg.get("bignlp_path")
    container_mounts = cfg.get("container_mounts")
    container = cfg.get("container")
    prompt_learn_cfg = cfg.get("prompt_learning")
    cluster_cfg = cfg.get("cluster")
    data_dir = cfg.get("data_dir")
    base_results_dir = cfg.get("base_results_dir")
    run_cfg = prompt_learn_cfg.get("run")

    # Run parameters
    name = run_cfg.get("name")
    results_dir = run_cfg.get("results_dir")
    time_limit = run_cfg.get("time_limit")
    task_name = run_cfg.get("task_name")

    os.makedirs(results_dir, exist_ok=True)

    # Shared between BCP and BCM
    new_script_path = os.path.join(bignlp_path, f"bignlp/prompt_learn_scripts/{name}.sh")
    code_path = os.path.join(bignlp_path, "bignlp/prompt_learn_scripts/prompt_learn_gpt.py")

    # prepare dataset for squad
    if task_name == 'squad':
        data_dir = os.path.join(data_dir, "prompt_data")
        squad_dir = os.path.join(data_dir, 'squad-v2.0')
        if not os.path.exists(squad_dir):
            os.makedirs(squad_dir)
            download_single_file("https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v2.0.json", squad_dir, "train-v2.0.json")
            download_single_file("https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json", squad_dir, "dev-v2.0.json")
            preprocess_script = os.path.join(bignlp_path, "bignlp/prompt_learn_scripts/data/squad/prompt_learning_squad_preprocessing.py")
            os.system(f"python {preprocess_script} "
                      f"--data-dir={squad_dir} "
                      f"--file-name='train-v2.0.json' "
                      f"--save-name-base='squad' "
                      f"--make-ground-truth "
                      f"--train-percent=0.8")

    base_cmd = f"python3 -u {code_path} \\\n  {hydra_args}"

    nodes = prompt_learn_cfg.trainer.num_nodes
    ntasks_per_node = prompt_learn_cfg.trainer.devices

    # BCM parameters
    if cfg.get("cluster_type") == "bcm":
        partition = cluster_cfg.get("partition")
        account = cluster_cfg.get("account")
        exclusive = cluster_cfg.get("exclusive")
        gpus_per_task = cluster_cfg.get("gpus_per_task")
        gpus_per_node = cluster_cfg.get("gpus_per_node")
        job_name_prefix = cluster_cfg.get("job_name_prefix")

        if dependency is None:
            dependency = run_cfg.get("dependency")
        job_name = job_name_prefix + name

        train_cmd = f"PYTHONPATH={bignlp_path}:${{PYTHONPATH}} \\\n {base_cmd}"

        # Process container-mounts.
        mounts_str = f"{bignlp_path}:{bignlp_path},{data_dir}:{data_dir},{base_results_dir}:{base_results_dir}"
        mounts_str += add_container_mounts(container_mounts)

        if cfg.get("ci_test"):  # Whether this job is running in CI or not.
            flags = (
                f"--container-image {container} --container-mounts {mounts_str} "
                f"-o {results_dir}/slurm_%j.log "
            )
        else:
            flags = (
                f"--container-image {container} --container-mounts {mounts_str} "
                f"-o {results_dir}/{name}-%j.log -e {results_dir}/{name}-%j.error "
            )

        create_slurm_file(
            new_script_path=new_script_path,
            slurm_cmd=train_cmd,
            job_name=job_name,
            flags=flags,
            dependency=dependency,
            exclusive=exclusive,
            time=time_limit,
            nodes=nodes,
            ntasks_per_node=ntasks_per_node,
            gpus_per_task=gpus_per_task,
            gpus_per_node=gpus_per_node,
            partition=partition,
            account=account,
        )
        if cfg.get("ci_test"):
            job_id = subprocess.check_output([f'sbatch {new_script_path} | tee "{results_dir}/launcher.log" '], shell=True)
        else:
            job_id = subprocess.check_output([f"sbatch --parsable {new_script_path}"], shell=True)
        dependency = job_id = job_id.decode("utf-8")
        print(f"Submitted Finetuning script with job id: {dependency}")
        return dependency

    # BCP parameters
    if cfg.get("cluster_type") == "bcp":
        env_exports = f"PYTHONPATH=${{PYTHONPATH}}:{bignlp_path}"
        create_bcp_file(
            new_script_path=new_script_path,
            bcp_cmd=base_cmd,
            num_nodes=nodes,
            log_file=f"{results_dir}/{name}.log",
            env_exports=env_exports,
        )
        submit_cmd = f"NGC_NTASKS_PER_NODE={ntasks_per_node} {new_script_path}"
        subprocess.check_output([f"{submit_cmd}"], shell=True)
        print(f"Training job submitted with command: \n{submit_cmd}")
        return None