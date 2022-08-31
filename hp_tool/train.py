import os
import shutil
import subprocess

from omegaconf import OmegaConf


def run_training(file_name, bignlp_scripts_path, model_name, results_dir, cfg):
    """
    Main function to launch a training job, with the config given in cfg.
    """
    training_container = cfg.get("training_container")
    data_dir = cfg.get("data_dir")
    cluster_cfg = cfg.get("cluster")
    bignlp_ci = f"BIGNLP_CI=1" if bool(os.getenv('BIGNLP_CI')) else ""
    
    # Copy cluster config to bignlp-scripts.
    dst = os.path.join(bignlp_scripts_path, "conf/cluster/bcm.yaml")
    with open(dst, "w") as f:
        OmegaConf.save(config=cluster_cfg, f=f)
    print(f"Cluster config copied to {dst}")

    main_path = os.path.join(bignlp_scripts_path, "main.py")
    file_name = file_name.replace('.yaml', '')
    cmd = f"HYDRA_FULL_ERROR=1 {bignlp_ci} python3 {main_path} training={model_name}/{file_name} base_results_dir={results_dir} container={training_container} stages=[training] bignlp_path={bignlp_scripts_path} data_dir={data_dir} training.exp_manager.create_checkpoint_callback=False "
    job_output = subprocess.check_output([cmd], shell=True).decode("utf-8")
    job_id = job_output.split(" ")[-1]
    print(f"Submitted Training script with job id: {job_id}")
    return job_id
