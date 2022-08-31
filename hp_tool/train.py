import os
import subprocess


def run_training(file_name, bignlp_hp_tool_path, bignlp_scripts_path, model_name, results_dir, cfg):
    """
    Main function to launch a training job, with the config given in cfg.
    """
    training_container = cfg.get("training_container")
    bignlp_path = cfg.get("bignlp_scripts_path")
    data_dir = cfg.get("data_dir")
    bignlp_ci = f"BIGNLP_CI=1" if bool(os.getenv('BIGNLP_CI')) else ""

    main_path = os.path.join(bignlp_scripts_path, "main.py")
    file_name = file_name.replace('.yaml', '')
    cmd = f"HYDRA_FULL_ERROR=1 {bignlp_ci} python3 {main_path} training={model_name}/{file_name} base_results_dir={results_dir} container={training_container} stages=[training] bignlp_path={bignlp_path} data_dir={data_dir} training.exp_manager.create_checkpoint_callback=False "
    job_output = subprocess.check_output([cmd], shell=True).decode("utf-8")
    job_id = job_output.split(" ")[-1]
    print(f"Submitted Training script with job id: {job_id}")
    return job_id
