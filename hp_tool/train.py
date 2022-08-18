import os
import subprocess


def run_training(file_name, bignlp_hp_tool_path, bignlp_scripts_path, model_name, results_dir):
    """
    Main function to launch a training job, with the config given in cfg.
    """
    main_path = os.path.join(bignlp_scripts_path, "main.py")

    config_path = os.path.join(bignlp_hp_tool_path, "bignlp_conf")
    file_name = file_name.replace('.yaml', '')
    cmd = f"python3 {main_path} --config-path={config_path} --config-name=config.yaml training={model_name}/{file_name} base_results_dir={results_dir}"
    print(cmd)
    job_id = subprocess.check_output([cmd], shell=True).decode("utf-8")
    print(f"Submitted Training script with job id: {job_id}")
    return job_id
