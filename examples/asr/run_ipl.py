import os
import copy
import subprocess
from omegaconf import OmegaConf
from typing import Dict, Any
from pathlib import Path
from omegaconf import OmegaConf, open_dict
from pathlib import Path

# import nemo_run as run
from omegaconf import OmegaConf, open_dict
from nemo.collections.asr.parts.utils.run_ipl_utils import *
from nemo.core.config import hydra_runner
from nemo.utils import logging
import glob 
import torch 
from omegaconf import  OmegaConf, open_dict
from omegaconf import OmegaConf, open_dict


def check_training_finished(log_dir):
    """
    Searches to see ig lightning finished training .
    Parameters:
        log_dir (str): Directory where logs are stored.
    """
    if not os.path.exists(log_dir):
        print(f"Log directory '{log_dir}' does not exist.")
        return

    log_pattern = os.path.join(log_dir, f"lightning_logs.txt")
    command = f"grep -ri '`Trainer.fit` stopped:' {log_pattern}"

    result = subprocess.run(
        command, shell=True, capture_output=True, text=True
    )
    if result.stdout:
        print("Stopping reasons found:")
        print(result.stdout)
        return True
    else:
        print("No stopping reasons found in the logs.")
        return False


def get_command_for_inference(
    inference_config: str,
    inference_config_dir: Union[str, Path],
    p_cache: float,
    checkpoint: str
) -> Tuple[str, List[str], List[str]]:
    """
    Generates the command string for running speech inference with transcribe_speech_parallel.

    Args:
        inference_config (str): Path to the base inference configuration file.
        inference_config_dir (Union[str, Path]): Directory to store temporary modified configurations.
        p_cache (float): Proportion of the dataset to be cached for pseudo-labeling.
        checkpoint (str): Path to the model checkpoint to use for inference.

    Returns:
        Tuple[str, List[str], List[str]]:
            - The command string to execute inference for all specified manifests.
            - List of output directories corresponding to each manifest.
            - List of completed full pass transcribed manifest paths, if any.
    """
    """"""
    manifests, tarr_audio_files = separate_multiple_transcriptions(inference_config)
    num_gpus = torch.cuda.device_count()
    output_dirs = []
    cmd = ""
    for i in range(len(manifests)):
        output_dir = os.path.dirname(manifests[i])
        output_dirs.append(output_dir)

        base_cfg = OmegaConf.load(inference_config)
        temp_config_dir = Path(str(inference_config_dir) + "/temp_configs").absolute()
        os.makedirs(temp_config_dir, exist_ok=True)
        modified_cfg = copy.deepcopy(base_cfg)

        # Check if we need to run inference on the whole set or update part of it
        full_pass_done = glob.glob(os.path.join(output_dir, 'transcribed_manifest*'))
        if full_pass_done:
            number_of_files = count_files_for_pseudo_labeling(manifests[i], bool(tarr_audio_files))
            limit_predict_batches = int((number_of_files * p_cache) / (modified_cfg.predict_ds.batch_size * num_gpus))
            OmegaConf.update(modified_cfg, "trainer.limit_predict_batches", limit_predict_batches)

        # Replace OmegaConf updates with simple assignments
        OmegaConf.update(modified_cfg, "output_path", output_dir)
        OmegaConf.update(modified_cfg, "predict_ds.manifest_filepath", manifests[i])
        if tarr_audio_files:
            OmegaConf.update(modified_cfg, "predict_ds.tarred_audio_filepaths", tarr_audio_files[i])
        OmegaConf.update(modified_cfg, "model", checkpoint)


        temp_config_file = os.path.join(temp_config_dir, f"modified_config_{i}.yaml")
        OmegaConf.save(modified_cfg, temp_config_file)
        cmd += f"python transcribe_speech_parallel.py --config-path {temp_config_dir} --config-name modified_config_{i}.yaml && "

    # Remove trailing '&&' from the final command string
    cmd = cmd.rstrip(" &&")

    print(f"Inference command: {cmd}")
    return cmd, output_dirs, full_pass_done

def merge_configs(script_config_path, run_config):
    # Load the configurations
    script_config = OmegaConf.load(script_config_path)

    print(run_config)

    # Keep track of the original keys in script_config
    original_script_keys = set(script_config.keys())

    # Merge only the 'training' part of run_config with script_config
    result = OmegaConf.merge(script_config, run_config)

    with open_dict(result):
        for k in run_config.keys():
            if k in result and k not in original_script_keys:
                del result[k]

    def check_missing_values(cfg):
        if hasattr(cfg, 'items'):
            for k, v in cfg.items():
                if hasattr(v, 'items'):
                    check_missing_values(v)
                elif v == '???':
                    raise ValueError(f"Missing value for key {k} in the config file")

    check_missing_values(result)
    result.exp_manager.resume_if_exists = True
    return result

def get_execution_script(cluster_script_path: str, config_name: str, config_path: str) -> str:
    """
    Constructs a command string to execute a training with the specified configuration.

    Args:
        cluster_script_path (str): Path to the cluster script to be executed.
        config_name (str): Name of the configuration file or object to be passed as a parameter.
        config_path (str): Path to the directory where the configuration resides.

    Returns:
        str: A formatted command string ready for execution.
    """
    # Create the command to run the script
    cmd = """
        cd {cluster_script_dir} && \
        python {cluster_script_path} --config-path {config_path} --config-name "{config_name}" 
    """
    print("in get_execution_script")
    format_dict = dict(
        cluster_script_dir=os.path.dirname(cluster_script_path),
        cluster_script_path=os.path.basename(cluster_script_path),
        config_path=config_path,
        config_name=config_name,
    )
    cmd = cmd.format(**format_dict)
    print(f"format cmd {cmd}")

    return cmd

def find_checkpoint_dir(base_path):
    """
    Find the 'checkpoints' folder in the directory structure.
    Parameters:
        base_path (str): The base directory path to search from.

    """
    for root, dirs, files in os.walk(base_path):
        for dir_name in dirs:
            if dir_name == "checkpoints":
                return os.path.join(root, dir_name), root
    return None

@hydra_runner(config_path='./', config_name='run_ipl')
def main(run_config):

    script_config = run_config.script_config
    script_path = run_config.script
    inference_config = run_config.inference_config
    inference_config_dir =  Path(run_config.inference_config_dir).absolute()
    script_config_path = os.path.dirname(Path(script_config).absolute())
    inference_config =os.path.join(inference_config_dir,inference_config )

    add_pl_datasets = True
    merged_config = merge_configs(script_config, run_config)
    config_filepath = os.path.join(script_config_path, "update_script_config.yaml")
    subprocess.run(f"touch {config_filepath}", shell=True)

    subprocess.run(f"echo '{OmegaConf.to_yaml(merged_config)}' > {config_filepath}", shell= True)
    training_command = get_execution_script(script_path, "update_script_config.yaml", script_config_path)
    subprocess.run(training_command, shell=True)
    
    checkpoint_path, logs_dir = find_checkpoint_dir(os.path.join(merged_config.exp_manager.exp_dir, merged_config.exp_manager.name))
    checkpoint = os.path.join(checkpoint_path, merged_config.exp_manager.name + ".nemo")
    while True:
        should_terminate = check_training_finished(logs_dir)
        if should_terminate:
            break
        cmd, output_dirs, full_pass_done  = get_command_for_inference(inference_config, inference_config_dir,0.5, checkpoint)
        subprocess.run(cmd, shell=True) 

        if not full_pass_done:
            if merged_config.model.train_ds.is_tarred:
                all_manifest_filepaths = create_transcribed_shard_manifests(output_dirs)
            else:
                all_manifest_filepaths = create_transcribed_manifests(output_dirs)
        else:
            if merged_config.model.train_ds.is_tarred:
                all_manifest_filepaths = write_sampled_shard_transcriptions(output_dirs)
            else:
                all_manifest_filepaths = write_sampled_transcriptions(output_dirs)
        if add_pl_datasets:
            base_cfg = OmegaConf.load(inference_config)
            merged_config = update_training_sets(merged_config, all_manifest_filepaths, base_cfg.predict_ds.get("tarred_audio_filepaths", None))
            add_pl_datasets = False

        subprocess.run(f"echo '{OmegaConf.to_yaml(merged_config)}' > {config_filepath}", shell= True)
        training_command = get_execution_script(script_path, "update_script_config.yaml", script_config_path)
        
        subprocess.run(training_command, shell=True)

if __name__ == '__main__':
    main()
