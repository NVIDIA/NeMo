import os
import yaml
import hydra
import subprocess


def create_slurm_file(
    new_script_path,
    cmd,
    job_name,
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
    account=None,
):
    """
    Creates a slurm file to launch a training job.
    """
    with open(new_script_path, "w") as f:
        f.writelines("#!/bin/bash\n")
        f.writelines(f"#SBATCH --nodes={nodes}\n")
        f.writelines(f"#SBATCH --ntasks-per-node={ntasks_per_node}\n")
        if gpus_per_task is not None:
            f.writelines(f"#SBATCH --gpus-per-task={gpus_per_task}\n")
        if dependency is not None:
            if dependency != "singleton":
                dependency = f"afterany:{dependency}"
            f.writelines(f"#SBATCH --dependency={dependency}\n")
        f.writelines(f"#SBATCH -p {partition}\n")
        if account is not None:
            f.writelines(f"#SBATCH -A {account}\n")
        f.writelines(f"#SBATCH --job-name={job_name}\n")
        f.writelines(f"#SBATCH --mem={mem}\n")
        if exclusive:
            f.writelines("#SBATCH --exclusive\n")
        if overcommit:
            f.writelines("#SBATCH --overcommit\n")
        f.writelines(f"#SBATCH --time={time}\n\n")
        f.writelines(f'srun {flags} sh -c "{cmd}"\n\n')
        f.writelines("set +x\n")

def search_inference_config(base_cfg, cfg):
    hp_cfg = cfg.search_config
    cluster_cfg = cfg.cluster
    inference_settings_cfg = hp_cfg.inference_settings

    bignlp_hp_tool_path = cfg.bignlp_hp_tool_path
    input_seq_len = inference_settings_cfg.input_seq_len
    output_seq_len = inference_settings_cfg.output_seq_len
    top_n = inference_settings_cfg.top_n
    vocab_size = inference_settings_cfg.vocab_size
    start_id = inference_settings_cfg.start_id
    end_id = inference_settings_cfg.end_id
    tensor_parallel_sizes = [
        str(x) for x in inference_settings_cfg.tensor_parallel_sizes
    ]
    pipeline_parallel_sizes = [
        str(x) for x in inference_settings_cfg.pipeline_parallel_sizes
    ]
    max_batch_sizes = [str(x) for x in inference_settings_cfg.max_batch_sizes]

    inference_profile_path = "/opt/bignlp/bignlp-scripts/bignlp/infer_scripts/profile_model_with_random_weights.py"
    cluster_config_path = os.path.join(bignlp_hp_tool_path, "conf/cluster/bcm.yaml")
    navigator_config_path = "/opt/bignlp/bignlp-scripts/conf/inference/profile_offline.yaml"
    model_spec_dir = os.path.join(bignlp_hp_tool_path, "tmp_test_model")
    if not os.path.isdir(model_spec_dir):
        os.mkdir(model_spec_dir)
    model_spec_path = os.path.join(model_spec_dir, "meta.yaml")

    # Create model_spec and save to yaml
    model_spec = {}
    model_spec["decoder_layers"] = base_cfg["model"]["num_layers"]
    model_spec["head_num"] = base_cfg["model"]["num_attention_heads"]
    model_spec["size_per_head"] = int(
        base_cfg["model"]["hidden_size"] / base_cfg["model"]["num_attention_heads"]
    )
    model_spec["inter_size"] = int(
        model_spec["size_per_head"] * model_spec["head_num"] * 4
    )
    model_spec["tensor_para_size"] = 8  # Value is ignored, but field is necessary
    model_spec["vocab_size"] = vocab_size
    model_spec["start_id"] = start_id
    model_spec["end_id"] = end_id

    with open(model_spec_path, "w") as f:
        yaml.dump(model_spec, f)

    # Launch profile script
    inference_profile_cmd = (
        f"python3 -u {inference_profile_path} "
        f"--cluster-config-path {cluster_config_path} "
        f"--navigator-config-path {navigator_config_path} "
        f"--model-path {model_spec_dir} "
        f"--model-name test_model "
        f"--tensor-parallel-sizes {' '.join(tensor_parallel_sizes)} "
        f"--pipeline-parallel-sizes {' '.join(pipeline_parallel_sizes)} "
        f"--input-output-lengths {input_seq_len},{output_seq_len} "
        f"--max-batch-sizes {' '.join(max_batch_sizes)} "
        f"--top-n-configs {top_n} "
        f"-v "
    )

    # Generates infer_workspace
    subprocess.check_output([inference_profile_cmd], shell=True)

    # Clean up test files
    os.remove(model_spec_path)
    os.rmdir(model_spec_dir)
