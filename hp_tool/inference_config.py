import os
import yaml
import hydra
import subprocess


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
    tensor_parallel_sizes = [str(x) for x in inference_settings_cfg.tensor_parallel_sizes]
    pipeline_parallel_sizes = [str(x) for x in inference_settings_cfg.pipeline_parallel_sizes]
    max_batch_sizes = [str(x) for x in inference_settings_cfg.max_batch_sizes]

    inference_profile_path = os.path.join(
        bignlp_hp_tool_path,
        "BigNLP-Inference-Scripts/bignlp/infer_scripts/profile_model_with_random_weights.py",
    )
    cluster_config_path = os.path.join(bignlp_hp_tool_path, "conf/cluster/bcm.yaml")
    navigator_config_path = os.path.join(
        bignlp_hp_tool_path, "BigNLP-Inference-Scripts/conf/inference/profile_offline.yaml"
    )
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
    model_spec["inter_size"] = int(model_spec["size_per_head"] * model_spec["head_num"] * 4)
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
        f"--workspace-path {bignlp_hp_tool_path + '/workspace_test'} "
    )

    # Generates infer_workspace
    subprocess.check_output([inference_profile_cmd], shell=True)

    # Clean up test files
    os.remove(model_spec_path)
    os.rmdir(model_spec_dir)
