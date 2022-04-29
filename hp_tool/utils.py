import os
import time
import copy

import yaml
import omegaconf


def _calculate_model_size(
    vocab_size=None,
    seq_length=None,
    hidden_size=None,
    num_layers=None,
    ffn_size=None,
    kv_channels=None,
    att_heads=None,
    model_name="gpt3",
):
    if model_name == "gpt3":
        model_size = (
            12
            * num_layers
            * hidden_size**2
            * (
                1
                + (13 / (12 * hidden_size))
                + ((vocab_size + seq_length) / (12 * num_layers * hidden_size))
            )
            / 1e9
        )
    elif model_name in ["t5", "mt5"]:
        # 2 L F + 3 L P + H (2 + 4 L F + L (21 + 12 P) + 1 S + 1 V)
        proj_size = att_heads * kv_channels
        model_size = (
            2 * num_layers * 1.5 * ffn_size
            + 3 * num_layers * proj_size
            + hidden_size
            * (
                2
                + 4 * num_layers * 1.5 * ffn_size
                + num_layers * (21 + 12 * proj_size)
                + seq_length
                + vocab_size
            )
        ) / 1e9
    return model_size


def calculate_model_size_params(
    model_size_in_b, vocab_size=51200, seq_length=2048, model_name="gpt3"
):
    """Calculates the parameters that affect model_size: hidden size, attention heads,
    KV channels, and FFN size. It also calculates the learning rate.

    Arguments:
        model_size_in_b: float, number of parameters in the desired model config, in billions.
        seq_length: int, sequence length to be used during training.
        vocab_size: int, size of the vocabulary to use for training.
        model_name: str, name of the model to be trained, i.e. gpt3, t5, mt5...
    Output:

    """
    ffn, kv = None, None  # Only needed for some models.
    if model_name == "gpt3":
        if model_size_in_b < 0.25:
            hs, att_h, lr = 768, 12, 6e-4
        elif model_size_in_b < 0.5:
            hs, att_h, lr = 1024, 16, 3e-4
        elif model_size_in_b < 1:
            hs, att_h, lr = 1536, 16, 2.5e-4
        elif model_size_in_b < 2:
            hs, att_h, lr = 2048, 16, 2e-4
        elif model_size_in_b < 3:
            hs, att_h, lr = 2560, 32, 1.6e-4
        elif model_size_in_b < 4.5:
            hs, att_h, lr = 3072, 32, 1.4e-4
        elif model_size_in_b < 8:
            hs, att_h, lr = 4096, 32, 1.2e-4
        elif model_size_in_b < 15:
            hs, att_h, lr = 5120, 40, 1e-4
        elif model_size_in_b < 25:
            hs, att_h, lr = 6144, 48, 1e-4
        elif model_size_in_b < 52:
            hs, att_h, lr = 8192, 64, 0.8e-4
        elif model_size_in_b < 105:
            hs, att_h, lr = 10240, 80, 0.7e-4
        elif model_size_in_b < 205:
            hs, att_h, lr = 12288, 96, 0.6e-4
        elif model_size_in_b < 405:
            hs, att_h, lr = 14336, 128, 0.5e-4
        elif model_size_in_b < 805:
            hs, att_h, lr = 20480, 128, 0.4e-4
        elif model_size_in_b < 1105:
            hs, att_h, lr = 24576, 128, 0.3e-4
        else:
            raise ValueError("Model_size for GPT-3 must be smaller than 1.1T parameters.")
    elif model_name == "t5":
        kv, lr = 64, 1e-4
        if model_size_in_b < 0.1:
            hs, att_h, ffn = 512, 6, 1024
        elif model_size_in_b < 0.4:
            hs, att_h, ffn = 768, 12, 2048
        elif model_size_in_b < 1:
            hs, att_h, ffn = 1024, 16, 2816
        elif model_size_in_b < 5:
            hs, att_h, ffn = 2048, 32, 5120
        elif model_size_in_b < 15:
            hs, att_h, ffn = 4096, 64, 10240
        elif model_size_in_b < 25.9:
            hs, att_h, ffn = 5120, 80, 10880
        elif model_size_in_b < 43.0:
            hs, att_h, ffn = 6144, 96, 10880
        else:
            raise ValueError("Model_size for T5 must be smaller than 43B parameters.")
    elif model_name == "mt5":
        kv, lr = 64, 1e-4
        if model_size_in_b < 0.25:
            hs, att_h, ffn = 512, 6, 1024
        elif model_size_in_b < 0.5:
            hs, att_h, ffn = 768, 12, 2048
        elif model_size_in_b < 1.2:
            hs, att_h, ffn = 1024, 16, 2816
        elif model_size_in_b < 5:
            hs, att_h, ffn = 2048, 32, 5120
        elif model_size_in_b < 15:
            hs, att_h, ffn = 4096, 64, 10240
        elif model_size_in_b < 25.9:
            hs, att_h, ffn = 5120, 80, 10880
        elif model_size_in_b < 43.0:
            hs, att_h, ffn = 6144, 96, 10880
        else:
            raise ValueError("Model_size for mT5 must be smaller than 43B parameters.")

    else:
        raise NotImplementedError("Model name is not valid.")

    # Try powers of 2
    margin = 0.01
    for attempt in range(0, 10):
        for layers in (2**p for p in range(1, 10)):
            out_size = _calculate_model_size(
                vocab_size=vocab_size,
                seq_length=seq_length,
                hidden_size=hs,
                num_layers=layers,
                ffn_size=ffn,
                kv_channels=kv,
                att_heads=att_h,
                model_name=model_name,
            )
            if model_size_in_b * (1.0 - margin) < out_size < model_size_in_b * (1.0 + margin):
                return layers, hs, att_h, ffn, kv, lr
        margin += 0.01  # Double margin of acceptable model sizes.

    # Try multiples of 16
    margin = 0.01
    for attempt in range(0, 6):
        for layers in range(16, 201, 16):
            out_size = _calculate_model_size(
                vocab_size=vocab_size,
                seq_length=seq_length,
                hidden_size=hs,
                num_layers=layers,
                ffn_size=ffn,
                kv_channels=kv,
                att_heads=att_h,
                model_name=model_name,
            )
            if model_size_in_b * (1.0 - margin) < out_size < model_size_in_b * (1.0 + margin):
                return layers, hs, att_h, ffn, kv, lr
        margin += 0.01  # Double margin of acceptable model sizes.

    # Try multiples of 2
    margin = 0.01
    for attempt in range(0, 6):
        for layers in range(2, 201, 2):
            out_size = _calculate_model_size(
                vocab_size=vocab_size,
                seq_length=seq_length,
                hidden_size=hs,
                num_layers=layers,
                ffn_size=ffn,
                kv_channels=kv,
                att_heads=att_h,
                model_name=model_name,
            )
            if model_size_in_b * (1.0 - margin) < out_size < model_size_in_b * (1.0 + margin):
                return layers, hs, att_h, ffn, kv, lr
        margin += 0.01  # Double margin of acceptable model sizes.

    # Try multiples of 5
    margin = 0.01
    for attempt in range(0, 6):
        for layers in range(5, 201, 5):
            out_size = _calculate_model_size(
                vocab_size=vocab_size,
                seq_length=seq_length,
                hidden_size=hs,
                num_layers=layers,
                ffn_size=ffn,
                kv_channels=kv,
                att_heads=att_h,
                model_name=model_name,
            )
            if model_size_in_b * (1.0 - margin) < out_size < model_size_in_b * (1.0 + margin):
                return layers, hs, att_h, ffn, kv, lr
        margin += 0.01  # Double margin of acceptable model sizes.

    # Try any valid number
    margin = 0.01
    for attempt in range(0, 10):
        for layers in range(1, 200):
            out_size = _calculate_model_size(
                vocab_size=vocab_size,
                seq_length=seq_length,
                hidden_size=hs,
                num_layers=layers,
                ffn_size=ffn,
                kv_channels=kv,
                att_heads=att_h,
                model_name=model_name,
            )
            if model_size_in_b * (1.0 - margin) < out_size < model_size_in_b * (1.0 + margin):
                return layers, hs, att_h, ffn, kv, lr
        margin += 0.01  # Double margin of acceptable model sizes.
    raise Exception("Number of layers not found, config is not possible.")


def estimate_training_time(model_size=None, num_tokens=None, num_gpus=None, tflops_per_gpu=None):
    training_time = (8 * num_tokens * 1e9 * model_size * 1e9) / (
        num_gpus * tflops_per_gpu * 1e12 * 3600 * 24
    )
    return training_time


def generic_base_config(cfg, model_name="gpt3"):
    with open(f"{cfg.bignlp_hp_tool_path}/base_configs/{model_name}.yaml") as f:
        base_cfg = yaml.safe_load(f)
    return base_cfg


def modify_cfg(base_cfg, act, tp, pp, mbs, max_minutes, max_pp):
    new_cfg = copy.deepcopy(base_cfg)
    new_cfg["model"]["activations_checkpoint_num_layers"] = act
    new_cfg["model"]["tensor_model_parallel_size"] = tp
    new_cfg["model"]["pipeline_model_parallel_size"] = pp
    new_cfg["model"]["micro_batch_size"] = mbs
    att_heads = new_cfg["model"]["num_attention_heads"]
    num_layers = new_cfg["model"]["num_layers"]

    # gbs = mbs * num_gpus * accumulate_grad_batches / (tp * pp)
    num_gpus = new_cfg["trainer"]["num_nodes"] * new_cfg["trainer"]["devices"]
    gbs = new_cfg["model"]["global_batch_size"]

    mod_gbs = gbs % (mbs * num_gpus / (tp * pp))
    mod_att_heads = att_heads % tp
    mod_layers = num_layers % pp
    if mod_gbs == 0 and mod_att_heads == 0 and mod_layers == 0:
        # Valid config
        new_cfg["trainer"]["num_nodes"] = max_pp  # Necessary for short single-node test.
        days = max_minutes // 3600
        hours = (max_minutes % 3600) // 60
        mins = (max_minutes % 3600) % 60
        new_cfg["run"]["time_limit"] = f"{days}-{hours}:{mins}:00"
        new_cfg["run"][
            "name"
        ] = f"{new_cfg['run']['name']}_tp_{tp}_pp_{pp}_mbs_{mbs}_act_ckpt_{act}"
        print(
            f"Valid config: GBS={gbs}, MBS={mbs}, TP={tp}, PP={pp}, act_ckpt_layers={act}. Adding to directory."
        )
        return new_cfg
    return None


def create_slurm_file(
    new_script_path,
    train_cmd,
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
        f.writelines(f'srun {flags} sh -c "{train_cmd}"\n\n')
        f.writelines("set +x\n")


def convert_to_cli(cfg):
    result = ""
    for k, v in cfg.items():
        if isinstance(v, omegaconf.dictconfig.DictConfig):
            output = convert_to_cli(v).split(" ")
            result += " ".join([f"{k}.{x}" for x in output if x != ""]) + " "
        elif isinstance(v, omegaconf.listconfig.ListConfig):
            if k == "data_prefix":
                if v is None:
                    v = "null"
                else:
                    v = [x for x in v]  # Needed because of lazy omegaconf interpolation.
            result += f"{k}={str(v).replace(' ', '')} "
        elif isinstance(v, str) and "{" in v:
            continue
        elif k in ["splits_string", "file_numbers", "languages"]:
            result += f"{k}=\\'{v}\\' "
        elif k == "checkpoint_name":
            v = v.replace("=", "\=")
            result += f"{k}='{v}' "
        elif k == "container":
            continue
        else:
            result += f"{k}={convert_to_null(v)} "
    return result


def convert_to_null(val):
    if val is None:
        return "null"
    return val
