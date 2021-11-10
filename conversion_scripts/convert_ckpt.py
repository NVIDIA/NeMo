import os
import sys
import re
import glob
import hydra
import torch
import shutil


@hydra.main(config_path="../conf", config_name="config")
def main(cfg):
    # Read Config
    bignlp_path = cfg.get("bignlp_path")
    convert_cfg = cfg.get("conversion")
    run_cfg = convert_cfg.get("run")
    model_cfg = convert_cfg.get("model")

    # Modify config
    args = sys.argv[1:]
    for index, arg in enumerate(args):
        k, v = arg.split("=")
        if arg.startswith("conversion."):
            k.lstrip("conversion.")
            if k.startswith("run."):
                k.lstrip("run.")
                run_cfg[k] = v
            if k.startswith("model."):
                k.lstrip("model.")
                model_cfg[k] = v

    # Model parameters
    checkpoint_folder = model_cfg.get("checkpoint_folder")
    checkpoint_name = model_cfg.get("checkpoint_name")
    tensor_model_parallel_size = model_cfg.get("tensor_model_parallel_size")
    vocab_file = model_cfg.get("vocab_file")
    merge_file = model_cfg.get("merge_file")

    if checkpoint_name == "latest":
        if tensor_model_parallel_size > 1:
            checkpoint_dir = os.path.join(checkpoint_folder, "mp_rank_00")
        else:
            checkpoint_dir = os.path.join(checkpoint_folder)
        checkpoint_list = glob.glob(checkpoint_dir + '/*.ckpt')
        latest_checkpoint = max(checkpoint_list, key=os.path.getctime)
        checkpoint_name = os.path.basename(latest_checkpoint)

    if tensor_model_parallel_size > 1:
        checkpoint = os.path.join(checkpoint_folder, "mp_rank_00", checkpoint_name)
    else:
        checkpoint = os.path.join(checkpoint_folder, checkpoint_name)
    checkpoint_list = glob.glob(checkpoint)
    if len(checkpoint_list) > 1:
        raise ValueError("Too many checkpoints fit the checkpoint name pattern in convert.yaml.")
    if len(checkpoint_list) == 0:
        raise ValueError("No checkpoint found with the checkpoint name pattern in convert.yaml.")
    checkpoint_name = os.path.basename(checkpoint_list[0])

    load = torch.load(checkpoint, map_location="cpu")
    ckpt_conf = load["hyper_parameters"]
    ckpt_merge_file = ckpt_conf.tokenizer.merge_file
    ckpt_vocab_file = ckpt_conf.tokenizer.vocab_file
    if merge_file != ckpt_merge_file:
        os.makedirs(os.path.dirname(ckpt_merge_file), exist_ok=True)
        shutil.copy2(merge_file, ckpt_merge_file)
    if vocab_file != ckpt_vocab_file:
        os.makedirs(os.path.dirname(ckpt_vocab_file), exist_ok=True)
        shutil.copy2(vocab_file, ckpt_vocab_file)
    del load

    # Run parameters
    name = run_cfg.get("name")
    nemo_file_name = run_cfg.get("nemo_file_name")
    log_dir = os.path.join(bignlp_path, run_cfg.get("output_path"), name)
    os.makedirs(log_dir, exist_ok=True)
    nemo_file_path = os.path.join(log_dir, nemo_file_name)

    code_path = "/opt/bignlp/NeMo/examples/nlp/language_modeling/megatron_gpt_ckpt_to_nemo.py"
    cmd = f"python -u {code_path} " \
          f"--checkpoint_folder {checkpoint_folder} " \
          f"--checkpoint_name {checkpoint_name} " \
          f"--nemo_file_path {nemo_file_path} " \
          f"--tensor_model_parallel_size {tensor_model_parallel_size} "

    os.system(f"{cmd}")


if __name__ == "__main__":
    main()
