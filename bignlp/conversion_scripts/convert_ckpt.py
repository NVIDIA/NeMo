import os
import sys
import glob
import hydra
import torch
import shutil
from nemo.utils.get_rank import is_global_rank_zero


@hydra.main(config_path="../../conf", config_name="config")
def main(cfg):
    # Read Config
    bignlp_path = cfg.bignlp_path
    convert_cfg = cfg.conversion
    run_cfg = convert_cfg.run
    model_cfg = convert_cfg.model

    # Model parameters
    checkpoint_folder = model_cfg.checkpoint_folder
    checkpoint_name = model_cfg.checkpoint_name
    tensor_model_parallel_size = model_cfg.tensor_model_parallel_size
    vocab_file = model_cfg.vocab_file
    merge_file = model_cfg.merge_file

    # Checkpoint finding
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
    checkpoint = checkpoint_list[0]

    load = torch.load(checkpoint, map_location="cpu")
    ckpt_conf = load["hyper_parameters"]
    ckpt_conf = ckpt_conf["cfg"] if "cfg" in ckpt_conf else ckpt_conf
    ckpt_merge_file = ckpt_conf.tokenizer.merge_file
    ckpt_vocab_file = ckpt_conf.tokenizer.vocab_file
    if ckpt_merge_file is not None and merge_file != ckpt_merge_file and is_global_rank_zero():
        os.makedirs(os.path.dirname(ckpt_merge_file), exist_ok=True)
        shutil.copy2(merge_file, ckpt_merge_file)
    if ckpt_vocab_file is not None and vocab_file != ckpt_vocab_file and is_global_rank_zero():
        os.makedirs(os.path.dirname(ckpt_vocab_file), exist_ok=True)
        shutil.copy2(vocab_file, ckpt_vocab_file)
    del load

    # Run parameters
    nemo_file_name = run_cfg.nemo_file_name
    output_dir = run_cfg.output_path
    os.makedirs(output_dir, exist_ok=True)
    nemo_file_path = os.path.join(output_dir, nemo_file_name)

    code_path = "/opt/bignlp/NeMo/examples/nlp/language_modeling/megatron_ckpt_to_nemo.py"
    cmd = f"python -u {code_path} " \
          f"--checkpoint_folder {checkpoint_folder} " \
          f"--checkpoint_name {checkpoint_name} " \
          f"--nemo_file_path {nemo_file_path} " \
          f"--tensor_model_parallel_size {tensor_model_parallel_size} " \
          f"--model_type gpt "

    os.system(cmd)


if __name__ == "__main__":
    main()
