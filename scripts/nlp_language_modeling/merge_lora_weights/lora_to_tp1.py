import argparse
import shutil

import torch


def load_lora(lora_checkpoint_dir, tp):
    lora_state_dict = {}

    for i in range(tp):
        ckpt_file = f"{lora_checkpoint_dir}/mp_rank_0{i}/model_weights.ckpt"
        loaded_state_dict = torch.load(ckpt_file, map_location=torch.device('cpu'))
        lora_state_dict[i] = loaded_state_dict
    return lora_state_dict


def to_tp1(lora_state_dict):
    tp = len(lora_state_dict)
    target_state_dict = {}
    for key in lora_state_dict[0].keys():
        for rank in range(tp):
            wt_lora = torch.cat([lora_state_dict[rank][key] for _tp in range(tp)], dim=0)
            target_state_dict[key] = wt_lora
    return target_state_dict


def save_tp1(state_dict_tp1, target_lora_checkpoint_dir, lora_checkpoint_dir):
    torch.save(state_dict_tp1, f'{target_lora_checkpoint_dir}/model_weights.ckpt')
    shutil.copy(f"{lora_checkpoint_dir}/model_config.yaml", f"{target_lora_checkpoint_dir}/model_config.yaml")


def main():
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--lora_checkpoint_dir",
        type=str,
        required=True,
        help="Path to the directory containing unpacked lora checkpoints.",
    )
    parser.add_argument(
        "--target_lora_checkpoint_dir",
        type=str,
        required=True,
        help="Path to the output directory containing unpacked lora checkpoints.",
    )
    parser.add_argument(
        "--tp", type=int, required=True, help="Tensor parallelism for the input lora checkpoint",
    )
    args = parser.parse_args()

    state_dict = load_lora(lora_checkpoint_dir=args.lora_checkpoint_dir, tp=args.tp)
    state_dict_tp1 = to_tp1(state_dict)
    save_tp1(state_dict_tp1, args.target_lora_checkpoint_dir, args.lora_checkpoint_dir)


if __name__ == "__main__":
    main()
