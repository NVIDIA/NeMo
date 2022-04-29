import sys
import os
import glob
import time

import hydra
from argparse import ArgumentParser
from omegaconf import OmegaConf
from nemo.utils.get_rank import is_global_rank_zero


def _inject_model_parallel_rank(filepath, tensor_model_parallel_size=1, pipeline_model_parallel_size=1):
    """
    Injects tensor/pipeline model parallel ranks into the filepath.
    Does nothing if not using model parallelism.
    """
    tensor_model_parallel_rank = pipeline_model_parallel_rank = 0
    if tensor_model_parallel_size > 1 or pipeline_model_parallel_size > 1:
        # filepath needs to be updated to include mp_rank
        dirname = os.path.dirname(filepath)
        basename = os.path.basename(filepath)
        if pipeline_model_parallel_size is None or pipeline_model_parallel_size == 1:
            filepath = f'{dirname}/mp_rank_{tensor_model_parallel_rank:02d}/{basename}'
        else:
            filepath = f'{dirname}/tp_rank_{tensor_model_parallel_rank:02d}_pp_rank_{pipeline_model_parallel_rank:03d}/{basename}'
        return filepath
    else:
        return filepath


def get_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--checkpoint_folder",
        type=str,
        default=None,
        required=True,
        help="Path to PTL checkpoints saved during training. Ex: /raid/nemo_experiments/megatron_gpt/checkpoints",
    )
    parser.add_argument(
        "--checkpoint_name",
        type=str,
        default=None,
        required=True,
        help="Name of checkpoint to be used. Ex: megatron_gpt--val_loss=6.34-step=649-last.ckpt",
    )

    parser.add_argument(
        "--hparams_file",
        type=str,
        default=None,
        required=False,
        help="Path config for restoring. It's created during training and may need to be modified during restore if restore environment is different than training. Ex: /raid/nemo_experiments/megatron_gpt/hparams.yaml",
    )
    parser.add_argument("--nemo_file_path", type=str, default=None, required=True, help="Path to output .nemo file.")
    parser.add_argument("--gpus_per_node", type=int, required=True, default=None)
    parser.add_argument("--tensor_model_parallel_size", type=int, required=True, default=None)
    parser.add_argument("--pipeline_model_parallel_size", type=int, required=True, default=None)
    parser.add_argument("--model_type", type=str, required=True, default="gpt", choices=["gpt", "t5", "bert"])
    parser.add_argument("--vocab_file", type=str, default=None, required=False, help="Path to vocab file.")
    parser.add_argument("--merge_file", type=str, default=None, required=False, help="Path to merge file.")
    parser.add_argument("--tokenizer_model", type=str, default=None, required=False, help="Path to sentencepiece tokenizer for mT5.")
    parser.add_argument("--bcp", action="store_true", help="Whether on BCP platform")

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()

    checkpoint_folder = args.checkpoint_folder
    checkpoint_name = args.checkpoint_name
    hparams_file = args.hparams_file
    nemo_file_path = args.nemo_file_path
    gpus_per_node = args.gpus_per_node
    tensor_model_parallel_size = args.tensor_model_parallel_size
    pipeline_model_parallel_size = args.pipeline_model_parallel_size
    model_type = args.model_type
    vocab_file = args.vocab_file
    merge_file = args.merge_file
    tokenizer_model = args.tokenizer_model
    bcp = args.bcp

    # Checkpoint search
    if checkpoint_name == "latest":
        checkpoints = os.path.join(checkpoint_folder, '*.ckpt')
        checkpoints = _inject_model_parallel_rank(checkpoints, tensor_model_parallel_size, pipeline_model_parallel_size)
        checkpoint_list = glob.glob(checkpoints)
        latest_checkpoint = max(checkpoint_list, key=os.path.getctime)
        checkpoint_name = os.path.basename(latest_checkpoint)

    checkpoint = os.path.join(checkpoint_folder, checkpoint_name)
    checkpoint = _inject_model_parallel_rank(checkpoint, tensor_model_parallel_size, pipeline_model_parallel_size)
    checkpoint_list = glob.glob(checkpoint)
    if len(checkpoint_list) > 1:
        raise ValueError("Too many checkpoints fit the checkpoint name pattern in conversion config.")
    if len(checkpoint_list) == 0:
        raise ValueError("No checkpoint found with the checkpoint name pattern in conversion config.")
    checkpoint_name = os.path.basename(checkpoint_list[0])

    # Create hparam override file for vocab and merge
    hparams_override_file = None
    if hparams_file is not None:
        output_path = os.path.dirname(nemo_file_path)
        hparams_override_file = os.path.join(output_path, "hparams_override.yaml")
        conf = OmegaConf.load(hparams_file)
        if vocab_file is not None:
            conf.cfg.tokenizer.vocab_file = vocab_file
        if merge_file is not None:
            conf.cfg.tokenizer.merge_file = merge_file
        if tokenizer_model is not None:
            conf.cfg.tokenizer.model = tokenizer_model

        if is_global_rank_zero():
            with open(hparams_override_file, 'w') as f:
                OmegaConf.save(config=conf, f=f)

        while not os.path.exists(hparams_override_file):
            time.sleep(1)

    code_path = "/opt/bignlp/NeMo/examples/nlp/language_modeling/megatron_ckpt_to_nemo.py"
    args = f"--gpus_per_node={gpus_per_node} " \
           f"--model_type={model_type} " \
           f"--checkpoint_folder={checkpoint_folder} " \
           f"--checkpoint_name={checkpoint_name} " \
           f"--hparams_file={hparams_override_file} " \
           f"--nemo_file_path={nemo_file_path} " \
           f"--tensor_model_parallel_size={tensor_model_parallel_size} " \
           f"--pipeline_model_parallel_size={pipeline_model_parallel_size} "
    args += "--bcp " if bcp else ""

    args = args.replace(" ", " \\\n  ")
    cmd_str = f"python3 -u {code_path} \\\n  {args}"

    if is_global_rank_zero():
        print("************** Converting commands ***********")
        print(f'\n{cmd_str}')
        print("**********************************************\n\n")

    os.system(f"{cmd_str}")
