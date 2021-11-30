# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import shutil
import subprocess
import tempfile
import time
from argparse import ArgumentParser

import torch
from omegaconf import OmegaConf


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

    parser.add_argument("--tensor_model_parallel_size", type=int, required=True, default=None)

    parser.add_argument(
        "--vocab_file",
        type=str,
        required=False,
        default=None,
        help="Custom vocab file path; overwrite current path stored in config",
    )
    parser.add_argument(
        "--merge_file",
        type=str,
        required=False,
        default=None,
        help="Custom merge file path; overwrite current path stored in config",
    )

    args = parser.parse_args()
    return args


def main():
    args = get_args()
    model_config_yaml = "model_config.yaml"
    model_weights_ckpt = "model_weights.ckpt"

    tmp_dir = tempfile.mkdtemp()
    t_gpu_num = args.tensor_model_parallel_size

    start_time = time.time()
    config_yaml = os.path.join(tmp_dir, model_config_yaml)
    if t_gpu_num == 1:
        sources = [os.path.join(args.checkpoint_folder, args.checkpoint_name)]
        model_weights = [os.path.join(tmp_dir, model_weights_ckpt)]
    else:
        sources = [
            os.path.join(args.checkpoint_folder, f"mp_rank_{i:02d}", args.checkpoint_name) for i in range(t_gpu_num)
        ]
        model_weights = [os.path.join(tmp_dir, f"mp_rank_{i:02d}", model_weights_ckpt) for i in range(t_gpu_num)]

    for s, t in zip(sources, model_weights):
        print("****** Start converting...", s, t)
        ckpt = torch.load(s, map_location="cpu")
        conf = ckpt["hyper_parameters"]
        ckpt = ckpt["state_dict"]
        os.makedirs(os.path.dirname(t), exist_ok=True)
        torch.save(ckpt, t)

    if args.vocab_file is not None:
        conf.tokenizer.vocab_file = args.vocab_file
        conf.vocab_file = args.vocab_file
    if args.merge_file is not None:
        conf.tokenizer.merge_file = args.merge_file
        conf.merges_file = args.merge_file

    print("****** Conf: ", conf)
    print("****** Checkpoints processing time: ", time.time() - start_time)

    tar_start_time = time.time()
    with open(config_yaml, 'w') as fout:
        OmegaConf.save(config=conf, f=fout, resolve=True)

    print("****** Calling tar...")
    subprocess.call(f'tar -czvf {args.nemo_file_path} ./*', cwd=tmp_dir, shell=True)

    shutil.rmtree(tmp_dir)
    print("****** Tar time: ", time.time() - tar_start_time)
    print("****** Total converting time: ", time.time() - start_time)
    print("****** Done.")


if __name__ == '__main__':
    main()
