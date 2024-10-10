# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
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

import argparse
import gc
import glob
import logging
import os
import shutil
import tarfile

import numpy as np
import tensorstore  # need to import it for bf16 support
import torch
import torch.distributed as dist
import zarr

try:
    from megatron.core.dist_checkpointing import load, save
    from megatron.core.dist_checkpointing.mapping import ShardedObject, ShardedTensor
    from megatron.core.dist_checkpointing.serialization import load_sharded_metadata
    from megatron.core.dist_checkpointing.strategies.torch import TorchDistSaveShardedStrategy
except ImportError:
    raise ImportError("Megatron is not installed. Please install Megatron-LM to use this script.")

from tqdm import tqdm


"""
Script to average distributed checkpoints and save new .nemo checkpoint.

Example: python scripts/checkpoint_averaging/torch_distributed_checkpoint_averaging.py \
                --name_prefix=<checkpoint name> \
                --checkpoint_dir=<folder with .discp files> \
                --untarred_nemo_folder=<path to the untarred nemo checkpoint to get config and tokenizers> \
                --steps <optinally a list of checkpoint steps to average, if not provided, it will average all the checkpoints>
"""


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def init_distributed():
    if not dist.is_available():
        raise ValueError("Distributed package is not available in your PyTorch build.")

    if not dist.is_initialized():
        logging.info(
            "Initializing distributed process group: rank %s, world_size %s",
            os.environ["LOCAL_RANK"],
            os.environ["WORLD_SIZE"],
        )
        dist.init_process_group(backend="gloo")
    else:
        logging.info("Distributed process group already initialized.")


def cleanup_distributed():
    """
    Cleans up the distributed process group.
    """
    dist.destroy_process_group()


def load_distributed_weights_dcp(ckpt_path):
    logging.info(f"Starting to load checkpoint from {ckpt_path}")
    metadata = load_sharded_metadata(ckpt_path)

    keys_to_delete = [k for k in metadata if k.startswith("optimizer.")]
    for k in keys_to_delete:
        del metadata[k]

    state_dict = load(
        sharded_state_dict=metadata,
        checkpoint_dir=ckpt_path,
        validate_access_integrity=True,
    )
    model_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("model."):
            model_state_dict[k] = v
    logging.info(f"Loaded checkpoint from {ckpt_path}")
    return model_state_dict, metadata


def save_distributed_weights_dcp(ckpt_path, averaged_state_dict):
    logging.info(f"Saving averaged checkpoint to {ckpt_path}")

    os.makedirs(ckpt_path, exist_ok=True)
    save(
        sharded_state_dict=averaged_state_dict,
        checkpoint_dir=ckpt_path,
        sharded_strategy=TorchDistSaveShardedStrategy(backend="torch_dist", version=1),
        validate_access_integrity=False,
    )
    logging.info(f"Saved averaged checkpoint to {ckpt_path}")


def main(args):
    init_distributed()

    checkpoint_paths = []
    for ckpt_dir in os.listdir(args.checkpoint_dir):
        logging.info("Processing %s", ckpt_dir)
        if not os.path.isdir(os.path.join(args.checkpoint_dir, ckpt_dir)) or ckpt_dir.endswith("0-last"):
            continue
        if args.steps is None:
            checkpoint_paths.append(ckpt_dir)
        else:
            for step in args.steps:
                key = f"-step={step}-"
                if key in ckpt_dir:
                    checkpoint_paths.append(ckpt_dir)

    checkpoint_paths = [p for p in checkpoint_paths if not p.endswith(".nemo")]
    n = len(checkpoint_paths)

    logging.info(f"Averaging {n} checkpoints ... {'at steps:' + str(args.steps) if args.steps is not None else ''}")

    avg_weights = {}
    with torch.no_grad():
        for ix, path in tqdm(enumerate(checkpoint_paths), total=len(checkpoint_paths)):
            full_path = os.path.join(args.checkpoint_dir, path)
            weights, metadata = load_distributed_weights_dcp(full_path)
            for k, v in weights.items():
                if "_extra_state" in k:  # Extra state is not averaged
                    if ix == 0:
                        avg_weights[k] = v
                    continue
                if k.startswith(
                    "optimizer."
                ):  # These should be filtered out during load but skipping them here just for safety
                    continue
                if k not in avg_weights:
                    logging.info(f'"Initialized average weights dict with: {k}"')
                    avg_weights[k] = v.data.clone().detach()
                else:
                    avg_weights[k] += v.data.clone().detach()
            del weights
            gc.collect()

        for k in avg_weights:
            if "_extra_state" in k:
                continue
            if str(avg_weights[k].dtype).startswith("int"):
                raise ValueError("Int type not supported")
            else:
                array = avg_weights[k] / n
                avg_weights[k] = array

        logging.info(f"Finished averaging {n} checkpoints")

    if args.steps is None:
        ckpt_name = os.path.join(args.checkpoint_dir, args.name_prefix + "-averaged", "model_weights")
    else:
        steps_combined = "_".join([str(x) for x in args.steps])
        ckpt_name = os.path.join(
            args.checkpoint_dir,
            args.name_prefix + "-" + steps_combined + "-averaged",
            "model_weights",
        )

    for k, v in metadata.items():
        if isinstance(v, ShardedTensor):
            averaged_weight = avg_weights[k]
            v.data = averaged_weight
            avg_weights[k] = v
        elif isinstance(v, ShardedObject):  # BytesIO, not averaged
            data = avg_weights[k]
            v.data = data
            avg_weights[k] = v

    save_distributed_weights_dcp(ckpt_name, avg_weights)
    if dist.get_rank() == 0:
        ckpt_name = os.path.dirname(ckpt_name)
        shutil.copy(
            os.path.join(args.untarred_nemo_folder, "model_config.yaml"),
            os.path.join(ckpt_name, "model_config.yaml"),
        )

        files = glob.glob(f"{args.untarred_nemo_folder}/*.model") + glob.glob(
            f"{args.untarred_nemo_folder}/*.vocab.json"
        )
        logging.info(f"Copying other files: {files}")
        for file in files:
            logging.info(f"Copying source: {file} destination: {os.path.join(ckpt_name, os.path.basename(file))}")
            shutil.copy(file, os.path.join(ckpt_name, os.path.basename(file)))

        with tarfile.open(ckpt_name + ".nemo", "w") as tar:
            tar.add(ckpt_name, arcname=os.path.sep)

        shutil.rmtree(ckpt_name)
        logging.info(f"Averaged distributed checkpoint saved as : {ckpt_name + '.nemo'}")

        cleanup_distributed()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--name_prefix",
        required=True,
        help="Name of the final checkpoint. Will append -averaged.nemo automatically.",
    )
    parser.add_argument(
        "--untarred_nemo_folder",
        required=True,
        help="Path to the untarred nemo checkpoint to get config and tokenizers",
    )
    parser.add_argument(
        "--checkpoint_dir",
        required=True,
        help="Folder containing all the distributed checkpoints.",
    )
    # list of checkpoint steps to average
    parser.add_argument(
        "--steps",
        nargs="+",
        type=int,
        help="List of checkpoint steps to average. If not specified, will average all.",
    )

    args = parser.parse_args()
    main(args)
