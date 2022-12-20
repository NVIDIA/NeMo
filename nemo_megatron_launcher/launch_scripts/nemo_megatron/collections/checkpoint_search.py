# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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

import glob
import os

import hydra


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
            filepath = f"{dirname}/mp_rank_{tensor_model_parallel_rank:02d}/{basename}"
        else:
            filepath = f"{dirname}/tp_rank_{tensor_model_parallel_rank:02d}_pp_rank_{pipeline_model_parallel_rank:03d}/{basename}"
        return filepath
    else:
        return filepath


@hydra.main(config_path="conf", config_name="checkpoint_search")
def checkpoint_search(cfg):
    """
    Search in the checkpoint folder for the latest checkpoint or a regex name.
    The checkpoint path are injected based on model parallelism.
    """

    # Checkpoint search
    checkpoint_folder = cfg.checkpoint_folder
    checkpoint_name = cfg.checkpoint_name
    tensor_model_parallel_size = cfg.tensor_model_parallel_size
    pipeline_model_parallel_size = cfg.pipeline_model_parallel_size

    if checkpoint_name == "latest":
        checkpoints = os.path.join(checkpoint_folder, "*.ckpt")
        checkpoints = _inject_model_parallel_rank(
            checkpoints, tensor_model_parallel_size, pipeline_model_parallel_size
        )
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
    print(checkpoint_name)


if __name__ == "__main__":
    checkpoint_search()
