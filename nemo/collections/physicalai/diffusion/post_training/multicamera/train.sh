# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

#!/bin/bash
# example slurm script for training diffusion

#SBATCH -p your_partition -A your_account -t 24:00:00 --nodes=16 --exclusive --mem=0 --overcommit --gpus-per-node 8 --ntasks-per-node=8 --dependency=singleton

export HF_TOKEN=xxx
export WANDB_API_KEY=xxx
export WANDB_PROJECT=xxx
export WANDB_RESUME=allow
export NVTE_FUSED_ATTN=0 
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=xxx
export XDG_CACHE_HOME=xxx

DIR=`pwd`

srun -l --container-image nvcr.io#nvidia/nemo:cosmos.1.0.1 --container-mounts "/lustre:/lustre/,/home:/home" --no-container-mount-home --mpi=pmix bash -c '
cd $DIR;
PYTHONPATH=.:nemo/collections/physicalai/datasets/dataverse:$PYTHONPATH;
python -u nemo/collections/physicalai/diffusion/post_training/multicamera/multicamera.py --yes'