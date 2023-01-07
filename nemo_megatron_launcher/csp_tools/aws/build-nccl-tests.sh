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

#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1

srun --container-mounts="$PWD:/nccl" \
     --container-image=../../nemo_megatron_training.sqsh \
     bash -c "
     cd /nccl &&
     curl -fSsL --proto '=https' https://github.com/NVIDIA/nccl-tests/tarball/master | tar xz &&
     mv NVIDIA-nccl-tests* nccl-tests &&
     cd nccl-tests &&
     make -j CUDA_HOME=/usr/local/cuda MPI=1 MPI_HOME=/opt/hpcx/ompi"
