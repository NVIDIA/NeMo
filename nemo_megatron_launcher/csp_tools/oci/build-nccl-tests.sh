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

HPCX_PATH="/opt/hpcx-v2.11-gcc-MLNX_OFED_LINUX-5-ubuntu20.04-cuda11-gdrcopy2-nccl2.11-x86_64"

export OMPI_MCA_pml=ucx
export OMPI_MCA_btl=^openib

srun --container-mounts="$PWD:/nccl,$HPCX_PATH:/opt/hpcx" \
     --container-image="nvcr.io/nvidia/pytorch:21.09-py3" \
     --container-name="nccl" \
     bash -c "
     cd /nccl &&
     git clone https://github.com/NVIDIA/nccl-tests.git &&
     source /opt/hpcx/hpcx-init.sh &&
     hpcx_load &&
     cd nccl-tests &&
     make MPI=1"
