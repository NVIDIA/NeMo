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
#SBATCH --job-name=dcgmi-diag
#SBATCH --time=1:00:00

# This is a Data Center GPU Manager container. This command will run GPU diagnostics.
# This script should not be called manually. It should only be called by cluster_validation.sh
srun --container-image=nvcr.io/nvidia/cloud-native/dcgm:2.3.5-1-ubi8 bash -c "dcgmi diag -r 3"
