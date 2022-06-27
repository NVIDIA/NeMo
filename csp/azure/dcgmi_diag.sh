#!/bin/bash
#SBATCH --job-name=bignlp-dcgmi-diag
#SBATCH --time=1:00:00

# This is a Data Center GPU Manager container. This command will run GPU diagnostics.
# This script should not be called manually. It should only be called by cluster_validation.sh
srun --container-image=nvcr.io/nvidia/cloud-native/dcgm:2.3.5-1-ubi8 bash -c "dcgmi diag -r 3"
