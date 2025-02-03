#!/bin/bash
#SBATCH -A coreai_dlalgo_genai             # slurm account
#SBATCH -p batch                           # slurm partition
#SBATCH -J coreai_dlalgo_genai-clip_pretrain.cc3m.launcher-workers_${NUM_WORKERS} # job name with num_workers
#SBATCH -N 8                               # number of nodes
#SBATCH --ntasks-per-node=8                # number of tasks per node (one per GPU)
#SBATCH --time=04:00:00                    # job duration
#SBATCH --output=/lustre/fsw/coreai_dlalgo_genai/abhgarg/clip/slurm_outputs/slurm-%j.out # Redirect Slurm stdout
#SBATCH --error=/lustre/fsw/coreai_dlalgo_genai/abhgarg/clip/slurm_outputs/slurm-%j.err  # Redirect Slurm stderr


# Parse the num_workers argument with a default value of 8
NUM_WORKERS=${1:-16}
# Container Image
CONTAINER_IMAGE="/lustre/fsw/coreai_dlalgo_genai/abhgarg/containers/custom-nemo.sqsh"

# Paths
CODE_DIR="/lustre/fsw/coreai_dlalgo_genai/abhgarg/code/NeMo/"
DATA_DIR="/lustre/fsw/coreai_dlalgo_genai/datasets"
BASE_DIR="/lustre/fsw/coreai_dlalgo_genai/abhgarg"
TRAIN_DATA="${DATA_DIR}/coyo-700m"
IMAGENET_VAL="${DATA_DIR}/imagenet_1k/val"
LOG_DIR="/lustre/fsw/coreai_dlalgo_genai/abhgarg/clip"
SRUN_OUTPUT_DIR="${LOG_DIR}/srun_outputs"
NAME="clip_pretrain_launcher_config_8_8_no_val_new_ddp_${NUM_WORKERS}"

# Ensure the output directory exists
mkdir -p ${SRUN_OUTPUT_DIR}
rm -rf ${LOG_DIR}/${NAME}/*unfinished*

# Set environment variables
export WANDB_API_KEY="876c14aec169a9ff64911dac73014aa6de13ec56"
export PYTHONUNBUFFERED=1  # Flush Python output immediately

# Dynamically set Slurm output and error paths
SBATCH_OUTPUT="${SRUN_OUTPUT_DIR}/slurm-%j.out"
SBATCH_ERROR="${SRUN_OUTPUT_DIR}/slurm-%j.err"

# Training script command
TRAINING_CMD="set -eux; \
cd ${CODE_DIR}; \
NEMO_ENV_VARNAME_TESTING=1 python ${CODE_DIR}/scripts/vlm/clip_finetune.py \
--data_path ${TRAIN_DATA} \
--log_dir ${LOG_DIR} \
--devices 8 \
--num_nodes 8 \
--wandb_project clip_nemo2 \
--name ${NAME} \
--gbs \$((64*500)) \
--mbs 500 \
--num_workers ${NUM_WORKERS} \
--imagenet_val ${IMAGENET_VAL}"

# Mounts
MOUNTS="--container-mounts=${BASE_DIR}:${BASE_DIR},${DATA_DIR}:${DATA_DIR},${CODE_DIR}:/opt/NeMo,/lustre/fsw/coreai_dlalgo_genai/abhgarg/code/webdataset/webdataset:/usr/local/lib/python3.10/dist-packages/webdataset"

# Output log files
SRUN_STDOUT="${SRUN_OUTPUT_DIR}/output-%j.out"
SRUN_STDERR="${SRUN_OUTPUT_DIR}/error-%j.err"

# Print the srun command
echo "Running the following srun command:"

# Run the script with srun
srun --export=ALL --unbuffered --container-image=${CONTAINER_IMAGE} ${MOUNTS} -o ${SRUN_STDOUT} -e ${SRUN_STDERR} bash -c "${TRAINING_CMD}"
