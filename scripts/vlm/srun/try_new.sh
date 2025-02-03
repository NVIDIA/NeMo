#!/bin/bash
#SBATCH -A coreai_dlalgo_genai             # slurm account
#SBATCH -p batch                           # slurm partition
#SBATCH -J coreai_dlalgo_genai-clip_pretrain.launcher
#SBATCH -N 8                               # number of nodes
#SBATCH --ntasks-per-node=8                # number of tasks per node (one per GPU)
#SBATCH --time=04:00:00                    # job duration
#SBATCH --output=/lustre/fsw/coreai_dlalgo_genai/abhgarg/clip/slurm_outputs/slurm-%j.out # Redirect Slurm stdout
#SBATCH --error=/lustre/fsw/coreai_dlalgo_genai/abhgarg/clip/slurm_outputs/slurm-%j.err  # Redirect Slurm stderr

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
NAME="clip_pretrain_launcher_config_8_rebase_clip_task_encoder_mock_4"

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
python examples/multimodal/vision_language_foundation/clip/megatron_clip_pretrain.py"
# Mounts
MOUNTS="--container-mounts=${BASE_DIR}:${BASE_DIR},${DATA_DIR}:${DATA_DIR},${CODE_DIR}:/opt/NeMo,/lustre/fsw/coreai_dlalgo_genai/abhgarg/code/webdataset/webdataset:/usr/local/lib/python3.10/dist-packages/webdataset12"

# Output log files
SRUN_STDOUT="${SRUN_OUTPUT_DIR}/output-%j.out"
SRUN_STDERR="${SRUN_OUTPUT_DIR}/error-%j.err"

# Print the srun command
echo "Running the following srun command:"
#echo "srun --export=ALL --container-image=${CONTAINER_IMAGE} ${MOUNTS} -o ${SRUN_STDOUT} -e ${SRUN_STDERR} bash -c \"${TRAINING_CMD}\""

# Run the script with srun
srun --export=ALL --unbuffered --container-image=${CONTAINER_IMAGE} ${MOUNTS} -o ${SRUN_STDOUT} -e ${SRUN_STDERR} bash -c "${TRAINING_CMD}"
