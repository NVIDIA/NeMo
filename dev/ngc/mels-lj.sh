#!/usr/bin/env bash

if [[ -z $1 || "$#" -ne 1 ]]; then
  echo "Usage: ./$(basename "$0") [experiment name id]"
  exit 1
fi

if [ ! -f setup.py ]; then
  echo "Script should be ran from project root."
  exit 1
fi

if [[ -z $WANDB_TOKEN ]]; then
  echo "Please, provide WanDB token."
  exit 1
fi

# ------------------------------------------------------ CONSTS ------------------------------------------------------
# NGC
IMAGE="nvidian/pytorch:19.12-py3"
GPU_MEM=16     # Default is 32.
NUM_GPU=1      # Default is 8.
OPT=O2         # Default is O0.
WS=stan        # Workspace name
WORKSPACE=/ws  # Workspace mount point
RESULT=/result # Results dir
# Script
SCRIPT=examples/tts/fasterspeech_mels.py
CONFIG=examples/tts/configs/fasterspeech-mels-lj.yaml
# LJSpeech
DATASET_SIZE=12500 # Train
NUM_EPOCHS=100     # Total number of epochs
BATCH_SIZE=128     # [1GPU,16G]: 97 its/e, 194 warmup
#BATCH_SIZE=256     # [1GPU,32G]: 48 its/e, 96 warmup
#BATCH_SIZE=128     # [8GPU,32G]: 97 its/e, 194 warmup
# Megatron run
#read GPU_MEM NUM_GPU BATCH_SIZE NUM_EPOCHS <<< '32 8 128 1500'

# ---------------------------------------------------- SAVE STATE ----------------------------------------------------
echo "Updating source code..."
# Choose run id.
script_id=$(basename "$0")
script_id=${script_id%.*}
name_id=$1
num_id=$(head /dev/urandom | tr -dc A-Za-z0-9 | head -c 13)
id="${script_id}"_"${name_id}"_"${num_id}"
# Choose tmp dir to save current state of the project.
tmp_dir=/tmp/nemos/"${id}"
echo "Tmp dir: ${tmp_dir}"
# Save current state.
mkdir -p "${tmp_dir}"
rsync -r . "${tmp_dir}" --exclude .git --filter=":- .gitignore"
ngc workspace upload "${WS}" --source "${tmp_dir}" --destination nemos/"${id}"

# -------------------------------------------------- CHOOSE COMMAND --------------------------------------------------
total_steps=$(((DATASET_SIZE / (BATCH_SIZE * NUM_GPU)) * NUM_EPOCHS))
eval_freq=100 # Fixed. Audio sampling process happens only once.
read -r -d '' cmd <<EOF
nvidia-smi \
&& apt-get update && apt-get install -y libsndfile1 && pip install -U librosa \
&& cp -R ${WORKSPACE}/nemos/${id} /nemo && cd /nemo && pip install .[all] \
&& pip install -U wandb && wandb login ${WANDB_TOKEN} \
&& python -m torch.distributed.launch --nproc_per_node=${NUM_GPU} ${SCRIPT} \
--model_config=${CONFIG} \
--amp_opt_level=${OPT} \
--batch_size=${BATCH_SIZE} \
--eval_batch_size=${BATCH_SIZE} \
--lr=$((BATCH_SIZE / 64))e-3 \
--train_freq=10 \
--eval_freq=${eval_freq} \
--warmup=$((total_steps / 50)) \
--num_epochs=${NUM_EPOCHS} \
--sample_warmup=$((total_steps - 2 * eval_freq)) \
--sample_freq=$((2 * eval_freq)) \
--work_dir=${RESULT} \
--wdb_name=${name_id} \
--wdb_tags=mels,ljspeech,opt \
--train_dataset=/data/ljspeech/split3/train.json \
--train_durs=/data/librimeta/durs/ljspeech_original-qn-15x5_2x-less/train.npy \
--eval_names \
eval \
test \
--eval_datasets \
/data/ljspeech/split3/eval.json \
/data/ljspeech/split3/test.json \
--eval_durs \
/data/librimeta/durs/ljspeech_original-qn-15x5_2x-less/eval.npy \
/data/librimeta/durs/ljspeech_original-qn-15x5_2x-less/test.npy \
--waveglow_checkpoint=/data/checkpoints/waveglow.pth \
--doubling
EOF

# ------------------------------------------------------- FIRE -------------------------------------------------------
ngc batch run \
  --name "${name_id}" \
  --image "${IMAGE}" \
  --ace nv-us-west-2 \
  --instance dgx1v."${GPU_MEM}"g."${NUM_GPU}".norm \
  --result "${RESULT}" \
  --datasetid 59943:/data/ljspeech \
  --datasetid 59319:/data/librimeta \
  --datasetid 59662:/data/checkpoints \
  --workspace "${WS}":"${WORKSPACE}" \
  --commandline "${cmd}"
