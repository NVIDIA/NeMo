#!/bin/bash
#SBATCH -A llmservice_nemo_speechlm  # convai_convaird_nemo-speech
#SBATCH -p polar
#SBATCH -N 1                   # number of nodes. !!!WARNING!!! - SET THIS
#SBATCH --gres=gpu:8            # number of GPUs per node
#SBATCH -t 04:00:00            # wall time
#SBATCH --time-min 04:00:00
#SBATCH --exclusive             # exclusive node access
#SBATCH --overcommit 
#SBATCH --mem=0                 # all mem avail
#SBATCH --mail-type=FAIL        # only send email on failure
#SBATCH --ntasks-per-node=8    # n tasks per machine (one task per GPU) !!!WARNING!!! - SET THIS TO NUMBER OF GPUs per Node
#SBATCH -J llmservice_nemo_speechlm-langspeech:ssl
#SBATCH --output=slurm_out/%x=%j --error=slurm_out/%x=%j


set -x
CLUSTER="cs"

if [ "$CLUSTER" = "sel" ]; then
    GPUS_PER_NODE=8
    SLURM_ACCOUNT=swdl/swdl-langspeech
    USERID=heh
    LUSTRE_ACCOUNT_PREFIX=/lustre/fsw/${SLURM_ACCOUNT}
elif [ "$CLUSTER" = "rno" ]; then
    GPUS_PER_NODE=16
    SLURM_ACCOUNT='ent_aiapps'
    USERID='users/heh'
    LUSTRE_ACCOUNT_PREFIX=/gpfs/fs1/projects/${SLURM_ACCOUNT}
elif [ "$CLUSTER" = "oci" ]; then
    GPUS_PER_NODE=8
    SLURM_ACCOUNT='llmservice'
    USERID='users/heh'
    LUSTRE_ACCOUNT_PREFIX=/lustre/fsw/portfolios/${SLURM_ACCOUNT}
elif [ "$CLUSTER" = "cs" ]; then
    GPUS_PER_NODE=8
    SLURM_ACCOUNT='llmservice'
    USERID='users/heh'
    LUSTRE_ACCOUNT_PREFIX=/lustre/fsw/portfolios/${SLURM_ACCOUNT}
fi


# << CHANGE THIS >>
CONTAINER="gitlab-master.nvidia.com/heh/nemo_containers:nemo-24.01"
# CONTAINER="nvcr.io/nvidia/nemo:24.01.speech"
# CONTAINER="nvcr.io/nvidian/nemo-nightly:latest-nightly-main"

DATASET='ml430k'
PROJECT_NAME=ssl_WavLM
VAL_CHECK_INTERVAL=2000
TRAIN_IS_TARRED=true

# Training parameters
MAX_EPOCHS=1000
WARMUP_STEPS=25000
TRAIN_BATCH_SIZE=8
EVAL_BATCH_SIZE=8

NUM_WORKERS=8
PIN_MEMORY=true
ACCUMULATE_GRAD_BATCHES=1
PRECISION=bf16
GRAD_CLIP_VAL=1.0
DROP_LAST=false
MODEL_SUFFIX=_n${SLURM_JOB_NUM_NODES}_r1

# SEG LENGTHS
TRAIN_MAX_DURATION=100.0
TRAIN_MIN_DURATION=0.1
EVAL_MIN_DURATION=0.1

#SSL params
MASK_POS="pre_conv"
MASK_PROB=0.01
BLOCK_SIZE=40
ALLOW_OVERLAP=true
FREEZE_MASK=true

WAVLM_AUG_PROB=0.2
NOISE_AUG_PROB=0.1

# Batch size
GLOBAL_BATCH_SIZE=`expr $SLURM_JOB_NUM_NODES \* $GPUS_PER_NODE \* $TRAIN_BATCH_SIZE \* $ACCUMULATE_GRAD_BATCHES`


#CONFORMER PARAMS
D_MODEL=1024

#LOSS

# OPTMIZER AND SCHEDULER
OPT=adamw
PEAK_LR=0.004
LR_SCHEDULE=NoamAnnealing
if [ "$LR_SCHEDULE" = "NoamAnnealing" ]; then
    LR=$(echo "scale=9; $PEAK_LR * sqrt($D_MODEL) * sqrt($WARMUP_STEPS)" | bc)
else
    LR=$PEAK_LR
fi
MIN_LR=1e-6
WD=1e-3

# EXP_NAME=${CLUSTER}_${DATASET}_bs${GLOBAL_BATCH_SIZE}_${OPT}lr${PEAK_LR}_wd${WD}_warmup${WARMUP_STEPS}_maxstep${MAX_STEPS}_mask${MASK_PROB}x${BLOCK_SIZE}${MODEL_SUFFIX}
EXP_NAME=${CLUSTER}_FC-XL_${DATASET}_bs${GLOBAL_BATCH_SIZE}_${OPT}lr${PEAK_LR}_wd${WD}_warmup${WARMUP_STEPS}_epoch${MAX_EPOCHS}_mask${MASK_PROB}x${BLOCK_SIZE}${MASK_POS}_wavLM${WAVLM_AUG_PROB}x${NOISE_AUG_PROB}${MODEL_SUFFIX}


NOISE_MANIFEST="[/data/noise_data/musan/musan_nonspeech_manifest.json,/data/noise_data/freesound/freesound_noise_manifest_filtered.json]"

CONCAT_SAMPLE_PROB=null
CONCAT_SAMPLE_TECH=round-robin
TRAIN_IS_CONCAT=False


if [ "$DATASET" = "debug" ]; then
    TRAIN_MANIFEST='/data/librispeech_sp_tarred/tarred_audio_manifest.json'
    TRAIN_FILEPATH="/data/librispeech_sp_tarred/audio__OP_0..511_CL_.tar"

    VAL_MANIFEST="[/data/ASR/librispeech_eval/librivox-dev-other.json,/data/ASR/librispeech_eval/librivox-dev-clean.json,/data/ASR/librispeech_eval/librivox-test-other.json,/data/ASR/librispeech_eval/librivox-test-clean.json]"
elif [ "$DATASET" = "ll_unlab-60k" ]; then
    TRAIN_MANIFEST='/data/ASR/unlab-60k_seg_tarred/tarred_audio_manifest.json'
    TRAIN_FILEPATH="/data/ASR/unlab-60k_seg_tarred/audio__OP_0..2047_CL_.tar"
    VAL_MANIFEST="[/data/ASR/librispeech_eval/librivox-dev-other.json,/data/ASR/librispeech_eval/librivox-dev-clean.json,/data/ASR/librispeech_eval/librivox-test-other.json,/data/ASR/librispeech_eval/librivox-test-clean.json]"
elif [ "$DATASET" = "ll_asrset" ]; then
    TRAIN_IS_CONCAT=True
    CONCAT_SAMPLE_PROB="[0.7,0.3]"
    CONCAT_SAMPLE_TECH=random

    LL_TRAIN_MANIFEST="/data/ASR/unlab-60k_seg_tarred/tarred_audio_manifest.json"
    LL_TRAIN_FILEPATH="/data/ASR/unlab-60k_seg_tarred/audio__OP_0..2047_CL_.tar"

    VOX_TRAIN_MANIFEST="/data/unsupervised/voxpopuli_en_16k_wav_30s_tarred/tarred_audio_manifest.json"
    VOX_TRAIN_FILEPATH="/data/unsupervised/voxpopuli_en_16k_wav_30s_tarred/audio__OP_0..2047_CL_.tar"

    DATA_DIR="/data/ASR/NeMo_ASR_SET/English/v3.0/train_bucketed"
    TRAIN_ASR_MANIFESTS="[[${DATA_DIR}/bucket1/tarred_audio_manifest.json],[${DATA_DIR}/bucket2/tarred_audio_manifest.json],[${DATA_DIR}/bucket3/tarred_audio_manifest.json],[${DATA_DIR}/bucket4/tarred_audio_manifest.json],[${DATA_DIR}/bucket5/tarred_audio_manifest.json],[${DATA_DIR}/bucket6/tarred_audio_manifest.json],[${DATA_DIR}/bucket7/tarred_audio_manifest.json],[${DATA_DIR}/bucket8/tarred_audio_manifest.json]]"
    TRAIN_ASR_FILEPATHS="[[${DATA_DIR}/bucket1/audio__OP_0..8191_CL_.tar],[${DATA_DIR}/bucket2/audio__OP_0..8191_CL_.tar],[${DATA_DIR}/bucket3/audio__OP_0..8191_CL_.tar],[${DATA_DIR}/bucket4/audio__OP_0..8191_CL_.tar],[${DATA_DIR}/bucket5/audio__OP_0..8191_CL_.tar],[${DATA_DIR}/bucket6/audio__OP_0..8191_CL_.tar],[${DATA_DIR}/bucket7/audio__OP_0..8191_CL_.tar],[${DATA_DIR}/bucket8/audio__OP_0..8191_CL_.tar]]"

    TRAIN_MANIFEST="[${LL_TRAIN_MANIFEST},${TRAIN_ASR_MANIFESTS}]"
    TRAIN_FILEPATH="[${LL_TRAIN_FILEPATH},${TRAIN_ASR_FILEPATHS}]"
    VAL_MANIFEST="[/data/ASR/librispeech_eval/librivox-dev-other.json,/data/ASR/librispeech_eval/librivox-dev-clean.json,/data/ASR/librispeech_eval/librivox-test-other.json,/data/ASR/librispeech_eval/librivox-test-clean.json]"
elif [ "$DATASET" = "ml430k" ]; then
    TRAIN_IS_CONCAT=True
    # ml_448k_scale0.5_data.sh
    . /lustre/fsw/portfolios/llmservice/users/heh/scripts/ssl/ml_430k_scale0.5_data.sh
    CONCAT_SAMPLE_PROB=$TRAIN_WEIGHT
    CONCAT_SAMPLE_TECH=random
else
    TRAIN_FIELPATHS="/data/ASR/VoxPopuli/Multilingual/en_de_es_fr/train/audio__OP_0..1023_CL_.tar"
    DEV_DIR=${LUSTRE_ACCOUNT_PREFIX}/erastorgueva/temp_data/dev/
fi



# Directories for manifests, data, etc.
RESULTS_DIR=${LUSTRE_ACCOUNT_PREFIX}/${USERID}/results/$PROJECT_NAME/$EXP_NAME
PRETRAINED_MODEL_DIR=${LUSTRE_ACCOUNT_PREFIX}/${USERID}/pretrained_models
QUESTIONS_DIR=${LUSTRE_ACCOUNT_PREFIX}/${USERID}/questions
HFCACHE=${LUSTRE_ACCOUNT_PREFIX}/${USERID}/hf_cache
CODE_DIR=${LUSTRE_ACCOUNT_PREFIX}/${USERID}/code/nemo-ssl
MANIFESTS_DIR=${LUSTRE_ACCOUNT_PREFIX}/${USERID}/manifests
DATA_DIR=/lustre/fsw/portfolios/llmservice/projects/llmservice_nemo_speechlm/data

# Make results dir
mkdir -p ${RESULTS_DIR}
OUTFILE=${RESULTS_DIR}/slurm-%j-%n.out
ERRFILE=${RESULTS_DIR}/error-%j-%n.out


# WandB info
# << CHANGE THIS >>
WANDB='f5029311df02a27459c2c99c5fbef08978dc709e'

# Config file
CONFIG_PATH=/code/workspace/configs/
CONFIG_NAME=fastconformer_xlarge_ssl_rq_dns

MOUNTS="--container-mounts=/lustre/fsw:/lustre/fsw,/lustre/fsw/portfolios/llmservice/projects/llmservice_nemo_speechlm:/lustre/fsw/portfolios/llmservice/projects/llmservice_nemo_speechlm,$CODE_DIR:/code,$RESULTS_DIR:/results,$DATA_DIR:/data,$PRETRAINED_MODEL_DIR:/pretrained,${QUESTIONS_DIR}:/questions/,${HFCACHE}:/hfcache/"


read -r -d '' cmd <<EOF
echo "*******STARTING********" \
&& echo "---------------" \
&& nvidia-smi \
&& export WANDB_API_KEY=${WANDB} \
&& cd /code \
&& git rev-parse HEAD \
&& cd /code \
&& pip show torch \
&& export HF_HOME="/hfcache/" \
&& export HYDRA_FULL_ERROR=1 \
&& export  PYTHONPATH="/code/.:${PYTHONPATH}" \
&& python -c 'import pytorch_lightning as ptl; print(ptl.__version__)' \
&& echo "Starting training" \
&& ls /lustre/fsw/portfolios/llmservice/projects/llmservice_nemo_speechlm/data \
&& python /code/workspace/speech_pretrain_denoise.py \
    --config-path=${CONFIG_PATH} \
    --config-name=${CONFIG_NAME} \
    model.train_ds.manifest_filepath=${TRAIN_MANIFEST} \
    ++model.train_ds.is_tarred=${TRAIN_IS_TARRED} \
    ++model.train_ds.is_concat=${TRAIN_IS_CONCAT} \
    ++model.train_ds.concat_sampling_technique=${CONCAT_SAMPLE_TECH} \
    ++model.train_ds.concat_sampling_probabilities=${CONCAT_SAMPLE_PROB} \
    ++model.train_ds.tarred_audio_filepaths=${TRAIN_FILEPATH} \
    model.train_ds.batch_size=$TRAIN_BATCH_SIZE \
    model.train_ds.num_workers=$NUM_WORKERS \
    model.train_ds.pin_memory=$PIN_MEMORY \
    model.train_ds.min_duration=$TRAIN_MIN_DURATION \
    model.train_ds.max_duration=$TRAIN_MAX_DURATION \
    model.train_ds.noise_manifest=$NOISE_MANIFEST \
    model.train_ds.batch_augmentor.prob=$WAVLM_AUG_PROB \
    model.train_ds.batch_augmentor.noise_ratio=$NOISE_AUG_PROB \
    ++model.train_ds.drop_last=${DROP_LAST} \
    model.validation_ds.manifest_filepath=$VAL_MANIFEST \
    model.validation_ds.batch_size=$EVAL_BATCH_SIZE \
    model.validation_ds.num_workers=$NUM_WORKERS \
    model.validation_ds.pin_memory=$PIN_MEMORY \
    model.validation_ds.min_duration=$EVAL_MIN_DURATION \
    model.validation_ds.noise_manifest=$NOISE_MANIFEST \
    model.validation_ds.batch_augmentor.prob=$WAVLM_AUG_PROB \
    model.validation_ds.batch_augmentor.noise_ratio=$NOISE_AUG_PROB \
    ++model.validation_ds.drop_last=${DROP_LAST} \
    trainer.num_nodes=$SLURM_JOB_NUM_NODES  \
    trainer.max_epochs=$MAX_EPOCHS \
    trainer.log_every_n_steps=10 \
    trainer.precision=$PRECISION \
    ++trainer.val_check_interval=$VAL_CHECK_INTERVAL \
    ++trainer.accumulate_grad_batches=$ACCUMULATE_GRAD_BATCHES \
    ++trainer.gradient_clip_val=$GRAD_CLIP_VAL \
    ++exp_manager.exp_dir=/results/ \
    ++exp_manager.create_wandb_logger=true \
    ++exp_manager.max_time_per_run=00:03:50:00 \
    ++exp_manager.name=$EXP_NAME \
    ++exp_manager.wandb_logger_kwargs.name=${EXP_NAME} \
    ++exp_manager.wandb_logger_kwargs.project=${PROJECT_NAME} \
    ++exp_manager.wandb_logger_kwargs.resume=true \
    ++exp_manager.resume_if_exists=true \
    ++exp_manager.resume_ignore_no_checkpoint=true \
    ++exp_manager.checkpoint_callback_params.save_top_k=1 \
    ++exp_manager.checkpoint_callback_params.always_save_nemo=false \
    ++exp_manager.checkpoint_callback_params.monitor="val_loss" \
    model.mask_position=$MASK_POS \
    model.masking.block_size=$BLOCK_SIZE \
    model.masking.mask_prob=$MASK_PROB \
    model.masking.allow_overlap=$ALLOW_OVERLAP \
    model.masking.freeze=$FREEZE_MASK \
    model.encoder.d_model=$D_MODEL \
    model.optim.name=$OPT \
    model.optim.lr=$LR \
    model.optim.betas=[0.9,0.98] \
    model.optim.weight_decay=$WD \
    model.optim.sched.warmup_steps=$WARMUP_STEPS \
    model.optim.sched.name=$LR_SCHEDULE \
    model.optim.sched.min_lr=$MIN_LR
EOF

# ++trainer.max_steps=$MAX_STEPS \
# ++model.optim.sched.max_steps=$MAX_STEPS

srun -o $OUTFILE -e $ERRFILE --container-image="$CONTAINER" $MOUNTS bash -c "${cmd}"
set +x

