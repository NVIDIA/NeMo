## Docker Container

`nvcr.io/nvidia/nemo:dev` or `nvcr.io/nvidia/nemo:24.07` 

I have an sqsh file already built for `nvcr.io/nvidia/nemo:dev` - much faster to start on eos/draco than giving the above docker containers. 
Sqsh path on EOS: `/lustre/fsw/llmservice_nemo_speechlm/users/pneekhara/launchscripts/nemodevNov24.sqsh`

Docker commands I run locally

```
docker run --runtime=nvidia -it --rm -v --shm-size=16g --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -v /home/pneekhara/2023:/home/pneekhara/2023 -v /datap/misc/:/datap/misc/ -v ~/.cache/torch:/root/.cache/torch -v ~/.netrc:/root/.netrc -v ~/.ssh:/root/.ssh --net=host nvcr.io/nvidia/nemo:dev
```

```
cd /home/pneekhara/2023/SimpleT5NeMo/NeMo; export PYTHONPATH="/home/pneekhara/2023/SimpleT5NeMo/NeMo.:${PYTHONPATH}" ; pip install ipdb ;
```

## Training Command

The manifest json files should contain the following keys: `audio_filepath, duration, text, speaker` . `speaker` is not currently being used so can be anything.
If we have already extracted the audio codes then they can also contain the key `target_audio_codes_path` pointing to the absolute path to the codes .pt file of shape (8, T).
Note: `target_audio_codes_path` should either be present in ALL training manifests or absent in ALL training manifest. Train set cannot be a mix of both. Same goes for val set.
If `target_audio_codes_path` is not present, codes are extracted on the fly (and training will be slower).


```
python examples/tts/t5tts.py \
max_epochs=1000 \
weighted_sampling_steps_per_epoch=1000 \
exp_manager.exp_dir="/datap/misc/Experiments/SimpleT5Explore/LocalTraining_LRH/" \
+train_ds_meta.hifittstrain.manifest_path="/home/pneekhara/2023/SimpleT5NeMo/manifests/hifitts__phoneme__nemo_audio_21fps_8codebooks_2kcodes_v2bWithWavLM_simplet5.json" \
+train_ds_meta.hifittstrain.audio_dir="/datap/misc/Datasets/hi_fi_tts_v0" \
+train_ds_meta.hifittstrain.feature_dir="/datap/misc/Datasets/hi_fi_tts_v0" \
+train_ds_meta.hifittstrain.sample_weight=1.0 \
+train_ds_meta.rivatrain.manifest_path="/home/pneekhara/2023/SimpleT5NeMo/manifests/rivaLindyRodney__phoneme__nemo_audio_21fps_8codebooks_2kcodes_v2bWithWavLM_simplet5.json" \
+train_ds_meta.rivatrain.audio_dir="/datap/misc/Datasets/riva" \
+train_ds_meta.rivatrain.feature_dir="/datap/misc/Datasets/riva" \
+train_ds_meta.rivatrain.sample_weight=1.0 \
+train_ds_meta.libri360.manifest_path="/home/pneekhara/2023/SimpleT5NeMo/manifests/libri360__phoneme__nemo_audio_21fps_8codebooks_2kcodes_v2bWithWavLM_simplet5.json" \
+train_ds_meta.libri360.audio_dir="/datap/misc/LibriTTSfromNemo/LibriTTS" \
+train_ds_meta.libri360.feature_dir="/datap/misc/LibriTTSfromNemo/LibriTTS" \
+train_ds_meta.libri360.sample_weight=1.0 \
+train_ds_meta.libri100.manifest_path="/home/pneekhara/2023/SimpleT5NeMo/manifests/libri100__phoneme__nemo_audio_21fps_8codebooks_2kcodes_v2bWithWavLM_simplet5.json" \
+train_ds_meta.libri100.audio_dir="/datap/misc/LibriTTSfromNemo/LibriTTS" \
+train_ds_meta.libri100.feature_dir="/datap/misc/LibriTTSfromNemo/LibriTTS" \
+train_ds_meta.libri100.sample_weight=1.0 \
+val_ds_meta.librival.manifest_path="/datap/misc/LibriTTSfromNemo/LibriTTS/devclean_small.json" \
+val_ds_meta.librival.audio_dir="/datap/misc/LibriTTSfromNemo/LibriTTS" \
+val_ds_meta.librival.feature_dir="/datap/misc/LibriTTSfromNemo/LibriTTS" \
model.codecmodel_path="/datap/misc/checkpoints/AudioCodec_21Hz_no_eliz.nemo" \
model.alignment_loss_scale=0.01 \
model.prior_scaling_factor=0.5 \
model.prior_scaledown_start_step=5000 \
model.prior_end_step=8000 \
model.t5_encoder.use_flash_self_attention=true \
model.t5_encoder.use_flash_x_attention=true \
model.t5_decoder.use_flash_self_attention=true \
model.t5_decoder.use_flash_x_attention=false \
model.train_ds.dataloader_params.num_workers=2 \
model.validation_ds.dataloader_params.num_workers=0 \
trainer.val_check_interval=500 \
trainer.devices=1 \
~model.optim.sched ;
```

If you change the codec model, make sure to adjust these model config params in `t5tts.yaml`:

```
model:
  num_audio_codebooks: 8
  num_audio_tokens_per_codebook: 2048 # Keep atleast 2 extra for eos/bos ids
  codec_model_downsample_factor: 1024
```

To train then model without CTC loss and prior, set the below params:

```
model.alignment_loss_scale=0.0 \
model.prior_scaling_factor=null \
```

### Training sub file for cluster.

```
#!/bin/bash
#SBATCH -t 4:00:00                 # wall time
#SBATCH --ntasks-per-node=8        # tasks per node
#SBATCH --exclusive                # exclusive node access
#SBATCH --mem=0                    # all mem avail
#SBATCH --mail-type=FAIL           # only send email on failure
#SBATCH --overcommit               # Needed for pytorch
#SBATCH -p batch
#SBATCH -N 2                   # number of nodes
#SBATCH -A 	llmservice_nemo_speechlm
#SBATCH -J 	llmservice_nemo_speechlm-sample:pneekhara_sample

source containers_2407.sh
export HYDRA_FULL_ERROR=1
export CUDA_LAUNCH_BLOCKING=1

set -x

EXP_DIR="/lustre/fsw/llmservice_nemo_speechlm/users/pneekhara/gitrepos/experiments/ExperimentalT5MultiGPUFixed/alldataWeighted_WithCTCPrior_Flash"
DOCKER_EXP_DIR="/gitrepos/experiments/ExperimentalT5MultiGPUFixed/alldataWeighted_WithCTCPrior_Flash"

mkdir -p $EXP_DIR

### export PYTORCH_NO_CUDA_MEMORY_CACHING=1
### export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:1000
read -r -d '' cmd <<EOF
echo "*******STARTING********" \
&& cd /gitrepos/SimpleNeMo/NeMo ; export PYTHONPATH="/gitrepos/SimpleNeMo/NeMo/.:${PYTHONPATH}" ; cd /gitrepos/SimpleNeMo/NeMo \
&& echo "Starting training" \
&& PYTHONFAULTHANDLER=1 python examples/tts/t5tts.py \
exp_manager.exp_dir="${DOCKER_EXP_DIR}" \
+exp_manager.version=0 \
max_epochs=500 \
weighted_sampling_steps_per_epoch=10000 \
batch_size=24 \
phoneme_dict_path="/gitrepos/SimpleNeMo/NeMo/scripts/tts_dataset_files/ipa_cmudict-0.7b_nv23.01.txt" \
heteronyms_path="/gitrepos/SimpleNeMo/NeMo/scripts/tts_dataset_files/heteronyms-052722" \
+train_ds_meta.hifittstrain.manifest_path="/data/TTS/manifests/hifitts__phoneme__nemo_audio_21fps_8codebooks_2kcodes_v2bWithWavLM_simplet5.json" \
+train_ds_meta.hifittstrain.audio_dir="/data/TTS/hi_fi_tts_v0" \
+train_ds_meta.hifittstrain.feature_dir="/data/TTS/hi_fi_tts_v0" \
+train_ds_meta.hifittstrain.sample_weight=1.0 \
+train_ds_meta.rivatrain.manifest_path="/data/TTS/manifests/rivaLindyRodney__phoneme__nemo_audio_21fps_8codebooks_2kcodes_v2bWithWavLM_simplet5.json" \
+train_ds_meta.rivatrain.audio_dir="/data/TTS/riva" \
+train_ds_meta.rivatrain.feature_dir="/data/TTS/riva" \
+train_ds_meta.rivatrain.sample_weight=1.0 \
+train_ds_meta.libri100train.manifest_path="/data/TTS/manifests/libri100__phoneme__nemo_audio_21fps_8codebooks_2kcodes_v2bWithWavLM_simplet5.json" \
+train_ds_meta.libri100train.audio_dir="/data/TTS/LibriTTS" \
+train_ds_meta.libri100train.feature_dir="/data/TTS/LibriTTS" \
+train_ds_meta.libri100train.sample_weight=1.0 \
+train_ds_meta.libri360train.manifest_path="/data/TTS/manifests/libri360__phoneme__nemo_audio_21fps_8codebooks_2kcodes_v2bWithWavLM_simplet5.json" \
+train_ds_meta.libri360train.audio_dir="/data/TTS/LibriTTS" \
+train_ds_meta.libri360train.feature_dir="/data/TTS/LibriTTS" \
+train_ds_meta.libri360train.sample_weight=1.0 \
+train_ds_meta.mlstrain.manifest_path="/data/TTS/manifests/mls17k__phoneme__nemo_audio_21fps_8codebooks_2kcodes_v2bWithWavLM_simplet5.json" \
+train_ds_meta.mlstrain.audio_dir="/MLS" \
+train_ds_meta.mlstrain.feature_dir="/MLS" \
+train_ds_meta.mlstrain.sample_weight=0.1 \
+val_ds_meta.libridev.manifest_path="/data/TTS/LibriTTS/dev_clean.json" \
+val_ds_meta.libridev.audio_dir="/data/TTS/LibriTTS" \
+val_ds_meta.libridev.feature_dir="/data/TTS/LibriTTS" \
model.codecmodel_path="/gitrepos/checkpoints/AudioCodec_21Hz_no_eliz.nemo" \
model.train_ds.dataset.min_duration=0.5 \
model.validation_ds.dataset.min_duration=0.5 \
model.alignment_loss_scale=0.01 \
model.prior_scaling_factor=0.5 \
model.prior_scaledown_start_step=8000 \
model.prior_end_step=12000 \
model.t5_encoder.use_flash_self_attention=true \
model.t5_encoder.use_flash_x_attention=true \
model.t5_decoder.use_flash_self_attention=true \
model.t5_decoder.use_flash_x_attention=false \
model.train_ds.dataloader_params.num_workers=4 \
model.validation_ds.dataloader_params.num_workers=0 \
trainer.devices=8 \
trainer.val_check_interval=1000 \
model.optim.lr=2e-4 \
trainer.num_nodes=${SLURM_JOB_NUM_NODES}
EOF

echo $cmd

srun -o ${EXP_DIR}/slurm-sft-%j-%n.out -e ${EXP_DIR}/slurm-sft-%j-%n.err --no-container-mount-home --container-image="$CONTAINER" $MOUNTS bash -c "${cmd}"
set +x
```

Here's my containers file:

```
#!/usr/bin/env sh

######################################################################
# @author      : adithyare (adithyare@selene-login-01)
# @file        : containers
# @created     : Saturday Aug 13, 2022 15:29:54 PDT
#
# @description : file with holds all the paths required for docker containers
######################################################################


CONTAINER="/lustre/fsw/llmservice_nemo_speechlm/users/pneekhara/launchscripts/nemodevNov24.sqsh"

CODE="/lustre/fsw/llmservice_nemo_speechlm/users/pneekhara/gitrepos:/gitrepos"
DATASETS="/lustre/fsw/llmservice_nemo_speechlm/data/speechlm_codecs_updated/:/datap/misc/speechllm_codecdatasets/,/lustre/fsw/llmservice_nemo_speechlm/data/MLS:/MLS,/lustre/fsw/llmservice_nemo_speechlm/data/speechllm_codecdatasets_new/:/datap/misc/speechllm_codecdatasets_new/,/lustre/fsw/llmservice_nemo_speechlm/data/TTS/:/data/TTS/"

MOUNTS="--container-mounts=$CODE,$DATASETS"
```



## Inference Command

For now, I am running inference as follows. This saves the generated audio somewhere in `exp_manager.exp_dir`

```
CUDA_VISIBLE_DEVICES=1 python examples/tts/t5tts.py \
--config-name=t5tts_inference \
+init_from_ptl_ckpt="/datap/misc/checkpoints/withprior10.ckpt" \
exp_manager.exp_dir="/datap/misc/Experiments/SimpleT5Explore/Inference/DracoNoPriorInterspeech" \
+test_ds_meta.riva_interspeech.manifest_path="/datap/misc/Datasets/riva/riva_interspeech.json" \
+test_ds_meta.riva_interspeech.audio_dir="/datap/misc/Datasets/riva" \
+test_ds_meta.riva_interspeech.feature_dir="/datap/misc/Datasets/riva" \
model.codecmodel_path="/datap/misc/checkpoints/AudioCodec_21Hz_no_eliz.nemo" \
model.t5_encoder.use_flash_self_attention=true \
model.t5_encoder.use_flash_x_attention=true \
model.t5_decoder.use_flash_self_attention=true \
model.t5_decoder.use_flash_x_attention=false \
model.prior_scaling_factor=null
```
