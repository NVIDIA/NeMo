## Docker Container

`nvcr.io/nvidia/nemo:dev` or `nvcr.io/nvidia/nemo:24.07` 

I have an sqsh file already built for `nvcr.io/nvidia/nemo:dev` - much faster to start on eos/draco than giving the above docker containers. 
Sqsh path on EOS: `/lustre/fsw/llmservice_nemo_speechlm/users/pneekhara/launchscripts/nemodevNov24.sqsh`

Docker commands I run locally

```
docker run --runtime=nvidia -it --rm -v --shm-size=16g --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -v /home/pneekhara/2023:/home/pneekhara/2023 -v /datap/misc/:/datap/misc/ -v ~/.cache/torch:/root/.cache/torch -v ~/.netrc:/root/.netrc -v ~/.ssh:/root/.ssh --net=host nvcr.io/nvidia/nemo:dev
```

```
cd /home/pneekhara/2023/SimpleT5NeMo/NeMo; export PYTHONPATH="/home/pneekhara/2023/SimpleT5NeMo/NeMo.:${PYTHONPATH}" ;
```

## Code
* Model `nemo/collections/tts/models/t5tts.py`
* Dataset Class `T5TTSDataset` in `nemo/collections/tts/data/text_to_speech_dataset.py`
* Transformer Module `nemo/collections/tts/modules/t5tts_transformer.py`
* Config Yaml `examples/tts/conf/t5tts/t5tts.yaml`
* Training/Inference Script `examples/tts/t5tts.py`

## Model Types

Currently supports three model types `single_encoder_sv_tts` , `multi_encoder_context_tts` and `decoder_context_tts` (`cfg.model.model_type` in t5tts.yaml)

1. `single_encoder_sv_tts` is a simple T5 model: Text goes into the encoder and target audio goes to the decoder.
   Additionally, speaker_embedding of target audio (or context audio if provided) from TitaNet gets added to encoder output (all timesteps). Text context is not supported in this model.

2. `multi_encoder_context_tts` is a multi-encoder T5 model: Transcript and context audio go to different encoders.
   Transcript encoding feeds to layers given by `cfg.model.transcript_decoder_layers` and the context encoding feeds into the layers given by `context_decoder_layers` .
   Also supports text context which gets encoded by the same encoder as context audio. Only one of context audio or contex text is supported.

3. `decoder_context_tts` : Text goes into the encoder; context & target audio go to the decoder.
   Also supports text context. Currently, I have tested the model with using fixed sized context so I set `context_duration_min` and `context_duration_max` to the same value (5 seconds). Text context, which is usually shorter than number of codec frames of 5 second of audio, is padded to the max context duration in this model.

4. `decoder_pretrain_synthesizer` : This is the model type used for pretraining the decoder only on audio data using next frame prediction loss. 

## Training Command

### Manifest structure
For `single_encoder_sv_tts`, the manifest json files should contain the following keys: `audio_filepath, duration, text, speaker` . `speaker` is not currently being used so can be anything. Optionally, we can have a `context_audio_filepath` and `context_audio_duration` as well, if we want to use that for speaker embedding instead of the `audio_filepath`.
If we have already extracted the audio codes then they can also contain the key `target_audio_codes_path` pointing to the absolute path to the codes .pt file of shape (8, T).
Note: `target_audio_codes_path` should either be present in ALL training manifests or absent in ALL training manifest. Train set cannot be a mix of both. Same goes for val set.
If `target_audio_codes_path` is not present, codes are extracted on the fly (and training will be slower).

For `multi_encoder_context_tts`, `decoder_context_tts`, in addition to the above, the manifest should contain `context_audio_filepath` and `context_audio_duration`. If we have codes already extracted, we can have `context_audio_codes_path` (abosolute path) instead of `context_audio_filepath`. 

For text context training, we can have `context_text` key for text context and drop `context_audio_duration` and `context_audio_filepath` (or `context_audio_codes_path`).

If we have both `audio_filepath` and `target_audio_codes_path` in the manifest, the dataloader will load from `target_audio_codes_path`. To disable this and extract codes on the fly set the parameter `model.load_cached_codes_if_available=false` during training. Same goes for context audio.

### Manifests and Datasets

Manifests can be found in: `/lustre/fsw/portfolios/llmservice/users/pneekhara/gitrepos/TTS/manifests` on draco-oci (`draco-oci-dc-02.draco-oci-iad.nvidia.com`)
I use the following for training.

```
Train:
hifitts__phoneme__nemo_audio_21fps_8codebooks_2kcodes_v2bWithWavLM_simplet5_withContextAudioPaths.json
rivaLindyRodney__phoneme__nemo_audio_21fps_8codebooks_2kcodes_v2bWithWavLM_simplet5_withContextAudioPaths.json
rivaLindyRodney__phoneme__nemo_audio_21fps_8codebooks_2kcodes_v2bWithWavLM_textContextsimplet5_withContextAudioPaths.json
libri100__phoneme__nemo_audio_21fps_8codebooks_2kcodes_v2bWithWavLM_simplet5_withContextAudioPaths.json
libri360__phoneme__nemo_audio_21fps_8codebooks_2kcodes_v2bWithWavLM_simplet5_withContextAudioPaths.json
mls17k__phoneme__nemo_audio_21fps_8codebooks_2kcodes_v2bWithWavLM_verified_simplet5_withContextAudioPaths.json

Val:
dev_clean_withContextAudioPaths.json
```

Audio File Directories:
```
HifiTTS: /lustre/fsw/portfolios/llmservice/users/pneekhara/gitrepos/TTS/hi_fi_tts_v0
Libri100, Libri360 Libri dev: /lustre/fsw/portfolios/llmservice/users/pneekhara/gitrepos/TTS/LibriTTS
Lindy/Rodney: /lustre/fsw/portfolios/llmservice/users/pneekhara/gitrepos/TTS/riva
MLS Audio: /lustre/fsw/portfolios/edgeai/projects/edgeai_riva_rivamlops/data/tts/datasets/mls17k/filtered_24khz/audio_24khz
```

Pre-extracted Audio Codes (21 FPS with WavLM)
```
/lustre/fs11/portfolios/edgeai/projects/edgeai_riva_rivamlops/data/tts/datasets/codecs
```

### Command
```
python examples/tts/t5tts.py \
--config-name=t5tts \
max_epochs=1000 \
weighted_sampling_steps_per_epoch=1000 \
exp_manager.exp_dir="/datap/misc/Experiments/SimpleT5Explore/LocalTraining_LRH/" \
+train_ds_meta.rivatrain.manifest_path="/home/pneekhara/2023/SimpleT5NeMo/manifests/rivaLindyRodney__phoneme__nemo_audio_21fps_8codebooks_2kcodes_v2bWithWavLM_simplet5.json" \
+train_ds_meta.rivatrain.audio_dir="/datap/misc/Datasets/riva" \
+train_ds_meta.rivatrain.feature_dir="/datap/misc/Datasets/riva" \
+train_ds_meta.rivatrain.sample_weight=1.0 \
+train_ds_meta.libri360train.manifest_path="/home/pneekhara/2023/SimpleT5NeMo/manifests/libri360__phoneme__nemo_audio_21fps_8codebooks_2kcodes_v2bWithWavLM_simplet5.json" \
+train_ds_meta.libri360train.audio_dir="/datap/misc/LibriTTSfromNemo/LibriTTS" \
+train_ds_meta.libri360train.feature_dir="/datap/misc/LibriTTSfromNemo/LibriTTS" \
+train_ds_meta.libri360train.sample_weight=1.0 \
+train_ds_meta.libri100train.manifest_path="/home/pneekhara/2023/SimpleT5NeMo/manifests/libri100__phoneme__nemo_audio_21fps_8codebooks_2kcodes_v2bWithWavLM_simplet5.json" \
+train_ds_meta.libri100train.audio_dir="/datap/misc/LibriTTSfromNemo/LibriTTS" \
+train_ds_meta.libri100train.feature_dir="/datap/misc/LibriTTSfromNemo/LibriTTS" \
+train_ds_meta.libri100train.sample_weight=1.0 \
+val_ds_meta.librival.manifest_path="/home/pneekhara/2023/SimpleT5NeMo/manifests/dev_clean_withcontext.json" \
+val_ds_meta.librival.audio_dir="/datap/misc/LibriTTSfromNemo/LibriTTS" \
+val_ds_meta.librival.feature_dir="/datap/misc/LibriTTSfromNemo/LibriTTS" \
model.model_type="single_encoder_sv_tts" \
model.use_text_conditioning_encoder=true \
model.codecmodel_path="/datap/misc/checkpoints/AudioCodec_21Hz_no_eliz.nemo" \
model.alignment_loss_scale=0.005 \
model.prior_scaling_factor=0.5 \
model.prior_scaledown_start_step=5000 \
model.prior_end_step=8000 \
model.context_duration_min=3.0 \
model.context_duration_max=8.0 \
model.train_ds.dataloader_params.num_workers=2 \
model.validation_ds.dataloader_params.num_workers=2 \
trainer.val_check_interval=500 \
trainer.devices=-1 \
~model.optim.sched ;
```

Audio filepaths in the manifests should be relative to `audio_dir`. Codec paths are absolute.

Set `model.model_type=multi_encoder_context_tts` for Multi Encoder T5TTS or `decoder_context_tts` for decoder context and `model.use_text_conditioning_encoder=true` if you want both audio/text contexts.

If you change the codec model, make sure to adjust these model config params in `t5tts.yaml`:

```
model:
  num_audio_codebooks: 8
  num_audio_tokens_per_codebook: 2048 # Keep atleast 4 extra for eos/bos ids
  codec_model_downsample_factor: 1024
```

To train then model without CTC loss and prior, set the below params:

```
model.alignment_loss_scale=0.0 \
model.prior_scaling_factor=null \
```


### Training sub file for cluster (Draco-OCI)

```
#!/bin/bash
#SBATCH -t 4:00:00                 # wall time
#SBATCH --ntasks-per-node=8        # tasks per node
#SBATCH --exclusive                # exclusive node access
#SBATCH --mem=0                    # all mem avail
#SBATCH --mail-type=FAIL           # only send email on failure
#SBATCH --overcommit               # Needed for pytorch
#SBATCH -p batch_block1,batch_block3,batch_block4
#SBATCH --gres=gpu:8
#SBATCH -N 2                   # number of nodes
#SBATCH -A 	convai_convaird_nemo-speech
#SBATCH -J 	convai_convaird_nemo-speech-sample:samplepneekhara

source containers_nemodev.sh
export HYDRA_FULL_ERROR=1
export CUDA_LAUNCH_BLOCKING=1

set -x

EXP_DIR="/lustre/fsw/portfolios/llmservice/users/pneekhara/gitrepos/experiments/NormalizedNewT5Experiments/unnormalizedLalign005_multiEncoder_textcontext_kernel3"
DOCKER_EXP_DIR="/gitrepos/experiments/NormalizedNewT5Experiments/unnormalizedLalign005_multiEncoder_textcontext_kernel3"

mkdir -p $EXP_DIR

### export PYTORCH_NO_CUDA_MEMORY_CACHING=1
### export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:1000
read -r -d '' cmd <<EOF
echo "*******STARTING********" \
&& cd /gitrepos/SimpleNeMo/NeMo ; export PYTHONPATH="/gitrepos/SimpleNeMo/NeMo/.:${PYTHONPATH}" ; cd /gitrepos/SimpleNeMo/NeMo \
&& echo "Starting training" \
&& python examples/tts/t5tts.py \
exp_manager.exp_dir="${DOCKER_EXP_DIR}" \
+exp_manager.version=0 \
weighted_sampling_steps_per_epoch=10000 \
max_epochs=500 \
batch_size=16 \
phoneme_dict_path="/gitrepos/SimpleNeMo/NeMo/scripts/tts_dataset_files/ipa_cmudict-0.7b_nv23.01.txt" \
heteronyms_path="/gitrepos/SimpleNeMo/NeMo/scripts/tts_dataset_files/heteronyms-052722" \
+train_ds_meta.hifittstrain.manifest_path="/data/TTS/manifests/hifitts__phoneme__nemo_audio_21fps_8codebooks_2kcodes_v2bWithWavLM_simplet5_withContextAudioPaths.json" \
+train_ds_meta.hifittstrain.audio_dir="/data/TTS/hi_fi_tts_v0" \
+train_ds_meta.hifittstrain.feature_dir="/data/TTS/hi_fi_tts_v0" \
+train_ds_meta.hifittstrain.sample_weight=1.0 \
+train_ds_meta.rivatrain.manifest_path="/data/TTS/manifests/rivaLindyRodney__phoneme__nemo_audio_21fps_8codebooks_2kcodes_v2bWithWavLM_simplet5_withContextAudioPaths.json" \
+train_ds_meta.rivatrain.audio_dir="/data/TTS/riva" \
+train_ds_meta.rivatrain.feature_dir="/data/TTS/riva" \
+train_ds_meta.rivatrain.sample_weight=1.0 \
+train_ds_meta.rivatraintextcontext.manifest_path="/data/TTS/manifests/rivaLindyRodney__phoneme__nemo_audio_21fps_8codebooks_2kcodes_v2bWithWavLM_textContextsimplet5_withContextAudioPaths.json" \
+train_ds_meta.rivatraintextcontext.audio_dir="/data/TTS/riva" \
+train_ds_meta.rivatraintextcontext.feature_dir="/data/TTS/riva" \
+train_ds_meta.rivatraintextcontext.sample_weight=1.0 \
+train_ds_meta.libri100train.manifest_path="/data/TTS/manifests/libri100__phoneme__nemo_audio_21fps_8codebooks_2kcodes_v2bWithWavLM_simplet5_withContextAudioPaths.json" \
+train_ds_meta.libri100train.audio_dir="/data/TTS/LibriTTS" \
+train_ds_meta.libri100train.feature_dir="/data/TTS/LibriTTS" \
+train_ds_meta.libri100train.sample_weight=1.0 \
+train_ds_meta.libri360train.manifest_path="/data/TTS/manifests/libri360__phoneme__nemo_audio_21fps_8codebooks_2kcodes_v2bWithWavLM_simplet5_withContextAudioPaths.json" \
+train_ds_meta.libri360train.audio_dir="/data/TTS/LibriTTS" \
+train_ds_meta.libri360train.feature_dir="/data/TTS/LibriTTS" \
+train_ds_meta.libri360train.sample_weight=1.0 \
+train_ds_meta.mlstrain.manifest_path="/data/TTS/manifests/mls17k__phoneme__nemo_audio_21fps_8codebooks_2kcodes_v2bWithWavLM_verified_simplet5_withContextAudioPaths.json" \
+train_ds_meta.mlstrain.audio_dir="/MLS" \
+train_ds_meta.mlstrain.feature_dir="/MLS" \
+train_ds_meta.mlstrain.sample_weight=0.1 \
+val_ds_meta.libridev.manifest_path="/data/TTS/manifests/dev_clean_withContextAudioPaths.json" \
+val_ds_meta.libridev.audio_dir="/data/TTS/LibriTTS" \
+val_ds_meta.libridev.feature_dir="/data/TTS/LibriTTS" \
model.train_ds.dataset.min_duration=0.5 \
model.validation_ds.dataset.min_duration=0.5 \
model.context_duration_min=3.0 \
model.context_duration_max=8.0 \
model.codecmodel_path="/gitrepos/checkpoints/AudioCodec_21Hz_no_eliz.nemo" \
model.model_type="multi_encoder_context_tts" \
model.transcript_decoder_layers='[3,4,5,6,7]' \
model.context_decoder_layers='[8,9]' \
model.use_text_conditioning_encoder=true \
model.use_perceiver=false \
model.alignment_loss_scale=0.005 \
model.prior_scaling_factor=0.5 \
model.prior_scaledown_start_step=8000 \
model.prior_end_step=12000 \
model.t5_encoder.pos_emb.name="learnable" \
model.t5_decoder.pos_emb.name="learnable" \
model.context_encoder.pos_emb.name="learnable" \
model.t5_encoder.kernel_size=3 \
model.t5_decoder.kernel_size=3 \
model.context_encoder.kernel_size=3 \
model.t5_encoder.n_layers=6 \
model.context_encoder.n_layers=6 \
model.train_ds.dataloader_params.num_workers=2 \
model.validation_ds.dataloader_params.num_workers=0 \
trainer.devices=8 \
trainer.val_check_interval=1000 \
model.optim.lr=1e-4 \
trainer.num_nodes=${SLURM_JOB_NUM_NODES}
EOF

echo $cmd

srun -o ${EXP_DIR}/slurm-sft-%j-%n.out -e ${EXP_DIR}/slurm-sft-%j-%n.err --no-container-mount-home --container-image="$CONTAINER" $MOUNTS bash -c "${cmd}"
set +x

```

Here's my containers file for relevant file mounts, docker container.

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

## Inference and Eval

To infer and evaluate from a given checkpoint and hparams.yaml file I use `scripts/t5tts/infer_and_evaluate.py`. To evaluate on a given manifest (same structure as discussed above), edit the `dataset_meta_info` in `scripts/t5tts/infer_and_evaluate.py` to point to the paths on your machine or add any other datasets if missing.

```
dataset_meta_info = {
    'vctk': {
        'manifest_path' : '/home/pneekhara/2023/SimpleT5NeMo/manifests/smallvctk__phoneme__nemo_audio_21fps_8codebooks_2kcodes_v2bWithWavLM_simplet5_withcontextaudiopaths.json',
        'audio_dir' : '/datap/misc/Datasets/VCTK-Corpus',
        'feature_dir' : '/datap/misc/Datasets/VCTK-Corpus',
    },
    'riva_challenging': {
        'manifest_path' : '/home/pneekhara/2023/SimpleT5NeMo/manifests/challengingLindyRodney__phoneme__nemo_audio_21fps_8codebooks_2kcodes_v2bWithWavLM_simplet5_withContextAudioPaths.json',
        'audio_dir' : '/datap/misc/Datasets/riva',
        'feature_dir' : '/datap/misc/Datasets/riva',
    },
    'libri_val': {
        'manifest_path' : '/home/pneekhara/2023/SimpleT5NeMo/manifests/libri360_val.json',
        'audio_dir' : '/datap/misc/LibriTTSfromNemo/LibriTTS',
        'feature_dir' : '/datap/misc/LibriTTSfromNemo/LibriTTS',
    }
}
```

Then run

```
python scripts/t5tts/infer_and_evaluate.py \
--hparams_file <Path to hparams.yaml which is usualy found in the training log dir> \
--checkpoint_file <Path to T5TTS checkpoint > \
--codecmodel_path /datap/misc/checkpoints/AudioCodec_21Hz_no_eliz.nemo \
--datasets "vctk,libri_val" \
--out_dir /datap/misc/Evals \
--temperature 0.6 \
--topk 80
```

Ignore the other params in the file, I also use this for evaluating ongoing experiments on the cluster by copying over the checkpoints and hparams..

### Inference Notebook

Inference Notebook: `t5tts_inference.ipynb` For quickly trying custom texts/contexts.