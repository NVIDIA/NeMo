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

## Training

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

### Command Lhotse dataset
```
python examples/tts/t5tts.py \
    --config-name=t5tts_lhotse.yaml \
    batch_size=32 \
    micro_batch_size=32 \
    max_steps=1000000 \
    limit_val_batches=20 \
    trainer.max_steps=1000000 \
    trainer.val_check_interval=500 \
    exp_manager.exp_dir="/datap/misc/Experiments/SimpleT5Explore/LocalTraining_LRH/" \
    model.codecmodel_path="/home/ecasanova/Projects/Checkpoints/Audio_codec/21Hz-no-eliz/AudioCodec_21Hz_no_eliz.nemo" \
    model.alignment_loss_scale=0.01 \
    model.prior_scaling_factor=0.5 \
    model.prior_scaledown_start_step=5000 \
    model.prior_end_step=8000 \
    model.t5_encoder.use_flash_self_attention=true \
    model.t5_encoder.use_flash_x_attention=true \
    model.t5_decoder.use_flash_self_attention=true \
    model.t5_decoder.use_flash_x_attention=false \
    trainer.devices=1 \
    ++model.load_cached_codes_if_available=False \
    ++model.num_audio_codebooks=8 \
    ++model.num_audio_tokens_per_codebook=2048 \
    ++model.codec_model_downsample_factor=1024 \
    ~model.optim.sched ;

HYDRA_FULL_ERROR=1 PYTHONFAULTHANDLER=1 python examples/tts/t5tts.py \
    --config-name=t5tts_lhotse.yaml \
    exp_manager.exp_dir="/datap/misc/Experiments/SimpleT5Explore/LocalTraining_LRH/" \
    +exp_manager.version=0 \
    eval_batch_size=64 \
    batch_size=384 \
    micro_batch_size=24 \
    max_steps=5000000 \
    batch_duration=350 \
    limit_val_batches=25 \
    trainer.max_steps=5000000 \
    model.codecmodel_path="/lustre/fsw/llmservice_nemo_speechlm/users/pneekhara/gitrepos/checkpoints/AudioCodec_21Hz_no_eliz.nemo" \
    ++model.train_ds.dataset.input_cfg.0.type="lhotse_shar" \
    ++model.train_ds.dataset.input_cfg.0.shar_path="/lustre/fsw/llmservice_nemo_speechlm/data/TTS/tts_lhotse_datasets/hifitts_v0/" \
    ++model.train_ds.dataset.input_cfg.0.weight=1.0 \
    ++model.train_ds.dataset.input_cfg.0.tags.lang="en" \
    ++model.train_ds.dataset.input_cfg.0.tags.s2s=True \
    ++model.train_ds.dataset.input_cfg.0.tags.tokenizer_names=["english_phoneme"] \
    ++model.train_ds.dataset.input_cfg.1.type="lhotse_shar" \
    ++model.train_ds.dataset.input_cfg.1.shar_path="/lustre/fsw/llmservice_nemo_speechlm/data/TTS/tts_lhotse_datasets/libri100/" \
    ++model.train_ds.dataset.input_cfg.1.weight=1.0 \
    ++model.train_ds.dataset.input_cfg.1.tags.lang="en" \
    ++model.train_ds.dataset.input_cfg.1.tags.s2s=True \
    ++model.train_ds.dataset.input_cfg.1.tags.tokenizer_names=["english_phoneme"] \
    ++model.train_ds.dataset.input_cfg.2.type="lhotse_shar" \
    ++model.train_ds.dataset.input_cfg.2.shar_path="/lustre/fsw/llmservice_nemo_speechlm/data/TTS/tts_lhotse_datasets/rivaLindyRodney/" \
    ++model.train_ds.dataset.input_cfg.2.weight=1.0 \
    ++model.train_ds.dataset.input_cfg.2.tags.lang="en" \
    ++model.train_ds.dataset.input_cfg.2.tags.s2s=True \
    ++model.train_ds.dataset.input_cfg.2.tags.tokenizer_names=["english_phoneme"] \
    ++model.train_ds.dataset.input_cfg.3.type="lhotse_shar" \
    ++model.train_ds.dataset.input_cfg.3.shar_path="/lustre/fsw/llmservice_nemo_speechlm/data/TTS/tts_lhotse_datasets/libri360/" \
    ++model.train_ds.dataset.input_cfg.3.weight=1.0 \
    ++model.train_ds.dataset.input_cfg.3.tags.lang="en" \
    ++model.train_ds.dataset.input_cfg.3.tags.s2s=True \
    ++model.train_ds.dataset.input_cfg.3.tags.tokenizer_names=["english_phoneme"] \
    ++model.validation_ds.dataset.input_cfg.0.type="lhotse_shar" \
    ++model.validation_ds.dataset.input_cfg.0.shar_path="/lustre/fsw/llmservice_nemo_speechlm/data/TTS/tts_lhotse_datasets/LibriTTS_dev_clean/" \
    ++model.validation_ds.dataset.input_cfg.0.weight=1.0 \
    ++model.validation_ds.dataset.input_cfg.0.tags.lang="en" \
    ++model.validation_ds.dataset.input_cfg.0.tags.s2s=True \
    ++model.validation_ds.dataset.input_cfg.0.tags.tokenizer_names=["english_phoneme"] \
    model.alignment_loss_scale=0.01 \
    model.prior_scaling_factor=0.5 \
    model.prior_scaledown_start_step=5000 \
    model.prior_end_step=8000 \
    model.t5_encoder.use_flash_self_attention=true \
    model.t5_encoder.use_flash_x_attention=true \
    model.t5_decoder.use_flash_self_attention=true \
    model.t5_decoder.use_flash_x_attention=false \
    trainer.val_check_interval=50 \
    trainer.devices=8 \
    ++model.load_cached_codes_if_available=False \
    ++model.num_audio_codebooks=8 \
    ++model.num_audio_tokens_per_codebook=2048 \
    ++model.codec_model_downsample_factor=1024 \
    model.optim.lr=2e-4 \
    trainer.num_nodes=${SLURM_JOB_NUM_NODES}

```
Set `model.model_type=multi_encoder_context_tts` for Multi Encoder T5TTS and `model.use_text_conditioning_encoder=true` if you are doing text context training.

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

### Training sub files on cluster

| Model Type | Cluster | Training Sub File |
|------------|---------|--------|
| multi_encoder_context_tts | draco-oci-login-01.draco-oci-iad.nvidia.com |/lustre/fsw/portfolios/llmservice/users/pneekhara/launchscripts/unnormalized_me.sub |
| decoder_context_tts | draco-oci-login-01.draco-oci-iad.nvidia.com | /lustre/fsw/portfolios/llmservice/users/pneekhara/launchscripts/unnormalizedt5_decoder.sub |
| single_encoder_sv_tts | draco-oci-login-01.draco-oci-iad.nvidia.com | /lustre/fsw/portfolios/llmservice/users/pneekhara/launchscripts/unnormalizedt5_singleencoder.sub |
| decoder_pretrain_synthesizer | login-eos | /lustre/fsw/llmservice_nemo_speechlm/users/pneekhara/scriptsSimpleT5/newt5_pretrain.sub |

## Pretrained Models and Results

Paths to pretrained checkpoints and their evaluation results on some test sets can be found [here](https://docs.google.com/spreadsheets/d/16AkvAHZ-ytWYnzEx9wtOG7yLkuU2wfB8gGMiDa5sROg/edit?usp=sharing)

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

### DPO Preference Alignment

Preference Alignment (DPO) involves the following steps
1) Create a list of text-context pairs for which we will generate preference data.
2) For each text-context pair generate multiple audios from a base T5-TTS checkpoint and calculate metrics (CER/SSIM) for each generation.
3) Create chosen-rejected pairs from the generated audio.
4) Finetune the base T5-TTS checkpoint on the chosen-rejected pairs.

#### 1. Create text-context pairs
We pair a list of challenging texts with context audios from from Riva and LibriTTS dataset. We add a similar number of regular texts from LibriTTS and Riva (paired with random context audios). We also include examples with text contexts. There are other options for generating text-context pairs. 

```
python scripts/t5tts/dpo/create_text_contextpairs.py \
    --challenging_texts /Data/DPOPairsInputData/challenging_texts_nemollm.txt \
    --regular_texts_for_audiocontext /Data/DPOPairsInputData/regular_texts_for_audiocontext.txt \
    --regular_texts_for_textcontext /Data/DPOPairsInputData/regular_texts_for_textcontext.txt \
    --audio_contexts /Data/DPOPairsInputData/audio_context_list.json \
    --text_contexts /Data/DPOPairsInputData/text_context_list.txt \
    --output_manifest /Data/DPOPairsInputData/text_context_pairs_v2.json \
    --nsamples_perpair 6 ;
```
Each pair is repeated `nsamples_perpair` times which specifies how many samples we want to generate for each pair. The output manifest serves as the input for the next step.

We can also explore other options for these text-context pairs as well depending on the task. 

#### 2. Generate audios for each text-context pair

Next, we can generate audios from a base T5-TTS checkpoint using the following command. We pass the `audio_dir` as "/" since our text context pairs contains absolute paths. Model config arguments should be modified accordingly to match the base checkpoint architecture. We can run the below command on cluster to generate audios across multiple nodes. This command saves the generated audios along with the metrics for each generation in the `exp_dir`. Each generated audio file is accompanied with a `.json` file that has the CER/SSIM metrics. 

Sample sub file on EOS: `/lustre/fsw/llmservice_nemo_speechlm/users/shehzeenh/launchscripts/newdatagendpo_decoder.sub`

```
python examples/tts/t5tts.py \
--config-name=t5tts_inference \
batch_size=64 \
+init_from_ptl_ckpt="/mountdir/checkpoints/continuouscheckpoints_ks1_ks3/decodercontext_small_282.ckpt" \
exp_manager.exp_dir="/lustre/fsw/llmservice_nemo_speechlm/data/TTS/DPOData/Generations/decodercontext_small_282" \
+test_ds_meta.textcontextpairs.manifest_path="/lustre/fsw/llmservice_nemo_speechlm/data/TTS/DPOData/manifests/dpo_textcontext_pairs.json" \
+test_ds_meta.textcontextpairs.audio_dir="/" \
+test_ds_meta.textcontextpairs.feature_dir="/" \
model.model_type="decoder_context_tts" \
model.t5_encoder.kernel_size=3 \
model.t5_decoder.kernel_size=1 \
model.context_duration_min=5.0 \
model.context_duration_max=5.0 \
model.use_text_conditioning_encoder=true \
model.codecmodel_path="/mountdir/checkpoints/AudioCodec_21Hz_no_eliz.nemo" \
model.alignment_loss_scale=0.002 \
model.prior_scaling_factor=null \
model.load_cached_codes_if_available=false \
+model.use_kv_cache_for_inference=true \
trainer.num_nodes=${SLURM_JOB_NUM_NODES}
```
#### 3. Create chosen-rejected pairs from the generations

Next, we go through the generated audio directory and create chosen-rejected pairs. 

```
python scripts/t5tts/dpo/create_preference_pairs.py \
--input_manifest /lustre/fsw/llmservice_nemo_speechlm/data/TTS/DPOData/manifests/dpo_textcontext_pairs.json \
--generated_audio_dir /lustre/fsw/llmservice_nemo_speechlm/data/TTS/DPOData/Generations/decodercontext_small_282/T5TTS/version_0/audios \
--group_size 6 \
--cer_threshold 0.01 \
--val_size 256 ;
```

`cer_threshold=0.01` means that filter out pairs in which the chosen CER > 0.01.

This command should save train and val manifests for DPO finetuning in the base directory of the generated_audio_dir, that is, `/lustre/fsw/llmservice_nemo_speechlm/data/TTS/DPOData/Generations/decodercontext_small_282/T5TTS/version_0/manifests/` 

#### 4. DPO Finetuning Command

Finally, we perform DPO finetuning using the following command:

```
python examples/tts/t5tts.py \
batch_size=4 \
+init_from_ptl_ckpt="/mountdir/checkpoints/decoder_21_epoch_2.ckpt" \
+mode="dpo_train" \
max_epochs=10 \
exp_manager.exp_dir="/lustre/fsw/llmservice_nemo_speechlm/data/TTS/DPOData/TrainingsICML/decodercontext_small_282" \
exp_manager.checkpoint_callback_params.always_save_nemo=false \
model.train_ds.dataset._target_="nemo.collections.tts.data.text_to_speech_dataset.T5TTSDatasetDPO" \
model.validation_ds.dataset._target_="nemo.collections.tts.data.text_to_speech_dataset.T5TTSDatasetDPO" \
+train_ds_meta.dpopreftrain.manifest_path="/lustre/fsw/llmservice_nemo_speechlm/data/TTS/DPOData/Generations/decodercontext_small_282/T5TTS/version_0/manifests/dpo_train_manifest.json" \
+train_ds_meta.dpopreftrain.audio_dir="/" \
+train_ds_meta.dpopreftrain.feature_dir="/" \
+val_ds_meta.dpoprefval.manifest_path="/lustre/fsw/llmservice_nemo_speechlm/data/TTS/DPOData/Generations/decodercontext_small_282/T5TTS/version_0/manifests/dpo_val_manifest.json" \
+val_ds_meta.dpoprefval.audio_dir="/" \
+val_ds_meta.dpoprefval.feature_dir="/" \
+model.dpo_beta=0.01 \
+model.dpo_sft_loss_weight=0.0 \
model.model_type="decoder_context_tts" \
model.context_duration_min=5.0 \
model.context_duration_max=5.0 \
model.use_text_conditioning_encoder=true \
model.codecmodel_path="/mountdir/checkpoints/AudioCodec_21Hz_no_eliz.nemo" \
model.alignment_loss_scale=0.001 \
model.prior_scaling_factor=null \
trainer.val_check_interval=200 \
trainer.log_every_n_steps=10 \
model.optim.lr=2e-7 \
~model.optim.sched \
trainer.num_nodes=${SLURM_JOB_NUM_NODES}
```

Note the following overrides in the above command: 

```
+mode="dpo_train" \
model.train_ds.dataset._target_="nemo.collections.tts.data.text_to_speech_dataset.T5TTSDatasetDPO" \
model.validation_ds.dataset._target_="nemo.collections.tts.data.text_to_speech_dataset.T5TTSDatasetDPO" \
```

Again, our manifest contain absolute paths so we specify `audio_dir="/"` .