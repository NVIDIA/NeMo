### Offline Preference Alignment (DPO/RPO)

Code: `nemo/collections/tts/models/magpietts_preference_optimization.py`

Preference Alignment (DPO/RPO) involves the following steps
1) Create a list of text-context pairs for which we will generate preference data.
2) For each text-context pair generate multiple audios from a base T5-TTS checkpoint and calculate metrics (CER/SSIM) for each generation.
3) Create chosen-rejected pairs from the generated audio.
4) Finetune the base T5-TTS checkpoint on the chosen-rejected pairs.

#### 1. Create text-context pairs
We pair a list of challenging texts with context audios from from Riva and LibriTTS dataset. We add a similar number of regular texts from LibriTTS and Riva (paired with random context audios). We also include examples with text contexts. There are other options for generating text-context pairs. 

```
python scripts/magpietts/dpo/create_text_contextpairs.py \
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


```
python examples/tts/magpietts.py \
--config-name=magpietts_inference_en \
mode=test \
batch_size=64 \
+init_from_ptl_ckpt="/mountdir/checkpoints/continuouscheckpoints_ks1_ks3/decodercontext_small_282.ckpt" \
exp_manager.exp_dir="/lustre/fsw/llmservice_nemo_speechlm/data/TTS/DPOData/Generations/decodercontext_small_282" \
+test_ds_meta.textcontextpairs.manifest_path="/lustre/fsw/llmservice_nemo_speechlm/data/TTS/DPOData/manifests/dpo_textcontext_pairs.json" \
+test_ds_meta.textcontextpairs.audio_dir="/" \
+test_ds_meta.textcontextpairs.feature_dir="/" \
model.model_type="decoder_context_tts" \
model.encoder.kernel_size=3 \
model.decoder.kernel_size=1 \
model.context_duration_min=5.0 \
model.context_duration_max=5.0 \
model.use_text_conditioning_encoder=true \
model.codecmodel_path="/mountdir/checkpoints/AudioCodec_21Hz_no_eliz.nemo" \
model.alignment_loss_scale=0.002 \
model.prior_scaling_factor=null \
model.load_cached_codes_if_available=false \
trainer.num_nodes=${SLURM_JOB_NUM_NODES}
```
#### 3. Create chosen-rejected pairs from the generations

Next, we go through the generated audio directory and create chosen-rejected pairs. 

```
python scripts/magpietts/dpo/create_preference_pairs.py \
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
python examples/tts/magpietts.py \
batch_size=4 \
+init_from_ptl_ckpt="/mountdir/checkpoints/decoder_21_epoch_2.ckpt" \
+mode="dpo_train" \
max_epochs=10 \
exp_manager.exp_dir="/lustre/fsw/llmservice_nemo_speechlm/data/TTS/DPOData/TrainingsICML/decodercontext_small_282" \
exp_manager.checkpoint_callback_params.always_save_nemo=false \
model.train_ds.dataset._target_="nemo.collections.tts.data.text_to_speech_dataset.MagpieTTSDatasetDPO" \
model.validation_ds.dataset._target_="nemo.collections.tts.data.text_to_speech_dataset.MagpieTTSDatasetDPO" \
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
model.train_ds.dataset._target_="nemo.collections.tts.data.text_to_speech_dataset.MagpieTTSDatasetDPO" \
model.validation_ds.dataset._target_="nemo.collections.tts.data.text_to_speech_dataset.MagpieTTSDatasetDPO" \
```

Again, our manifest contain absolute paths so we specify `audio_dir="/"` .

### Online Preference Optimization (GRPO)

For online preference optmization, process is much simpler.

1) Create a list of text-context pairs for which we will generate preference data (just one pair for a text-context not repeated).
We'll use the same process as above, just set `nsamples_perpair 1` in the command.
```
python scripts/magpietts/dpo/create_text_contextpairs.py \
    --challenging_texts /Data/DPOPairsInputData/challenging_texts_nemollm.txt \
    --regular_texts_for_audiocontext /Data/DPOPairsInputData/regular_texts_for_audiocontext.txt \
    --regular_texts_for_textcontext /Data/DPOPairsInputData/regular_texts_for_textcontext.txt \
    --audio_contexts /Data/DPOPairsInputData/audio_context_list.json \
    --text_contexts /Data/DPOPairsInputData/text_context_list.txt \
    --output_manifest /Data/DPOPairsInputData/text_context_pairs_v2.json \
    --nsamples_perpair 1 ;
```

2. Train using GRPO

```
python examples/tts/magpietts.py \
+mode="onlinepo_train" \
+init_from_ptl_ckpt="/Data/ICML2025_CKPTS/icml2025_base_checkpoints/decodercontext_small_sp_ks3CorrectWithPrior_onlyphoneme_epoch161.ckpt" \
max_epochs=1000 \
exp_manager.exp_dir="/Data/Experiments/NewT5TTSGRPO/Try3NoDropoutBeta0.01_CFG/" \
+train_ds_meta.grpotrainnomls.manifest_path="/Data/DPOPairsInputDatav2/text_context_pairs_grpo_train_nomls.json" \
+train_ds_meta.grpotrainnomls.audio_dir="/" \
+train_ds_meta.grpotrainnomls.feature_dir="/" \
+val_ds_meta.grpovalnomls.manifest_path="/Data/DPOPairsInputDatav2/text_context_pairs_grpo_val_unseenspeakers_tinysubset.json" \
+val_ds_meta.grpovalnomls.audio_dir="/" \
+val_ds_meta.grpovalnomls.feature_dir="/" \
+model.num_generations_per_item=6 \
+model.grpo_beta=0.01 \
+model.reference_free=true \
model.decoder.p_dropout=0.0 \
model.encoder.p_dropout=0.0 \
model.model_type="decoder_context_tts" \
model.use_text_conditioning_encoder=true \
model.context_duration_min=5.0 \
model.context_duration_max=5.0 \
model.codecmodel_path="/Data/Checkpoints/AudioCodec_21Hz_no_eliz.nemo" \
model.alignment_loss_scale=0.0 \
model.prior_scaling_factor=null \
model.train_ds.dataloader_params.num_workers=0 \
model.validation_ds.dataloader_params.num_workers=0 \
exp_manager.checkpoint_callback_params.monitor="val_mean_reward" \
exp_manager.checkpoint_callback_params.mode="max" \
+trainer.use_distributed_sampler=False \
+model.inference_cfg_prob=0.5 \
+model.inference_cfg_scale=2.5 \
batch_size=2 \
model.optim.lr=1e-6 \
trainer.devices=2 \
trainer.log_every_n_steps=1 \
trainer.val_check_interval=50 \
~model.optim.sched \
trainer.num_nodes=${SLURM_JOB_NUM_NODES} ;
```

Note that setting `+model.reference_free=true` makes the `grpo_beta` param effectively 0 since it does not use the KL regularization loss and saves memory. If using the `grpo_beta > 0` and `+model.reference_free=false`, make sure to set dropout params to 0, `model.decoder.p_dropout=0.0` and
`model.encoder.p_dropout=0.0` for training stabilization. Recommended learning rate is `model.optim.lr=1e-6` or lower. Setting `+model.inference_cfg_prob=0.5` means that for half of the generations will be generated using cfg, so that we optimize for our preferences in both cfg and non cfg inference modes. You may set `+model.inference_cfg_prob=0.0` if we only care about non-cfg inference.