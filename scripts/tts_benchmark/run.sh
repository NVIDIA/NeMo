N_REPEATS=${N_REPEATS:-10}  # More is unnecessary.
BATCH_SIZE=${BATCH_SIZE:-32}
N_CHARS=${N_CHARS:-128}
N_SAMPLES=${N_SAMPLES:-1024}

# TODO: hardcode, change it
manifest_path=/home/otatanov/data/datasets/lj_speech/train_manifest.json
mixer_tts_1_path=/home/otatanov/data/exp_manager_folder/lj_new_mixer_tts_aGF7NdKLwFdsk/checkpoints/lj_new_mixer_tts_aGF7NdKLwFdsk--val_mel_loss=0.5616-epoch=999-last_ptl_fix.ckpt
mixer_tts_2_path=/home/otatanov/data/exp_manager_folder/lj_new_mixer_tts_ufcsJbLKhWPF8/checkpoints/lj_new_mixer_tts_ufcsJbLKhWPF8--val_mel_loss=0.5541-epoch=999-last_ptl_fix.ckpt
mixer_tts_3_path=/home/otatanov/data/exp_manager_folder/lj_new_mixer_tts_7MUAttWjw2zhr/checkpoints/lj_new_mixer_tts_7MUAttWjw2zhr--val_mel_loss=0.5424-epoch=999-last_ptl_fix.ckpt
fastpitch_path=/home/otatanov/data/exp_manager_folder/lj_new_mixer_tts_v7biccFf2Z5pR/checkpoints/lj_new_mixer_tts_v7biccFf2Z5pR--val_mel_loss=0.5986-epoch=999-last_ptl_fix.ckpt

model_name=$1
# shellcheck disable=SC2154
if [[ $model_name -eq "mixer-tts-1" ]]
then
  # mixer-tts, baseline, but nlp via nlp aligner, phonemes, RC 1.0
  model_ckpt_path=$mixer_tts_1_path
  model_args="--without-matching"
elif [[ $model_name -eq "mixer-tts-2" ]]
then
  # mixer-tts, baseline - nlp (phonemes), RC 1.0
  model_ckpt_path=$mixer_tts_2_path
  model_args="--without-matching"
elif [[ $model_name -eq "mixer-tts-3" ]]
then
  # mixer-tts, baseline (chars), RC 1.0
  model_ckpt_path=$mixer_tts_3_path
  model_args=" "
elif [[ $model_name -eq "fastpitch" ]]
then
  # mixer-tts, fastpitch + aligner, phonemes, RC 1.0
  model_ckpt_path=$fastpitch_path
  model_args="--without-matching"
else
  echo "Wrong model name."
  exit 1
fi

python scripts/tts_benchmark/inference.py \
  --model-ckpt-path=$model_ckpt_path \
  --manifest-path=$manifest_path \
  --cudnn-benchmark --amp-half \
  --n-repeats="$N_REPEATS" --batch-size="$BATCH_SIZE" --n-chars="$N_CHARS" --n-samples="$N_SAMPLES" \
  $model_args