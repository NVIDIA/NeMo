python ../scripts/dataset_processing/add_noise.py \
    --input_manifest=/NeMo/project/synth_audio_val/synth_manifest.json \
    --noise_manifest=/NeMo/project/synth_audio_val/synth_manifest.json \
    --snrs 0 5 10 15 \
    --out_dir=/NeMo/project/synth_audio_val_noisy