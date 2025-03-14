coverage run --branch -a --data-file=/workspace/.coverage --source=/workspace/nemo examples/asr/transcribe_speech.py \
    audio_dir=/home/TestData/asr/canary/dev-other-wav \
    output_filename=preds.json \
    batch_size=10 \
    pretrained_name=nvidia/canary-1b \
    num_workers=0 \
    amp=false \
    compute_dtype=bfloat16 \
    matmul_precision=medium
