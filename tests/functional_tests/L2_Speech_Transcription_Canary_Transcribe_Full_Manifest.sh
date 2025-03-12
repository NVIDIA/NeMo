python examples/asr/transcribe_speech.py \
dataset_manifest=/home/TestData/asr/canary/dev-other-wav-10-canary-fields.json \
output_filename=/tmp/preds.json \
batch_size=10 \
pretrained_name=nvidia/canary-1b \
num_workers=0 \
amp=false \
compute_dtype=bfloat16 \
matmul_precision=medium
