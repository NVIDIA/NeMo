CUDA_VISIBLE_DEVICES=0 python examples/asr/speech_to_text_eval.py \
    pretrained_name=QuartzNet15x5Base-En  \
    dataset_manifest=/home/TestData/librispeech/librivox-dev-other.json \
    batch_size=64 \
    tolerance=0.1012
