
HF_ROOT=/home/heh/github/transformers
SCRIPT=../tools/convert_llama2_to_hf.py

MODEL_SIZE="llama-2-7b-chat"
LLAMA_DIR=/media/data3/pretrained_models/llama2_raw
OUTPUT_DIR=/media/data3/pretrained_models/llama2_hf/$MODEL_SIZE


python $SCRIPT --input_dir $LLAMA_DIR --model_size $MODEL_SIZE --output_dir $OUTPUT_DIR

