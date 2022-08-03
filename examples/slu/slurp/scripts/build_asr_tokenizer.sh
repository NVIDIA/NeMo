DATA_ROOT="./slurp_data"
MODE="asr"
python ../../../scripts/tokenizers/process_asr_text_tokenizer.py \
  --manifest="${DATA_ROOT}/train_${MODE}.json,${DATA_ROOT}/train_synthetic_${MODE}.json" \
  --data_root="${DATA_ROOT}/tokenizers_${MODE}/" \
  --vocab_size=58 \
  --tokenizer="spe" \
  --spe_type="unigram" \
  --log \
  --spe_bos \
  --spe_eos \
  --spe_pad
