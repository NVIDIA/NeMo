set -x
# 1D buckets [SSL, CTC]
coverage run --branch -a --data-file=/workspace/.coverage --source=/workspace/nemo scripts/speech_recognition/estimate_duration_bins.py \
  /home/TestData/an4_dataset/an4_train.json \
  --buckets 5
# 2D buckets [CTC, RNNT, TDT] / with tokenizer
coverage run --branch -a --data-file=/workspace/.coverage --source=/workspace/nemo scripts/speech_recognition/estimate_duration_bins_2d.py \
  /home/TestData/an4_dataset/an4_train_lang.json \
  --tokenizer /home/TestData/asr_tokenizers/canary/en/tokenizer_spe_bpe_v1024_max_4/tokenizer.model \
  --buckets 5 \
  --sub-buckets 2
# TODO(pzelasko): Figure out how to quote the value in the test properly for CI to accept it...
# 2D buckets with prompt [AED/Canary, SpeechLM] / with aggregate tokenizer + prompt format
# coverage run --branch -a --data-file=/workspace/.coverage --source=/workspace/nemo scripts/speech_recognition/estimate_duration_bins_2d.py \
#   /home/TestData/an4_dataset/an4_train_lang.json \
#   --tokenizer /home/TestData/asr_tokenizers/canary/canary_spl_tokenizer_v32/tokenizer.model \
#      /home/TestData/asr_tokenizers/canary/en/tokenizer_spe_bpe_v1024_max_4/tokenizer.model \
#      /home/TestData/asr_tokenizers/canary/es/tokenizer_spe_bpe_v1024_max_4/tokenizer.model \
#   --langs spl_tokens en es \
#   --prompt-format canary \
#   --prompt '[{"role":"user","slots":{"source_lang":"en","target_lang":"en","task":"asr","pnc":"yes"}}]' \
#   --buckets 5 \
#   --sub-buckets 2
