python tools/preprocess_data.py \
       --input /home/hshin/data1/datasets/sec_qna_jsonls/sec_qna_2020-2022_num-200-shuffle_text-unfiltered-metaremoved_train-ln100.jsonl \
       --output-prefix /home/hshin/data1/datasets/sec_qna_jsonls/sec_qna_2020-2022_num-200-shuffle_text-unfiltered-metaremoved_train-ln100 \
       --tokenizer-model /home/hshin/data1/data/mt_nlg_plus_multilingual_ja_zh_the_stack_frac_015_256k.model \
       --tokenizer-type GPTSentencePieceTokenizer \
       --append-eod \
       --workers 8 \
&&\
python tools/preprocess_data.py \
       --input /home/hshin/data1/datasets/sec_qna_jsonls/sec_qna_2020-2022_num-200-shuffle_text-unfiltered-metaremoved_val-ln50.jsonl \
       --output-prefix /home/hshin/data1/datasets/sec_qna_jsonls/sec_qna_2020-2022_num-200-shuffle_text-unfiltered-metaremoved_val-ln50 \
       --tokenizer-model /home/hshin/data1/data/mt_nlg_plus_multilingual_ja_zh_the_stack_frac_015_256k.model \
       --tokenizer-type GPTSentencePieceTokenizer \
       --append-eod \
       --workers 8 \
&&\
python tools/preprocess_data.py \
       --input /home/hshin/data1/datasets/sec_qna_jsonls/sec_qna_2020-2022_num-200-shuffle_text-unfiltered-metaremoved_test-ln50.jsonl \
       --output-prefix /home/hshin/data1/datasets/sec_qna_jsonls/sec_qna_2020-2022_num-200-shuffle_text-unfiltered-metaremoved_test-ln50 \
       --tokenizer-model /home/hshin/data1/data/mt_nlg_plus_multilingual_ja_zh_the_stack_frac_015_256k.model \
       --tokenizer-type GPTSentencePieceTokenizer \
       --append-eobd \
       --workers 8 \
&&\

python tools/preprocess_data.py \
       --input /home/hshin/data1/datasets/sec_qna_jsonls/sec_qna_2023_Q1_num-100_text-unfiltered_txts-metaremoved_test-ln50.jsonl \
       --output-prefix /home/hshin/data1/datasets/sec_qna_jsonls/sec_qna_2023_Q1_num-100_text-unfiltered_txts-metaremoved_test-ln50 \
       --tokenizer-model /home/hshin/data1/data/mt_nlg_plus_multilingual_ja_zh_the_stack_frac_015_256k.model \
       --tokenizer-type GPTSentencePieceTokenizer \
       --append-eod \
       --workers 8 \
&&\
python tools/preprocess_data.py \
       --input /home/hshin/data1/datasets/sec_qna_jsonls/sec_qna_2023_Q1_num-100_text-unfiltered_txts-metaremoved_train-ln40.jsonl \
       --output-prefix /home/hshin/data1/datasets/sec_qna_jsonls/sec_qna_2023_Q1_num-100_text-unfiltered_txts-metaremoved_train-ln40 \
       --tokenizer-model /home/hshin/data1/data/mt_nlg_plus_multilingual_ja_zh_the_stack_frac_015_256k.model \
       --tokenizer-type GPTSentencePieceTokenizer \
       --append-eod \
       --workers 8 \
&&\
python tools/preprocess_data.py \
       --input /home/hshin/data1/datasets/sec_qna_jsonls/sec_qna_2023_Q1_num-100_text-unfiltered_txts-metaremoved_val-ln10.jsonl \
       --output-prefix /home/hshin/data1/datasets/sec_qna_jsonls/sec_qna_2023_Q1_num-100_text-unfiltered_txts-metaremoved_val-ln10 \
       --tokenizer-model /home/hshin/data1/data/mt_nlg_plus_multilingual_ja_zh_the_stack_frac_015_256k.model \
       --tokenizer-type GPTSentencePieceTokenizer \
       --append-eod \
       --workers 8 \
&&\

python tools/preprocess_data.py \
       --input /home/hshin/data1/datasets/sec_qna_jsonls/sec_qna_2020-2022_num-200-shuffle_text-unfiltered-metaremoved.jsonl \
       --output-prefix /home/hshin/data1/datasets/sec_qna_jsonls/sec_qna_2020-2022_num-200-shuffle_text-unfiltered-metaremoved \
       --tokenizer-model /home/hshin/data1/data/mt_nlg_plus_multilingual_ja_zh_the_stack_frac_015_256k.model \
       --tokenizer-type GPTSentencePieceTokenizer \
       --append-eod \
       --workers 8

python tools/preprocess_data.py \
       --input /home/hshin/data1/datasets/sec_qna_jsonls/sec_qna_2023_Q1_num-100_text-unfiltered_txts-metaremoved.jsonl \
       --output-prefix /home/hshin/data1/datasets/sec_qna_jsonls/sec_qna_2023_Q1_num-100_text-unfiltered_txts-metaremoved \
       --tokenizer-model /home/hshin/data1/data/mt_nlg_plus_multilingual_ja_zh_the_stack_frac_015_256k.model \
       --tokenizer-type GPTSentencePieceTokenizer \
       --append-eod \
       --workers 8
