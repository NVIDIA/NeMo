!/usr/bin/env bash

PATH_TO_SRC_FILE="/data/megatron_3b_1TP/megatron_t5_expanded_vocab_3.nemo"
PATH_TO_TGT_FILE="/data/megatron_3b_xTP/megatron_t5_expanded_vocab_3.nemo"

python examples/nlp/language_modeling/megatron_change_num_partitions.py \
    --model_file=${PATH_TO_SRC_FILE} \
    --target_file=${PATH_TO_TGT_FILE} \
    --model_class="nemo.collections.nlp.models.language_modeling.megatron_t5_model.MegatronT5Model" \
    --tensor_model_parallel_size=1 \
    --target_tensor_model_parallel_size=2 \
    --pipeline_model_parallel_size=1 \
    --target_pipeline_model_parallel_size=1 \
    --target_pipeline_model_parallel_split_rank=0 \
    --precision=bf16