#!/bin/bash
# Preprocess the data using the megatron code
python3 /workspace/bignlp_scripts/megatron-lm/tools/preprocess_data.py \
	--input /workspace/bignlp_scripts/prepare_dataset/the_pile/train/$(printf "%02d" $SLURM_ARRAY_TASK_ID).jsonl \
	--output-prefix /workspace/bignlp_scripts/prepare_dataset/the_pile/train/my-gpt3_$(printf "%02d" $SLURM_ARRAY_TASK_ID) \
	--vocab /workspace/bignlp_scripts/prepare_dataset/bpe/vocab.json \
	--dataset-impl mmap \
	--tokenizer-type GPT2BPETokenizer \
	--merge-file /workspace/bignlp_scripts/prepare_dataset/bpe/merges.txt \
	--workers $SLURM_CPUS_ON_NODE \
	--append-eod
# Remove the unprocessed data
rm /workspace/bignlp_scripts/prepare_dataset/the_pile/train/$(printf "%02d" $SLURM_ARRAY_TASK_ID).jsonl
