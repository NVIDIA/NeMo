#!/bin/bash
# Decompress file
zstd -d -v $SLURM_SUBMIT_DIR/the_pile/train/$(printf "%02d" $SLURM_ARRAY_TASK_ID).jsonl.zst
# Delete compressed file
rm $SLURM_SUBMIT_DIR/the_pile/train/$(printf "%02d" $SLURM_ARRAY_TASK_ID).jsonl.zst
