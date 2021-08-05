#!/bin/bash
wget "https://the-eye.eu/public/AI/pile/train/$(printf "%02d" $SLURM_ARRAY_TASK_ID).jsonl.zst" -P $SLURM_SUBMIT_DIR/the_pile/train
