#!/bin/bash
# DOWNLOAD vocab and merges files from HuggingFace
mkdir bpe
wget https://huggingface.co/gpt2/resolve/main/vocab.json -P ./bpe/
wget https://huggingface.co/gpt2/resolve/main/merges.txt -P ./bpe/

# DOWNLOAD: first job, no dependencies
jobid1=$(sbatch --parsable download/download_all_pile_files.sh)
echo $jobid1

# EXTRACT: second job, after download
jobid2=$(sbatch --parsable --depend=aftercorr:$jobid1 extract/extract_all_pile_files.sh)
echo $jobid2

# PREPROCESS: third job, after extract
jobid3=$(sbatch --parsable --depend=aftercorr:$jobid2 preprocess/preprocess_all_pile_files.sh)

