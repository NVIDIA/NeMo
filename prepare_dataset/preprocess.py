import os


def main():
    data_save_dir = "/workspace/bignlp-scripts/prepare_dataset/the_pile"
    file_number = int(os.environ.get("SLURM_ARRAY_TASK_ID"))

    extracted_path = os.path.join(data_save_dir, f"{file_number:02d}.jsonl")
    code_path = "/workspace/bignlp-scripts/NeMo/examples/nlp/language_modeling/preprocess_data_for_megatron.py"

    flags = (
        f"--input /workspace/bignlp-scripts/prepare_dataset/the_pile/{file_number:02d}.jsonl "
        f"--output-prefix /workspace/bignlp-scripts/prepare_dataset/the_pile/my-gpt3_{file_number:02d} "
        f"--vocab /workspace/bignlp-scripts/prepare_dataset/bpe/vocab.json "
        f"--merge-file /workspace/bignlp-scripts/prepare_dataset/bpe/merges.txt "
        f"--dataset-impl mmap "
        f"--tokenizer-library megatron "
        f"--tokenizer-type GPT2BPETokenizer "
        f"--workers $SLURM_CPUS_ON_NODE "
        f"--append-eod "
    )
    cmd = f'cd /workspace/bignlp-scripts/NeMo; git rev-parse HEAD; cd /workspace/bignlp-scripts/NeMo/nemo/collections/nlp/data/language_modeling/megatron; make; export PYTHONPATH="/workspace/bignlp-scripts/NeMo/.:$PYTHONPATH"; CUDA_VISIBLE_DEVICES=0 python3 {code_path} {flags}'
    os.system(f"{cmd}")
    #os.remove(extracted_path)

"""
read -r -d '' cmd <<EOF
echo "*******STARTING********" \
&& echo "---------------" \
&& cd /gpfs/fs1/mausin/bignlp-scripts/NeMo \
&& git rev-parse HEAD \
&& cd nemo/collections/nlp/data/language_modeling/megatron \
&& make \
&& export PYTHONPATH="/gpfs/fs1/mausin/bignlp-scripts/NeMo/.:${PYTHONPATH}" \
&& CUDA_VISIBLE_DEVICES=0,1,2,3 python3 /gpfs/fs1/mausin/bignlp-scripts/NeMo/examples/nlp/language_modeling/preprocess_data_for_megatron.py
EOF
"""


if __name__ == "__main__":
    main()
