import os

from utils import extract_single_zst_file


def main():
    data_save_dir = "/workspace/bignlp-scripts/prepare_dataset/the_pile"
    file_number = int(os.environ.get("SLURM_ARRAY_TASK_ID"))

    extracted_path = os.path.join(data_save_dir, f"{file_number:02d}.jsonl")
    code_path = "/workspace/bignlp-scripts/megatron-lm/tools/preprocess_data.py"

    flags = (
        f"--input /workspace/bignlp-scripts/prepare_dataset/the_pile/{file_number:02d}.jsonl "
        f"--output-prefix /workspace/bignlp-scripts/prepare_dataset/the_pile/my-gpt3_{file_number:02d} "
        f"--vocab /workspace/bignlp-scripts/prepare_dataset/bpe/vocab.json "
        f"--merge-file /workspace/bignlp-scripts/prepare_dataset/bpe/merges.txt "
        f"--dataset-impl mmap "
        f"--tokenizer-type GPT2BPETokenizer "
        f"--workers $SLURM_CPUS_ON_NODE "
        f"--append-eod "
    )
    os.system(f"python3 {code_path} {flags}")
    os.remove(extracted_path)


if __name__ == "__main__":
    main()
