import os
import hydra

from utils import extract_single_zst_file


@hydra.main(config_path="../conf", config_name="config")
def main(cfg) -> None:
    data_cfg = cfg["data_preparation"]
    data_save_dir = "/workspace/bignlp-scripts/prepare_dataset/the_pile"
    file_number = int(os.environ.get("SLURM_ARRAY_TASK_ID"))

    extracted_path = os.path.join(data_save_dir, f"{file_number:02d}.jsonl")
    code_path = "/workspace/bignlp-scripts/megatron-lm/tools/preprocess_data.py"

    flags = f"--input /workspace/bignlp-scripts/prepare_dataset/the_pile/{file_number:02d}.jsonl \
              --output-prefix /workspace/bignlp-scripts/prepare_dataset/the_pile/my-gpt3_{file_number:02d} \
              --vocab /workspace/bignlp_scripts/prepare_dataset/bpe/vocab.json \
	      --merge-file /workspace/bignlp_scripts/prepare_dataset/bpe/merges.txt \
	      --dataset-impl mmap \
	      --tokenizer-type GPT2BPETokenizer \
	      --workers $SLURM_CPUS_ON_NODE \
	      --append-eod "
    os.system(f"python3 {code_path} {flags}")

if __name__ == "__main__":
    main()
