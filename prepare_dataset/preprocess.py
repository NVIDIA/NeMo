import os

import hydra


@hydra.main(config_path="../conf", config_name="config")
def main(cfg):
    bignlp_path = cfg["bignlp_path"]
    data_cfg = cfg["data_preparation"]
    data_save_dir = data_cfg.get("data_save_dir")
    assert data_save_dir is not None, "data_save_dir must be a valid path"
    full_data_save_dir = os.path.join(bignlp_path, data_save_dir)

    # Vocab
    vocab_dir = data_cfg.get("vocab_save_dir")
    assert vocab_dir is not None, "vocab_save_dir must be a valid path."
    vocab_path = os.path.join(bignlp_path, vocab_dir, "vocab.json")

    # Merges
    merges_dir = data_cfg.get("merges_save_dir")
    assert merges_dir is not None, "merges_save_dir must be a valid path."
    merges_path = os.path.join(bignlp_path, merges_dir, "merges.txt")

    file_number = int(os.environ.get("SLURM_ARRAY_TASK_ID"))
    extracted_path = os.path.join(full_data_save_dir, f"{file_number:02d}.jsonl")
    code_path = "/opt/bignlp/NeMo/examples/nlp/language_modeling/preprocess_data_for_megatron.py"
    output_prefix = os.path.join(full_data_save_dir, f"my-gpt3_{file_number:02d}")

    flags = (
        f"--input {extracted_path} "
        f"--output-prefix {output_prefix} "
        f"--vocab {vocab_path} "
        f"--merge-file {merges_path} "
        f"--dataset-impl mmap "
        f"--tokenizer-library megatron "
        f"--tokenizer-type GPT2BPETokenizer "
        f"--workers $SLURM_CPUS_ON_NODE "
        f"--append-eod "
    )
    cmd = f'cd /opt/bignlp/NeMo; git rev-parse HEAD; cd /opt/bignlp/NeMo/nemo/collections/nlp/data/language_modeling/megatron; make; export PYTHONPATH="/opt/bignlp/NeMo/.:$PYTHONPATH"; CUDA_VISIBLE_DEVICES=0,4,2,6,1,5,3,7 python3 {code_path} {flags}'
    os.system(f"{cmd}")
    os.remove(extracted_path)


if __name__ == "__main__":
    main()
