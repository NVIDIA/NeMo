import os


def main():
    code_path = "/workspace/bignlp-scripts/NeMo/examples/nlp/language_modeling/megatron_gpt_pretraining.py"
    flags = (
        f"--config-path=/workspace/bignlp-scripts/conf/training "
        f"--config-name=126m "
    )
    cmd = f'cd /workspace/bignlp-scripts/NeMo; git rev-parse HEAD; cd /workspace/bignlp-scripts/NeMo/nemo/collections/nlp/data/language_modeling/megatron; make; export PYTHONPATH="/workspace/bignlp-scripts/NeMo/.:$PYTHONPATH"; CUDA_VISIBLE_DEVICES=0 python3 {code_path} {flags}'
    os.system(f"{cmd}")


if __name__ == "__main__":
    main()
