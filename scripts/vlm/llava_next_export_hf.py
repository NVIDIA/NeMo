import argparse
from pathlib import Path
from nemo.collections import llm


def main():
    parser = argparse.ArgumentParser(description="Export NeMo checkpoint to Hugging Face format.")
    parser.add_argument(
        "--path",
        type=str,
        default="/root/.cache/nemo/models/llava-hf/llava-v1.6-vicuna-7b-hf",
        help="Path to the NeMo checkpoint directory. (Default: /root/.cache/nemo/models/llava-hf/llava-v1.6-vicuna-7b-hf)",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="path/to/converted/hf/ckpt",
        help="Path to save the converted Hugging Face checkpoint. (Default: path/to/converted/hf/ckpt)",
    )

    args = parser.parse_args()

    llm.export_ckpt(
        path=Path(args.path),
        target='hf',
        output_path=Path(args.output_path),
    )


if __name__ == '__main__':
    main()
