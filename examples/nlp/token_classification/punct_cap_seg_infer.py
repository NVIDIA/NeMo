import argparse
from typing import List

import torch

from nemo.collections.nlp.models import PunctCapSegModel


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    input_opts = parser.add_mutually_exclusive_group()
    input_opts.add_argument("--text-file")
    input_opts.add_argument("--sentence")
    parser.add_argument(
        "--num-passes",
        default=None,
        type=int,
        choices=[None, 1, 2, 3],
        help="Number of passes through the encoder. Will default to '1' for models trained in single-pass mode and '2' "
        "for model trained in multi-pass mode.",
    )
    parser.add_argument("--punctuation-threshold", type=float, default=0.5)
    parser.add_argument("--segmentation-threshold", type=float, default=0.5)
    parser.add_argument("--truecase-threshold", type=float, default=0.5)
    parser.add_argument("--max-length", type=int, default=None)
    args = parser.parse_args()
    return args


def main() -> None:
    args = get_args()

    m: PunctCapSegModel
    if args.model.endswith(".nemo"):
        m = PunctCapSegModel.restore_from(args.model, map_location=torch.device("cpu"))
    else:
        m = PunctCapSegModel.load_from_checkpoint(args.model, map_location="cpu")

    texts: List[str] = []
    if args.text_file is not None:
        with open(args.text_file) as f:
            for text in f:
                text = text.strip()
                # Skip comments
                if text.startswith("#"):
                    continue
                texts.append(text)
    else:
        # alternative is a single string on the command line
        texts = [args.sentence]

    processed_texts: List[List[str]] = m.infer(
        texts,
        batch_size=32,
        punct_threshold=args.punctuation_threshold,
        seg_threshold=args.segmentation_threshold,
        cap_threshold=args.truecase_threshold,
        max_length=args.max_length,
    )
    for i, text in enumerate(texts):
        print(f"Input {i}: {text}")
        print(f"Output:")
        for j, processed_text in enumerate(processed_texts[i]):
            print(f"    {processed_text}")


if __name__ == "__main__":
    main()
