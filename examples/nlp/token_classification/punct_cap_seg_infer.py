
import argparse
import json
from pathlib import Path
from typing import Dict, List, Union

import torch.cuda

from nemo.collections.nlp.models import PunctCapSegModel


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("text_file")
    parser.add_argument("--punctuation-threshold", type=float, default=0.5)
    parser.add_argument("--segmentation-threshold", type=float, default=0.5)
    parser.add_argument("--truecase-threshold", type=float, default=0.5)
    args = parser.parse_args()
    return args


def main() -> None:
    args = get_args()

    m: PunctCapSegModel
    if args.model.endswith(".nemo"):
        m = PunctCapSegModel.restore_from(args.model, map_location=torch.device("cpu"))
    else:
        m = PunctCapSegModel.load_from_checkpoint(args.model, map_location="cpu")
    m = m.eval().float()

    texts: List[str] = []
    with open(args.text_file) as f:
        for text in f:
            text = text.strip()
            # Skip comments
            if text.startswith("#"):
                continue
            texts.append(text)

    processed_texts: List[List[str]] = m.infer(
        texts,
        punct_threshold=args.punctuation_threshold,
        seg_threshold=args.segmentation_threshold
    )
    for i, text in enumerate(texts):
        print(f"Input {i}: {text}")
        print(f"Output:")
        for j, processed_text in enumerate(processed_texts[i]):
            print(f"    {processed_text}")
    # print(processed_texts)


if __name__ == "__main__":
    main()
