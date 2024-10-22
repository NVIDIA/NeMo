#!/usr/bin/env python
from pathlib import Path

import click

from nemo.collections.common.tokenizers import CanaryTokenizer


@click.command()
@click.argument("output_dir", type=click.Path())
def main(output_dir: str) -> None:
    """
    Builds the special tokens tokenizer for NVIDIA Canary-1B model.
    It's intended to be used with CanaryTokenizer (a specialized AggregateTokenizer)
    under name ``spl_tokens``.
    """
    CanaryTokenizer.build_special_tokenizer(
        [
            "<|endoftext|>",
            "<|startoftranscript|>",
            "<|transcribe|>",
            "<|translate|>",
            "<|nopnc|>",
            "<|pnc|>",
            "<|nospeech|>",
        ]
        + [
            "<|en|>",
            "<|es|>",
            "<|de|>",
            "<|fr|>",
        ]
        + [f"<|spltoken{i}|>" for i in range(16)],
        model_dir=output_dir,
        force_rebuild=True,
    )


if __name__ == "__main__":
    main()
