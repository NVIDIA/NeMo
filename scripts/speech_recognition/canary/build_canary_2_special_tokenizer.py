#!/usr/bin/env python
import math

import click

from nemo.collections.common.tokenizers import CanaryTokenizer


@click.command()
@click.argument("output_dir", type=click.Path())
def main(output_dir: str) -> None:
    """
    Builds the special tokens tokenizer for NVIDIA Canary-2.0 model.
    It's intended to be used with CanaryTokenizer (a specialized AggregateTokenizer)
    under name ``spl_tokens``.
    """

    tokens = (
        [
            # Generic special tokens
            "<|endoftext|>",
            "<|startoftranscript|>",
            "<|nopnc|>",
            "<|pnc|>",
            "<|nospeech|>",
            "<|startofcontext|>",
            "<|itn|>",
            "<|noitn|>",
            "<|timestamp|>",
            "<|notimestamp|>",
            "<|diarize|>",
            "<|nodiarize|>",
            "<|spkchange|>",
            "<|audioseparator|>",
            "<|emo:undefined|>",
            "<|emo:neutral|>",
            "<|emo:happy|>",
            "<|emo:sad|>",
            "<|emo:angry|>",
        ]
        + [
            # Language special tokens
            "<|unklang|>",
            "<|ar-AR|>",
            "<|cs-CZ|>",
            "<|da-DA|>",
            "<|de-DE|>",
            "<|en-US|>",
            "<|en-GB|>",
            "<|es-US|>",
            "<|es-ES|>",
            "<|fr-CA|>",
            "<|fr-FR|>",
            "<|hi-IN|>",
            "<|he-IL|>",
            "<|it-IT|>",
            "<|ja-JP|>",
            "<|ko-KR|>",
            "<|nb-NO|>",
            "<|nl-NL|>",
            "<|nn-NO|>",
            "<|pl-PO|>",
            "<|pt-PT|>",
            "<|pt-BR|>",
            "<|ru-RU|>",
            "<|sv-SW|>",
            "<|th-TH|>",
            "<|tr-TR|>",
            "<|zh-CN|>",
        ]
        + [
            # Timestamp frame special tokens
            f"<|{i}|>"
            for i in range(900)
        ]
        + [
            # Speaker indicator special tokens
            f"<|spk{i}|>"
            for i in range(16)
        ]
    )

    num_tokens = len(tokens) + 3  # count "<pad>", "<unk>", "_" too
    print(f"We have {num_tokens} special tokens.")
    next_pow_of_2 = next_power_of_2(num_tokens)
    num_extra_tokens = next_pow_of_2 - num_tokens
    print(f"Adding extra {num_extra_tokens} unused special tokens for a total vocab size of {next_pow_of_2}")

    tokens += [
        # Timestamp related special tokens
        f"<|spltoken{i}|>"
        for i in range(num_extra_tokens)
    ]

    tokenizer = CanaryTokenizer.build_special_tokenizer(
        tokens=tokens,
        model_dir=output_dir,
        force_rebuild=True,
    )

    assert tokenizer.vocab_size == 1024, tokenizer.vocab_size


def next_power_of_2(n):
    if n <= 0:
        return 1
    return 2 ** math.ceil(math.log2(n))


if __name__ == "__main__":
    main()
