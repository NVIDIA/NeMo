from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from nemo.collections.common.tokenizers import TokenizerSpec
from nemo.collections.llm.gpt.data.core import create_sft_dataset
from nemo.utils import logging
from nemo.utils.sequence_packing_utils import create_hist, create_packing_strategy, fill_packing_strategy


def tokenize_dataset(path: Path, tokenizer: TokenizerSpec, max_seq_length: int, seed: int):
    dataset = create_sft_dataset(
        path=path,
        tokenizer=tokenizer,
        seq_length=max_seq_length,
        seed=seed,
        is_test=True,
    )
    return np.array([dataset[i] for i in range(len(dataset))])


def prepare_packed_sequence_data(
    input_path: Path,
    output_path: Path,
    packed_sequence_size: int,
    tokenizer: TokenizerSpec,
    max_seq_length: int,
    seed: Optional[int] = 0,
    packing_algorithm: str = "first_fit_shuffle",
):
    logging.info(f"Preparing packed sequence from {input_path}")
    dataset = tokenize_dataset(input_path, tokenizer, max_seq_length, seed)
    sequences, histogram = create_hist(dataset, max_seq_length)

    assignments = create_packing_strategy(histogram, packed_sequence_size, packing_algorithm)
    output_data = fill_packing_strategy(assignments, sequences, packed_sequence_size)

    # save output data
    np.save(output_path, output_data)
    logging.info(f"Packed sequence is prepared and saved to {output_path}")


def manipulate_batch_to_mbs1(batch: List[Dict[str, List]], micro_batch_size: int) -> List[Dict[str, List]]:
    """
    Manipulate batch to mbs=1 by concatenating samples in a micro batch.

    This function is called before the input to GPTSFTPackedDataset.collate_fn.
    As such, `batch` is a list of length `global_batch_size`, and each element is a dictionary containing
    `input_ids`, `seq_boundaries`, and `loss_mask` (only if answer_only_loss=True).

    Returns `batch` in the same format, with length shrunk by a factor of mbs.
    """
    new_batch = []

    for i, sample in enumerate(batch):
        if i % micro_batch_size == 0:
            # Beginning of new micro batch
            new_batch.append(
                {
                    "input_ids": sample["input_ids"],
                    "seq_boundaries": sample["seq_boundaries"],
                    "loss_mask": sample["loss_mask"],
                }
            )
        else:
            # Middle of new micro batch
            new_batch[-1]["seq_boundaries"].extend(
                np.array(sample["seq_boundaries"]) + len(new_batch[-1]["input_ids"])
            )
            new_batch[-1]["input_ids"].extend(sample["input_ids"])
            new_batch[-1]["loss_mask"].extend(sample["loss_mask"])

    return new_batch
