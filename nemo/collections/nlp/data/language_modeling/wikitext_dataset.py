import itertools
import os

import torch
from datasets import load_dataset

from nemo.core.classes import Dataset


def fixed_seq_length_of_datasets(
    datasets, fixed_seq_length, load_from_cache_file=False,
):
    block_size = fixed_seq_length

    # Main data processing function that will concatenate all texts from our dataset and generate chunks of block_size.
    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: list(itertools.chain(*examples[k])) for k in examples.keys()}
        input_ids = concatenated_examples["input_ids"]
        total_length = len(input_ids)

        if total_length < block_size:
            return {"input_ids": []}

        # Cut off the excess tokens to align it with the group size.
        if total_length % block_size != 0:
            input_ids = input_ids[: total_length - total_length % block_size]

        # Split by chunks of max_len.
        result = {"input_ids": [input_ids[i : i + block_size] for i in range(0, total_length, block_size)]}

        return result

    lm_datasets = datasets.map(
        group_texts,
        batched=True,
        num_proc=os.cpu_count(),
        load_from_cache_file=load_from_cache_file,
        desc=f"Grouping texts in chunks of {block_size}",
    )
    lm_datasets = lm_datasets.filter(
        function=lambda batch: [len(ids) > 0 for ids in batch["input_ids"]],
        batched=True,
        num_proc=os.cpu_count(),
        load_from_cache_file=load_from_cache_file,
        desc=f"Delete empty ids generated in the previous process.",
    )

    return lm_datasets


class WikitextDataset(Dataset):
    def __init__(
        self, tokenizer, seq_length, split="test", name=None, load_from_cache_file=True,
    ):
        """HuggingFace's WikiText dataset.
        link: https://huggingface.co/datasets/wikitext
        """
        super().__init__()
        self.seq_length = seq_length
        self.name = name
        raw_dataset = load_dataset("wikitext", name, split=split)
        column_names = raw_dataset.column_names

        def tokenize(examples):
            ids = []
            for text in examples["text"]:
                id_or_ids = tokenizer.text_to_ids(text)
                if id_or_ids is not list:
                    id_or_ids = [id_or_ids]
                ids += id_or_ids
            return ids

        tokenized_dataset = raw_dataset.map(
            lambda examples: {"input_ids": tokenize(examples)},
            batched=True,
            remove_columns=column_names,
            num_proc=os.cpu_count(),
            load_from_cache_file=load_from_cache_file,
            desc="Running tokenizer on dataset",
        )

        lm_dataset = fixed_seq_length_of_datasets(
            tokenized_dataset, seq_length, load_from_cache_file=load_from_cache_file,
        )
        self.dataset = lm_dataset
        self.attention_mask = torch.tril(torch.ones((self.seq_length, self.seq_length))).unsqueeze(0)
        self.attention_mask = self.attention_mask < 0.5
        self.loss_mask = torch.ones(self.seq_length, dtype=torch.float)
        self.position_ids = torch.arange(self.seq_length, dtype=torch.int64)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        data_item = dict(
            tokens=torch.tensor(item["input_ids"], dtype=torch.int64),
            labels=torch.tensor(item["input_ids"], dtype=torch.int64),
            attention_mask=self.attention_mask,
            loss_mask=self.loss_mask,
            position_ids=self.position_ids,
        )

        return data_item
