# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import re
from functools import partial
from tqdm import tqdm
from typing import Dict, List, Optional, Union

import lightning.pytorch as pl
import numpy as np
import torch
import torch.distributed as dist
from datasets import Dataset, DatasetDict, load_dataset
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn import functional as F

from nemo.utils import logging

# TODO Move it to utils or somethings for sake of clean code
CROSS_ENTROPY_IGNORE_IDX = -100
PACK_TYPE = Dict[str, Union[torch.Tensor, List[int]]]

def clean_split(name):
    """removes split from name

    Args:
        name (str): partition name (e.g. "train[:100]")

    Returns:
        str: return partition name without any selector (e.g. "train").
    """
    if "[" in name:
        name = name.split("[")[0]
    if '+' in name:
        name = name.split('+')[0]
    return name


def make_dataset_splits(dataset, split, split_aliases):
    """
    Given a dataset (e.g. from datasets.load_dataset or datasets.Dataset.from_dict) it
    returns a dictionary containing the corresponding dataset splits.

    For example:

    $ ds = load_dataset("dataset-id")
    $ ans = make_dataset_splits(ds)

    # `ds` contains the following
    $ print(ds)
    > DatasetDict({
    >    train: Dataset({
    >        features: ['id', 'title', 'context', 'question', 'answers'],
    >        num_rows: 87599
    >    })
    >    validation: Dataset({
    >        features: ['id', 'title', 'context', 'question', 'answers'],
    >        num_rows: 10570
    >    })
    > })

    # In this case the value of `ans` (returned value) will be:
    $ print(ans)
    > {
    >    "train": Dataset .. (with 87599 rows),
    >    "val": Dataset .. (with 10570 rows),
    > }
    """
    valid_split_names = ["train", "test", "val"]
    dataset_splits = {_split: None for _split in valid_split_names}

    alias_to_split = {}
    for split_name, _split_aliases in split_aliases.items():
        assert split_name in valid_split_names
        for alias in _split_aliases:
            alias_to_split[alias] = split_name
    for name in valid_split_names:
        alias_to_split[name] = name

    if isinstance(dataset, Dataset):
        assert isinstance(split, str), "Expected split to be a string, but got {}".format(type(split))
        split = clean_split(split)
        split = alias_to_split[split]
        dataset_splits[split] = dataset
    elif isinstance(dataset, DatasetDict):
        dataset_split_names = dataset.keys()
        logging.info("HF dataset has the following splits: {}".format(dataset_split_names))
        for alias_split_name, split in dataset.items():
            split_name = alias_to_split[alias_split_name]
            assert dataset_splits[split_name] is None
            dataset_splits[split_name] = split
    elif isinstance(split, list):
        logging.info("Loaded HF dataset will use {} splits.".format(split))
        assert isinstance(dataset, list)
        for i, alias_split_name in enumerate(map(clean_split, split)):
            split_name = alias_to_split[alias_split_name]
            assert dataset_splits[split_name] is None
            dataset_splits[split_name] = dataset[i]
    elif isinstance(split, str):
        logging.info("Loaded HF dataset has a single split.")
        assert not isinstance(dataset, list)
        alias_split_name = split
        if "+" in alias_split_name:
            raise ValueError("Split concatenation not supported")
        elif "[" in alias_split_name:
            alias_split_name = alias_split_name.split("[")[0]
        split_name = alias_to_split[alias_split_name]
        assert dataset_splits[split_name] is None
        dataset_splits[split_name] = dataset
    else:
        raise ValueError("Expected split name to be None, str or a list")

    assert set(valid_split_names) == set(dataset_splits.keys()), dataset_splits.keys()
    num_init_splits = sum(map(lambda x: x is not None, dataset_splits.values()))
    assert num_init_splits > 0, "Expected at least one split to have been initialized {}".format(num_init_splits)
    return dataset_splits


def has_dist_env_init_or_rank_env_var():
    """returns whether it runs on a dist-environment"""
    return dist.is_initialized() or int(os.environ.get('WORLD_SIZE', '0')) > 1


def batchify(tensor):
    """Ensures that the input tensor has at least two dimensions by adding an extra batch dimension if necessary.

    Parameters
    ----------
    tensor : torch.Tensor
        The input tensor to be batchified.

    Returns
    -------
    torch.Tensor
        The tensor with an extra dimension added if it was originally 1-dimensional.
        Otherwise, the tensor is returned as-is.
    """
    if tensor.ndim == 1:
        return tensor.unsqueeze_(0)
    return tensor


def extract_key_from_dicts(batch, key):
    """Extracts the value of the given key from each dictionary in a list of dictionaries.

    Parameters
    ----------
    batch : List[dict]
        A list of dictionaries.
    key : str
        The key whose values are to be extracted from each dictionary.

    Returns
    -------
    List
        A list of values associated with the specified key, in the same order as
        the dictionaries in the input batch.
    """
    return list(map(lambda x: x[key], batch))


def pad_within_micro(batch, pad_token_id, pad_seq_len_divisible=None):
    """Pads each list in a batch of lists to the same length with a specified token.

    Parameters
    ----------
    batch : List[List[int]]
        A batch of sequences (e.g., token IDs), where each sequence is a list of integers.
    pad_token_id : int
        The token ID to use for padding shorter sequences.
    pad_seq_len_divisible : int
        The value to use for padding sequence length so that it is divisible by pad_seq_len_divisible.
    Returns
    -------
    List[List[int]]
        A batch of sequences where each inner list has been padded with the pad token
        to match the length of the longest sequence in the batch.
    """
    max_len = max(map(len, batch))
    if pad_seq_len_divisible:
        max_len = (pad_seq_len_divisible - max_len % pad_seq_len_divisible) + max_len
    return [item + [pad_token_id] * (max_len - len(item)) for item in batch]


class HFDatasetDataModule(pl.LightningDataModule):
    """A PyTorch Lightning DataModule for loading and managing datasets from the `datasets` library.

    Args:
        path_or_dataset (str | Dataset | DatasetDict): The dataset name from HF or a preloaded dataset.
        split (str | list, optional): The dataset split(s) to load (e.g., "train" or ["train", "validation"]).
            Defaults to None.
        collate_fn (callable, optional): Custom function for batching data; defaults to a padding-based collation.
            Defaults to None.
        num_workers (int, optional): Number of workers for data loading. Defaults to 2.
        pin_memory (bool, optional): Whether to use pinned memory for faster GPU transfers. Defaults to True.
        persistent_workers (bool, optional): Whether to keep worker threads alive between epochs. Defaults to True.
        seq_length (int, optional): Maximum sequence length for tokenized inputs. Defaults to 1024.
        micro_batch_size (int, optional): Batch size per device. Defaults to 2.
        global_batch_size (int, optional): Total batch size across all devices. Defaults to 2.
        pad_token_id (int, optional): Token ID used for padding sequences. Defaults to 0.
        use_dist_sampler (bool, optional): Whether to enable distributed sampling. Defaults to False.
        train_aliases (list, optional): Alternative names for the training split. Defaults to ["train", "training"].
        test_aliases (list, optional): Alternative names for the test split. Defaults to ["test", "testing"].
        val_aliases (list, optional): Alternative names for the validation split.
            Defaults to ["val", "validation", "valid", "eval"].
        **kwargs: Additional arguments passed to `datasets.load_dataset`.

    Raises:
        ValueError: If `path_or_dataset` is not a valid dataset type (str, Dataset, or DatasetDict).

    Examples:
        Load a single split (train) from a dataset:
        ```python
        data_module = HFDatasetDataModule("rajpurkar/squad", split="train")
        ```

        Load multiple splits (train and validation):
        ```python
        data_module = HFDatasetDataModule("rajpurkar/squad", split=["train", "validation"])
        ```

        Use a preloaded dataset:
        ```python
        from datasets import load_dataset
        dataset = load_dataset("imdb")
        data_module = HFDatasetDataModule(dataset, split="train")
        ```

    Notes:
        - If `use_dist_sampler` is not enabled, but a distributed environment is detected,
        HFDatasetDataModule will use a distributed-sampler automatically.
        - If no collation function is provided, a default function with padding using `pad_token_id` is applied.
    """

    def __init__(
        self,
        path_or_dataset,
        split=None,
        collate_fn=None,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True,
        seq_length=1024,
        micro_batch_size=2,
        global_batch_size=2,
        pad_token_id=0,
        use_dist_sampler=False,
        train_aliases=["train", "training"],
        test_aliases=["test", "testing"],
        val_aliases=["val", "validation", "valid", "eval"],
        pad_seq_len_divisible=None,
        num_replicas=None,
        rank=None,
        **kwargs,
    ) -> None:
        super().__init__()
        assert pad_token_id is not None
        # A dataset usually will have several splits (e.g. train, val, test, etc).
        # We map synonym names to canonical names (train, test, val).
        # A synonym can be a prefix/suffixed word e.g. train <> training.
        split_aliases = {
            "train": train_aliases,
            "test": test_aliases,
            "val": val_aliases,
        }

        # self.dataset_splits will hold the actual dataset for each split.
        if isinstance(path_or_dataset, str):
            logging.info("Loading HF dataset from {}, this may take a moment.".format(path_or_dataset))
            dataset = load_dataset(path_or_dataset, split=split, **kwargs)
        elif isinstance(path_or_dataset, Dataset) or isinstance(path_or_dataset, DatasetDict):
            logging.info("Using passed HF dataset {}".format(path_or_dataset))
            dataset = path_or_dataset
        else:
            raise ValueError(
                "Expected `path_or_dataset` to be str, Dataset, DatasetDict, but got {}".format(type(path_or_dataset))
            )

        self.dataset_splits = make_dataset_splits(dataset, split, split_aliases)

        if collate_fn is None:
            self._collate_fn = lambda x: HFDatasetDataModule.collate_fn(
                x, pad_token_id=self.pad_token_id, pad_seq_len_divisible=pad_seq_len_divisible
            )
        else:
            self._collate_fn = collate_fn

        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        self.seq_length = seq_length
        self.micro_batch_size = micro_batch_size
        self.global_batch_size = global_batch_size
        self.pad_token_id = pad_token_id
        self.use_dist_sampler = use_dist_sampler
        self.pad_seq_len_divisible = pad_seq_len_divisible

        # TODO: refractor this
        self.num_replicas = num_replicas
        self.rank = rank

    @staticmethod
    def from_dict(dataset_dict, split, **kwargs):
        """wraps Dataset's from_dict method"""
        dataset = Dataset.from_dict(dataset_dict)
        return HFDatasetDataModule(path_or_dataset=dataset, split=split, **kwargs)

    @staticmethod
    def collate_fn(batch, pad_token_id=0, pad_seq_len_divisible=None):
        """Default batch collator"""
        # print("--len(batch)---", len(batch))
        # print("---batch[0].items()----", batch[0].items())
        # print("---len(batch[0]['input_ids'])----", len(batch[0]['input_ids']))
        # print("---len(batch[0]['loss_mask'])----", len(batch[0]['loss_mask']))
        return {
            key: batchify(
                torch.LongTensor(
                    pad_within_micro(
                        extract_key_from_dicts(batch, key),
                        pad_token_id if key != 'loss_mask' else 0,
                        pad_seq_len_divisible,
                    )
                )
            )
            for key in batch[0].keys()
        }

    def setup(self, stage: str):
        """setups sampler"""
        # Turn-on dist-sampler if the user is running inside a dist-env.
        if not self.use_dist_sampler and has_dist_env_init_or_rank_env_var():
            self.use_dist_sampler = True
            logging.info("Turning on distributed data sampler")

    def get_data_sampler(self, dataset):
        """returns the data sampler"""
        if self.use_dist_sampler:
            return DistributedSampler(dataset, num_replicas=self.num_replicas, rank=self.rank)
        else:
            return None

    def _make_dataloader(self, dataset, collate_fn=None):
        """Dataloader creator"""
        assert dataset is not None

        if collate_fn is None:
            collate_fn = lambda x: HFDatasetDataModule.collate_fn(
                x, pad_token_id=self.pad_token_id, pad_seq_len_divisible=self.pad_seq_len_divisible
            )

        return DataLoader(
            dataset,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            collate_fn=collate_fn,
            batch_size=self.micro_batch_size,
            sampler=self.get_data_sampler(dataset),
        )

    @property
    def train(self):
        """Returns the training partition"""
        return self.dataset_splits["train"]

    @property
    def val(self):
        """Returns the validation partition"""
        return self.dataset_splits["val"]

    @property
    def test(self):
        """Returns the test partition"""
        return self.dataset_splits["test"]

    def train_dataloader(self):
        """Returns the train dataloader"""
        return self._make_dataloader(self.train, self._collate_fn)

    def val_dataloader(self):
        """Returns the validation dataloader"""
        return self._make_dataloader(self.val, self._collate_fn)

    def test_dataloader(self):
        """Returns the test dataloader"""
        return self._make_dataloader(self.test, self._collate_fn)

    def map(self, function=None, split_names=None, **kwargs):
        """Maps a function to all/selected splits
        Additional arguments can be passed down to dataset's map via kwargs"""
        if isinstance(split_names, str):
            split_names = [split_names]
        elif isinstance(split_names, list):
            pass
        elif split_names is None:
            split_names = self.dataset_splits.keys()
        else:
            raise ValueError("split_names must None/str/list")

        for split_name in split_names:
            if not self.dataset_splits[split_name] is None:
                self.dataset_splits[split_name] = self.dataset_splits[split_name].map(function, **kwargs)

class HellaSwagHFDataModule(HFDatasetDataModule):
    """A data module for handling the HellaSwag dataset using HFDatasetDataModule."""

    def __init__(self, tokenizer, dataset_name="Rowan/hellaswag", *args, **kwargs):
        tokenizer.pad_token = tokenizer.eos_token
        self.pad_token_id = tokenizer.eos_id
        dataset = load_dataset(dataset_name)
        kwargs['split'] = 'train'
        super().__init__(HellaSwagHFDataModule.preprocess_dataset(tokenizer, 7500, dataset["train"]), *args, **kwargs)

    @staticmethod
    def preprocess(text):
        """Preprocesses text data by removing unwanted characters and artifacts."""
        text = text.strip()
        # NOTE: Brackets are artifacts of the WikiHow dataset portion of HellaSwag.
        text = text.replace(" [title]", ". ")
        text = re.sub("\\[.*?\\]", "", text)
        text = text.replace("  ", " ")
        return text

    @staticmethod
    def process_doc(doc):
        """Processes a document from the HellaSwag dataset into a structured format suitable for training."""
        ctx = doc["ctx_a"] + " " + doc["ctx_b"].capitalize()
        query = HellaSwagHFDataModule.preprocess(doc["activity_label"] + ": " + ctx)
        choices = [HellaSwagHFDataModule.preprocess(ending) for ending in doc["endings"]]
        gold = int(doc["label"])
        out_doc = {
            "query": query,
            "choices": choices,
            "gold": gold,
            "text": query + " " + choices[gold],
        }
        return out_doc

    # Note: I'm training the model causally not through multiclass classification.
    @staticmethod
    def preprocess_dataset(tokenizer, max_length, dataset, seed=42):
        """Preprocesses a dataset for training a language model."""
        # Format each prompt.
        print("Preprocessing dataset...")
        dataset = dataset.map(HellaSwagHFDataModule.process_doc)

        def preprocess(example, tokenizer, max_length):
            input_ids = tokenizer.text_to_ids(example["text"])
            if max_length > 0:
                input_ids = input_ids[:max_length]
            return dict(input_ids=input_ids, labels=input_ids[1:] + [-100])

        # Apply preprocessing to each example of the dataset & and remove "conversations" and "text" fields.
        _preprocessing_function = partial(preprocess, max_length=max_length, tokenizer=tokenizer)
        dataset = dataset.map(
            _preprocessing_function,
            batched=False,
        ).select_columns(["input_ids", "labels"])

        # Shuffle dataset.
        dataset = dataset.shuffle(seed=seed)

        return dataset

    def _convert_to_tensors(self, pack: PACK_TYPE) -> PACK_TYPE:
        """Converts a pack into tensors. Pack comes in as a dict of lists and is converted to tensors."""
        tensor_pack = {
            "tokens": torch.tensor(pack["tokens"], dtype=torch.long),
            "labels": torch.tensor(pack["labels"], dtype=torch.long),
            "input_pos": torch.tensor(pack["input_pos"], dtype=torch.long),
            "seq_lens": torch.tensor(pack["seq_lens"], dtype=torch.long),
        }
        if self.contains_loss_mask: tensor_pack["loss_mask"] = torch.tensor(pack["loss_mask"], dtype=torch.long)
        return tensor_pack

    def _pad_pack(self, pack: PACK_TYPE, padding_idx: int) -> PACK_TYPE:
        """Pads a pack to ``self.packed_sequence_size``."""
        # Pad tokens
        num_padding_tokens = self.packed_sequence_size - len(pack["tokens"])
        padded_tokens = F.pad(
            pack["tokens"],
            (0, num_padding_tokens),
            value=padding_idx,
        )

        # Pad labels
        padded_labels = F.pad(
            pack["labels"],
            (0, self.packed_sequence_size - len(pack["labels"])),
            value=CROSS_ENTROPY_IGNORE_IDX,
        )

        # Pad loss_mask
        #TODO loss_mask is very specific to Squad. For ex: Dolly does not have loss_mask.
        # Can it be gneralized ?
        if self.contains_loss_mask:
            padded_loss_mask = F.pad(
                pack["loss_mask"],
                    (0, self.packed_sequence_size - len(pack["loss_mask"])),
                    value=0,
            )

        # Add padding tokens as a last seq len to ensure sum is max_seq_len
        padded_seq_lens = (
            torch.cat([pack["seq_lens"], torch.tensor([num_padding_tokens])])
            if num_padding_tokens > 0
            else pack["seq_lens"]
        )

        # Pad input_pos continuing the sequence from last value
        # in input_pos
        # e.g. [0 1 2] -> [0 1 2 3 4 5] for self.packed_sequence_size = 6
        num_range = torch.arange(
            pack["input_pos"][-1] + 1,
            pack["input_pos"][-1] + self.packed_sequence_size - len(pack["input_pos"]) + 1,
        )
        # Clamp to max_seq_len - 1 to avoid out of bounds error
        clamped_num_range = torch.clamp(num_range, 0, self.packed_sequence_size - 1)
        padded_input_pos = torch.cat([pack["input_pos"], clamped_num_range])

        padded_pack = {
            "tokens": padded_tokens,
            "labels": padded_labels,
            "input_pos": padded_input_pos,
            "seq_lens": padded_seq_lens,
        }
        if self.contains_loss_mask: padded_pack["loss_mask"] = padded_loss_mask
        return padded_pack

    def _add_pack(self, pack: PACK_TYPE) -> None:
        """Processes, pads and adds a pack to ``self.packs``."""
        pack = self._convert_to_tensors(pack)
        pack = self._pad_pack(pack, padding_idx=self.padding_idx)
        self.packs.append(pack)

    def _split_and_add_pack(self, current_pack: PACK_TYPE) -> PACK_TYPE:
        """Splits the current pack at the boundary, processes it, adds it to ``self.packs`` and
        returns the start of the next pack."""

        if self.split_across_pack:
            boundary = self.packed_sequence_size
            # The last elem in ``seq_lens`` ensures that ``sum(seq_lens) == self.max_seq_len``
            leftover_seq_len = self.packed_sequence_size - sum(current_pack["seq_lens"][:-1])
            seq_len_padding = [leftover_seq_len] if leftover_seq_len > 0 else []
        else:
            boundary = self.previous_sample_boundary
            # If we aren't splitting across packs, we leave out the last sample b/c
            # it will go into the next pack
            seq_len_padding = []

        pack = {
            "tokens": current_pack["tokens"][:boundary],
            "labels": current_pack["labels"][:boundary],
            "input_pos": current_pack["input_pos"][:boundary],
            "seq_lens": current_pack["seq_lens"][:-1] + seq_len_padding,
        }
        if self.contains_loss_mask: pack["loss_mask"] = current_pack["loss_mask"][:boundary]

        # Process and add the pack
        self._add_pack(pack)

        # Return the length of the first sample in next pack if we are splitting across packs,
        # otherwise return the length of the last sample in the current pack
        next_seq_len = (
            len(current_pack["tokens"][boundary:])
            if self.split_across_pack
            else current_pack["seq_lens"][-1]
        )

        output_dict = {
            "tokens": current_pack["tokens"][boundary:],
            "labels": current_pack["labels"][boundary:],
            "input_pos": current_pack["input_pos"][boundary:],
            "seq_lens": [next_seq_len],
        }
        if self.contains_loss_mask:
            output_dict["loss_mask"] = current_pack["loss_mask"][boundary:]
        return output_dict


    def _should_stop_packing(self) -> bool:
        """If max packs is set, stop packing when we reach that number."""

        if self.max_packs is not None and len(self.packs) == self.max_packs:
            return True
        return False

    def pack(self, packed_sequence_size, split_across_pack=False, max_packs=None):
        """Iterate through the dataset. Use a buffer to hold samples until max_seq_len,
        then append the buffer to self.packs as a single "packed" sample. Continue
        until max_packs or end of dataset.

        Args:

        Returns:


        """
        self.packed_sequence_size = packed_sequence_size
        self.split_across_pack = split_across_pack
        self.max_packs = max_packs
        ## 'TODO' check if nemo's impl also does this padding
        self.padding_idx = 0 # Padding value to pack a sequence to self.packed_sequence_size
        self.contains_loss_mask = False

        # Only show progress bar on rank 0
        rank = (
            torch.distributed.get_rank()
            if torch.distributed.is_available() and torch.distributed.is_initialized()
            else 0
        )

        # Pack dataset in each split of self.dataset_splits (i.e 'train', 'val', 'test')
        for split, ds in self.dataset_splits.items():
            if ds is None:
                continue
            self.packs: List[PACK_TYPE] = []
            if "loss_mask" in ds[0]:
                self.contains_loss_mask = True
            # Buffer to hold samples until they are long enough to be added to self.packs
            current_pack = {
                "tokens": [],
                "labels": [],
                "input_pos": [],
                "seq_lens": [],
            }
            if self.contains_loss_mask: current_pack["loss_mask"] = []
            self.previous_sample_boundary: int = 0
            if rank == 0:
                pbar = tqdm(total=len(ds), desc=f"Packing {split} dataset", dynamic_ncols=True)
            for sample in ds:
                #print("-----sample----",sample)
                tokens, labels = sample["input_ids"], sample["labels"]
                if self.contains_loss_mask:
                    loss_mask = sample["loss_mask"]
                # If the dataset outputs samples that are larger than the specified
                # packed_sequence_size and we're unable to split it, user needs to modify
                # one of the two parameters
                seq_len = len(tokens)
                if seq_len > packed_sequence_size and not split_across_pack:
                    raise ValueError(
                        f"Dataset sample is too long ({seq_len} > {self.max_seq_len}). "
                        "Please set `split_across_pack=True` or increase `max_seq_len`."
                    )
                # Update the current pack
                # "input_pos" is the pos ids, "seq_lens" is the len of each seq within the pack
                current_pack["tokens"] += tokens
                current_pack["labels"] += labels
                current_pack["input_pos"] += [x % packed_sequence_size for x in range(seq_len)]
                current_pack["seq_lens"] += [seq_len]
                if self.contains_loss_mask:
                    current_pack["loss_mask"] += loss_mask

                # If the current pack is over the packed_sequence_size, add it to self.packs and
                # retain any truncated or bumped samples for next pack
                while (
                    len(current_pack["tokens"]) > packed_sequence_size
                    and not self._should_stop_packing()
                ):
                    current_pack = self._split_and_add_pack(current_pack)

                if rank == 0:
                    pbar.update()

                # Keep track of previous sample boundary
                self.previous_sample_boundary = len(current_pack["tokens"])

                if self._should_stop_packing():
                    break

            # Handle the last pack if there's leftover and we haven't filled up the max packs
            if len(current_pack["tokens"]) > 0 and (
                self.max_packs is None or len(self.packs) < self.max_packs
            ):
                # No need to handle splitting at this point so we can just add the current pack
                self._add_pack(current_pack)

            # After packing all samples, convert self.packs to a Dataset object
            packed_dataset = Dataset.from_dict({
                key: [pack[key] for pack in self.packs] for key in self.packs[0].keys()
            })

            # Assign the packed dataset to self.dataset_splits[split]
            self.dataset_splits[split] = packed_dataset


class HellaSwagHFDataModule(HFDatasetDataModule):
    """A data module for handling the HellaSwag dataset using HFDatasetDataModule."""

    def __init__(self, tokenizer, dataset_name="Rowan/hellaswag", *args, **kwargs):
        tokenizer.pad_token = tokenizer.eos_token
        self.pad_token_id = tokenizer.eos_id
        dataset = load_dataset(dataset_name)
        super().__init__(HellaSwagHFDataModule.preprocess_dataset(tokenizer, 7500, dataset["train"]), *args, **kwargs)

    @staticmethod
    def preprocess(text):
        """Preprocesses text data by removing unwanted characters and artifacts."""
        text = text.strip()
        # NOTE: Brackets are artifacts of the WikiHow dataset portion of HellaSwag.
        text = text.replace(" [title]", ". ")
        text = re.sub("\\[.*?\\]", "", text)
        text = text.replace("  ", " ")
        return text

    @staticmethod
    def process_doc(doc):
        """Processes a document from the HellaSwag dataset into a structured format suitable for training."""
        ctx = doc["ctx_a"] + " " + doc["ctx_b"].capitalize()
        query = HellaSwagHFDataModule.preprocess(doc["activity_label"] + ": " + ctx)
        choices = [HellaSwagHFDataModule.preprocess(ending) for ending in doc["endings"]]
        gold = int(doc["label"])
        out_doc = {
            "query": query,
            "choices": choices,
            "gold": gold,
            "text": query + " " + choices[gold],
        }
        return out_doc

    # Note: I'm training the model causally not through multiclass classification.
    @staticmethod
    def preprocess_dataset(tokenizer, max_length, dataset, seed=42):
        """Preprocesses a dataset for training a language model."""
        # Format each prompt.
        print("Preprocessing dataset...")
        dataset = dataset.map(HellaSwagHFDataModule.process_doc)

        def preprocess_batch(batch, tokenizer, max_length):
            ans = tokenizer(
                batch["text"],
                max_length=max_length,
                truncation=True,
            )
            ans["labels"] = [x[1:] + [-100] for x in ans["input_ids"]]
            return ans

        # Apply preprocessing to each batch of the dataset & and remove "conversations" and "text" fields.
        _preprocessing_function = partial(preprocess_batch, max_length=max_length, tokenizer=tokenizer)
        dataset = dataset.map(
            _preprocessing_function,
            batched=True,
        ).select_columns(["input_ids", "attention_mask", "labels"])

        # Shuffle dataset.
        dataset = dataset.shuffle(seed=seed)

        return dataset

class SquadHFDataModule(HFDatasetDataModule):
    """
    A data module for handling the SQuAD dataset using HFDatasetDataModule.

    This class is responsible for tokenizing and formatting the SQuAD dataset for training
    language models. It extends `HFDatasetDataModule` and implements a prompt-based
    formatting function suitable for causal language modeling.

    Attributes:
        tokenizer: A tokenizer instance used to convert text into token IDs.
    """

    def __init__(self, tokenizer, **kwargs):
        """
        Initializes the SquadHFDataModule.

        Args:
            tokenizer: A tokenizer instance for processing text data.
            **kwargs: Additional arguments passed to the parent class (`HFDatasetDataModule`).
        """
        super().__init__(**kwargs)
        self.tokenizer = tokenizer
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.pad_token_id = self.tokenizer.eos_id

    def formatting_prompts_func(self, example):
        """
        Formats a given example into a structured prompt for training.

        This method converts a dataset example (containing context, question, and answer)
        into a structured format, tokenizes it, and prepares input IDs and labels for
        training a language model.

        Args:
            example (dict): A dictionary containing the following keys:
                - 'context': The passage from which the question is derived.
                - 'question': The question about the passage.
                - 'answers': A dictionary with a 'text' key containing the answer(s).

        Returns:
            dict: A dictionary containing:
                - 'input_ids': Tokenized input sequence (excluding the last token).
                - 'labels': Tokenized output sequence (excluding the first token).
                - 'loss_mask': A mask indicating which tokens contribute to the loss.
        """
        formatted_text = [
            f"Context: {example['context']} Question: {example['question']} Answer:",
            f" {example['answers']['text'][0].strip()}",
        ]
        context_ids, answer_ids = list(map(self.tokenizer.text_to_ids, formatted_text))
        if len(context_ids) > 0 and context_ids[0] != self.tokenizer.bos_id:
            context_ids.insert(0, self.tokenizer.bos_id)
        if len(answer_ids) > 0 and answer_ids[-1] != self.tokenizer.eos_id:
            answer_ids.append(self.tokenizer.eos_id)

        return dict(
            labels=(context_ids + answer_ids)[1:],
            input_ids=(context_ids + answer_ids)[:-1],
            loss_mask=[0] * (len(context_ids) - 1) + [1] * len(answer_ids),
        )

    def setup(self, stage):
        """
        Prepares the dataset for training and applies formatting.

        Args:
            stage (str): The stage of training.
        """
        super().setup(stage)

        self.map(
            self.formatting_prompts_func,
            batched=False,
            batch_size=2,
            remove_columns=["id", "title", "context", "question", "answers"],
        )


class HFMockDataModule(pl.LightningDataModule):
    """A PyTorch Lightning DataModule for generating mock data for testing purposes."""

    def __init__(
        self,
        seq_length: int = 2048,
        vocab_size: int = 1024,
        micro_batch_size: int = 4,
        global_batch_size: int = 8,
        rampup_batch_size=None,
        num_train_samples: int = 10_000,
        num_val_samples: int = 10_000,
        num_test_samples: int = 10_000,
        num_workers: int = 8,
        pin_memory: bool = True,
        persistent_workers: bool = False,
        create_attention_mask: bool = False,
        vocab_file=None,
        merges_file=None,
        pad_seq_len_divisible=None,
    ):
        super().__init__()
        self.seq_length = seq_length
        self.micro_batch_size = micro_batch_size
        self.global_batch_size = global_batch_size
        self.num_train_samples = num_train_samples
        self.num_val_samples = num_val_samples
        self.num_test_samples = num_test_samples
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        self.create_attention_mask = create_attention_mask
        self.collate_fn = lambda x: HFDatasetDataModule.collate_fn(x, pad_token_id=0)
        self.vocab_size = vocab_size
        if pad_seq_len_divisible is not None:
            self.seq_length = (seq_length + pad_seq_len_divisible - 1) // pad_seq_len_divisible * pad_seq_len_divisible

    def setup(self, stage: str = None) -> None:
        """setup"""
        self._train_ds = _MockGPTDataset(
            self.vocab_size,
            "train",
            self.num_train_samples,
            self.seq_length,
            self.create_attention_mask,
        )
        self._val_ds = _MockGPTDataset(
            self.vocab_size,
            "valid",
            self.num_val_samples,
            self.seq_length,
            self.create_attention_mask,
        )
        self._test_ds = _MockGPTDataset(
            self.vocab_size,
            "test",
            self.num_test_samples,
            self.seq_length,
            self.create_attention_mask,
        )

    def train_dataloader(self) -> DataLoader:
        """train_dataloader"""
        return self._create_dataloader(self._train_ds)

    def val_dataloader(self) -> DataLoader:
        """val_dataloader"""
        return self._create_dataloader(self._val_ds)

    def test_dataloader(self) -> DataLoader:
        """test_dataloader"""
        return self._create_dataloader(self._test_ds)

    def _create_dataloader(self, dataset) -> DataLoader:
        """creates the dataloader for given dataset"""
        return DataLoader(
            dataset,
            batch_size=self.micro_batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            collate_fn=self.collate_fn,
        )


class _MockGPTDataset(torch.utils.data.Dataset):
    """A mock dataset for generating random data for testing purposes."""

    def __init__(
        self,
        vocab_size: int,
        name: str,
        num_samples: int,
        seq_length: int,
        create_attention_mask: bool = False,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.name = name
        self.seq_length = seq_length
        self.vocab_size = vocab_size
        self.length = num_samples
        self.seed = seed
        self.create_attention_mask = create_attention_mask

        if create_attention_mask:
            self.attention_mask = np.tril(np.ones((self.seq_length, self.seq_length), dtype=np.float32))[
                np.newaxis, :
            ].tolist()

        self.loss_mask = np.ones(self.seq_length, dtype=np.float32).tolist()
        self.position_ids = np.arange(self.seq_length, dtype=np.int64).tolist()

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx) -> Dict[str, list]:
        np_gen = np.random.default_rng(seed=(self.seed + idx))
        tokens = np_gen.integers(self.vocab_size, size=[self.seq_length], dtype=np.int64).tolist()
        labels = np_gen.integers(self.vocab_size, size=[self.seq_length], dtype=np.int64).tolist()

        batch = {
            "tokens": tokens,
            "labels": labels,
            "loss_mask": self.loss_mask,
            "position_ids": self.position_ids,
        }

        if self.create_attention_mask:
            batch["attention_mask"] = self.attention_mask

        return batch
