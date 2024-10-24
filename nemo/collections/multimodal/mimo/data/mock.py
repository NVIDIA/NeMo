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

from typing import Dict, List, Optional, Tuple
from nemo.collections.common.tokenizers.huggingface.auto_tokenizer import AutoTokenizer
import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from torch.utils import data
from torch.utils.data import DataLoader, Dataset

from nemo.lightning.pytorch.plugins import MegatronDataSampler


class MockDataModule(pl.LightningDataModule):
    def __init__(
        self,
        tokenizer,
        seq_length: int = 2048,
        decoder_seq_length: Optional[int] = 2048,
        vocab_size: int = 128256,
        crop_size: Tuple[int, int] = (336, 336),
        micro_batch_size: int = 4,
        global_batch_size: int = 8,
        rampup_batch_size: Optional[List[int]] = None,
        num_train_samples: int = 10_000,
        num_val_samples: int = 10_000,
        num_test_samples: int = 10_000,
        num_workers: int = 8,
        pin_memory: bool = True,
        persistent_workers: bool = False,
    ):
        super().__init__()
        self.tokenizer=tokenizer
        self.seq_length = seq_length
        self.decoder_seq_length = decoder_seq_length
        self.num_train_samples = num_train_samples
        self.num_val_samples = num_val_samples
        self.num_test_samples = num_test_samples
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        self.vocab_size = vocab_size
        self.crop_size = crop_size
        self.micro_batch_size = micro_batch_size

        self.data_sampler = MegatronDataSampler(
            seq_len=self.seq_length,
            micro_batch_size=micro_batch_size,
            global_batch_size=global_batch_size,
            rampup_batch_size=rampup_batch_size,
        )

    def setup(self, stage: str = "") -> None:
        self._train_ds = _MockMimoDataset(
            self.vocab_size, self.tokenizer, self.crop_size, "train", self.num_train_samples, self.decoder_seq_length
        )
        self._validation_ds = _MockMimoDataset(
            self.vocab_size, self.tokenizer, self.crop_size, "valid", self.num_val_samples, self.decoder_seq_length
        )
        self._test_ds = _MockMimoDataset(
            self.vocab_size,self.tokenizer, self.crop_size, "test", self.num_test_samples, self.decoder_seq_length
        )

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        if not hasattr(self, "_train_ds"):
            self.setup()
        return self._create_dataloader(self._train_ds)

    def val_dataloader(self) -> EVAL_DATALOADERS:
        if not hasattr(self, "_validation_ds"):
            self.setup()
        return self._create_dataloader(self._validation_ds)

    def test_dataloader(self) -> EVAL_DATALOADERS:
        if not hasattr(self, "_test_ds"):
            self.setup()
        return self._create_dataloader(self._test_ds)

    def _create_dataloader(self, dataset, **kwargs) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size  = self.micro_batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            collate_fn=dataset.collate_fn,
            **kwargs,
        )

class _MockMimoDataset(Dataset):
    def __init__(
        self,
        vocab_size,
        tokenizer,
        crop_size,
        name: str,
        num_samples: int,
        seq_length: int,
        ignore_placeholder: int = -100,  # Placeholder for ignored tokens
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.name = name
        self.seq_length = seq_length
        self.tokenizer = tokenizer
        self.ignore_placeholder = ignore_placeholder

        self.vocab_size = vocab_size
        self.image_height, self.image_width = crop_size
        self.length = num_samples
        self.seed = seed

        self.input_text = "Generate image of dog."
        self.label_text = "Here is the image of dog"
        self.special_tokens = [f"IMG_{i}" for i in range(8)]
    def __len__(self) -> int:
        return self.length

    def tokenize_text(self, text: str) -> torch.Tensor:
        """Tokenize the input text using the provided tokenizer."""
        tokens = self.tokenizer.tokenizer(
            text,return_tensors="pt"
        )
        return tokens["input_ids"].squeeze(0)  # Return as 1D tensor

    def find_pattern_indices(self,template, pattern, search_start_index=0, allow_first_token_mismatch=False):
        template_len = len(template)
        pattern_len = len(pattern)
        for i in range(search_start_index, template_len - pattern_len + 1):
            match = template[i : i + pattern_len] == pattern
            if torch.all(match) or (allow_first_token_mismatch and torch.all(match[1:])):
                return i, i + pattern_len
        return -1, -1

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        # Generate images with normal distribution
        np_gen = np.random.default_rng(seed=(self.seed + idx))
        images = torch.from_numpy(np_gen.standard_normal((3, self.image_height, self.image_width))).float()

        # Tokenize input text and label text
        input_tokens = self.tokenize_text(self.input_text)
        label_tokens = self.tokenize_text(self.label_text)

        # Manually add special tokens (IMG_1, IMG_2, etc.) after tokenization
        special_token_ids = [self.tokenizer.tokenizer.convert_tokens_to_ids(token) for token in self.special_tokens]
        label_tokens = torch.cat([label_tokens, torch.tensor(special_token_ids, dtype=torch.long)])

        # Combine input and label tokens into one sequence
        combined_tokens = torch.cat([input_tokens, label_tokens])

        # Create labels with placeholder values initially
        labels = torch.ones_like(combined_tokens) * self.ignore_placeholder

        # Assign labels for the answer portion (the part containing the label text and special tokens)
        answer_start = len(input_tokens)
        labels[answer_start:] = combined_tokens[answer_start:]

        # Shift tokens and labels for next-token prediction
        # tokens = combined_tokens[:-1]
        # labels = labels[1:]
        #inseting dummy image token
        tokens = torch.cat([torch.tensor([-200]), combined_tokens[:-1]])

        # Adjust labels: Insert -100 at index 0 (used as ignored index)
        labels = torch.cat([torch.tensor([self.ignore_placeholder]), labels[1:]])

        # Adjust loss mask: Insert 0.0 at index 0 (to ignore this position in loss computation)
        # loss_mask = torch.cat([torch.tensor([0.0]), (labels != self.ignore_placeholder).float()])
        loss_mask = (labels != self.ignore_placeholder).float()

        return {
            "images": images,
            "tokens": tokens,
            "position_ids": torch.arange(len(tokens), dtype=torch.int64),
            "labels": labels,
            "loss_mask": loss_mask,
            "input_text": self.input_text
        }


    def _collate_fn(self, batch):
        """Default collation function for the dataloader."""
        collated_batch = {}
        collated_batch["attention_mask"] = None
        collated_batch.update(data.dataloader.default_collate(batch))
        return collated_batch

    def collate_fn(self, batch):
        """Method to use as the `collate_fn` in DataLoader."""
        return self._collate_fn(batch)
    
if __name__ == "__main__":
    tokenizer = AutoTokenizer("llava-hf/llava-v1.6-vicuna-7b-hf")
    special_tokens = [f"IMG_{i}" for i in range(8)]
    tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
    data_module = MockDataModule(tokenizer = tokenizer,vocab_size = tokenizer.vocab_size, micro_batch_size=1 )
    data_module.setup()
    dataloader = data_module.test_dataloader()
    batch = next(iter(dataloader))

    print("Sample Batch:")
    print("Images:", batch["images"].shape)
    print("Tokens:", batch["tokens"])
    print("Position IDs:", batch["position_ids"])
    print("Labels:", batch["labels"])
    print("Loss Mask:", batch["loss_mask"])
    print("Input text:", batch["input_text"])