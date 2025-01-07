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
# limitations under the License. Add some stuff

import os
from typing import Dict, List, Optional, Tuple

import lightning.pytorch as pl
import numpy as np
import torch
from diffusers.image_processor import VaeImageProcessor
from PIL import Image
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from torch.utils import data
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataloader import default_collate
from torchvision import transforms
from transformers import AutoProcessor

from nemo.collections.common.tokenizers.huggingface.auto_tokenizer import AutoTokenizer
from nemo.lightning.pytorch.plugins import MegatronDataSampler


class MockDataModule(pl.LightningDataModule):
    def __init__(
        self,
        tokenizer,
        image_processor,
        stage="encoder_alignment",
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
        num_workers: int = 1,
        pin_memory: bool = True,
        persistent_workers: bool = False,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.stage = stage
        self.image_processor = image_processor
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
        self.global_batch_size = global_batch_size

        self.data_sampler = MegatronDataSampler(
            seq_len=self.seq_length,
            micro_batch_size=micro_batch_size,
            global_batch_size=global_batch_size,
            rampup_batch_size=rampup_batch_size,
        )

    def setup(self, stage: str = "") -> None:
        self._train_ds = _MockMimoDataset(
            self.stage,
            self.vocab_size,
            self.tokenizer,
            self.image_processor,
            self.crop_size,
            "train",
            self.num_train_samples,
            self.decoder_seq_length,
        )
        self._validation_ds = _MockMimoDataset(
            self.stage,
            self.vocab_size,
            self.tokenizer,
            self.image_processor,
            self.crop_size,
            "valid",
            self.num_val_samples,
            self.decoder_seq_length,
        )
        self._test_ds = _MockMimoDataset(
            self.stage,
            self.vocab_size,
            self.tokenizer,
            self.image_processor,
            self.crop_size,
            "test",
            self.num_test_samples,
            self.decoder_seq_length,
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
            batch_size=self.micro_batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            collate_fn=dataset.collate_fn,
            **kwargs,
        )


class _MockMimoDataset(Dataset):
    def __init__(
        self,
        stage,
        vocab_size,
        tokenizer,
        image_processor,
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
        self.stage = stage
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.ignore_placeholder = ignore_placeholder

        self.vocab_size = vocab_size
        self.image_height, self.image_width = crop_size
        self.length = num_samples
        self.seed = seed
        if stage == "encoder_alignment":
            self.input_text = "what is shown in image."
            self.label_text = "Image of a happy dog."
        elif stage == "decoder_alignment":
            self.input_text = "Generate image of dog."
            self.label_text = "Here is the image of dog"
        else:
            NotImplementedError(f"Mock data not implemented for stage {self.stage} :( ")
        self.special_tokens = [f"IMG_{i}" for i in range(8)]
        self.output_image_processor = VaeImageProcessor()

        current_file_dir = os.path.dirname(os.path.abspath(__file__))
        image_path = os.path.join(current_file_dir, "dog.png")
        self.image = Image.open(image_path).convert("RGB")
        self.image_tensor = self.output_image_processor.preprocess(
            image=self.image, height=224, width=224, resize_mode='crop'
        ).squeeze()

    def __len__(self) -> int:
        return self.length

    def tokenize_text(self, text: str) -> torch.Tensor:
        """Tokenize the input text using the provided tokenizer."""
        tokens = self.tokenizer.tokenizer(text, return_tensors="pt", add_special_tokens=False)
        return tokens["input_ids"].squeeze(0)  # Return as 1D tensor

    def find_pattern_indices(self, template, pattern, search_start_index=0, allow_first_token_mismatch=False):
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
        # Tokenize input text and label text
        input_tokens = self.tokenize_text(self.input_text)
        label_tokens = self.tokenize_text(self.label_text)

        if self.stage == "encoder_alignment":
            media = self.image_processor(self.image)['pixel_values'][0]
            combined_tokens = torch.cat([input_tokens, label_tokens])
            labels = torch.ones_like(combined_tokens) * self.ignore_placeholder

            # answer_start = len(input_tokens)
            answer_start = 0
            labels[answer_start:] = combined_tokens[answer_start:]

            tokens = torch.cat([torch.tensor([-200]), combined_tokens[:-1]])
            labels = torch.cat([torch.tensor([self.ignore_placeholder]), labels[1:]])
            loss_mask = (labels != self.ignore_placeholder).float()

            image_token_mask = tokens == -200

            return {
                "images": media,
                "tokens": tokens,
                "position_ids": torch.arange(len(tokens), dtype=torch.int64),
                "labels": labels,
                "loss_mask": loss_mask,
                "num_image_tiles": media.shape[0],
                "image_token_mask": image_token_mask,
                "input_text": None,
                "output_images": None,
            }
        elif self.stage == "decoder_alignment":
            # images = torch.zeros((3, self.image_height, self.image_width))
            special_token_ids = [
                self.tokenizer.tokenizer.convert_tokens_to_ids(token) for token in self.special_tokens
            ]
            label_tokens = torch.cat([label_tokens, torch.tensor(special_token_ids, dtype=torch.long)])

            combined_tokens = torch.cat([input_tokens, label_tokens])
            labels = torch.ones_like(combined_tokens) * self.ignore_placeholder
            answer_start = len(input_tokens)
            labels[answer_start:] = combined_tokens[answer_start:]

            tokens = combined_tokens[:-1]
            labels = labels[1:]
            loss_mask = (labels != self.ignore_placeholder).float()

            image_token_mask = tokens == -200

            return {
                "images": None,
                "tokens": tokens,
                "position_ids": torch.arange(len(tokens), dtype=torch.int64),
                "labels": labels,
                "loss_mask": loss_mask,
                "num_image_tiles": None,
                "image_token_mask": image_token_mask,
                "input_text": self.input_text,
                "output_images": self.image_tensor,
            }
        else:
            NotImplementedError()

    def _collate_fn(self, batch):
        """Default collation function for the dataloader with None handling."""
        collated_batch = {}

        # Iterate over the keys in the first batch item to handle all keys
        for key in batch[0]:
            values = [item[key] for item in batch]  # Collect all values for this key

            # Handle None values: Skip collating for None or provide a placeholder
            if any(v is None for v in values):
                collated_batch[key] = None  # Use None as a placeholder
            else:
                collated_batch[key] = default_collate(values)
        if collated_batch['images'] is not None:
            collated_batch['images'] = (
                collated_batch['images'].contiguous().view(-1, *collated_batch['images'].shape[2:])
            )
        if collated_batch['num_image_tiles'] is not None:
            collated_batch['num_image_tiles'] = collated_batch['num_image_tiles'].to(dtype=torch.int32)
        else:
            collated_batch["num_image_tiles"] = torch.empty(0, dtype=torch.int32)

        return collated_batch

    def collate_fn(self, batch):
        """Method to use as the `collate_fn` in DataLoader."""
        return self._collate_fn(batch)


if __name__ == "__main__":
    model_id = 'llava-hf/llava-v1.6-vicuna-7b-hf'
    stage = 'encoder_alignment'
    tokenizer = AutoTokenizer(model_id)
    processor = AutoProcessor.from_pretrained(model_id)
    image_processor = processor.image_processor
    special_tokens = [f"IMG_{i}" for i in range(8)]
    tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
    data_module = MockDataModule(
        tokenizer=tokenizer,
        image_processor=image_processor,
        stage=stage,
        vocab_size=tokenizer.vocab_size,
        micro_batch_size=1,
    )
    data_module.setup()
    dataloader = data_module.test_dataloader()
    batch = next(iter(dataloader))

    print("Sample Batch:")

    if batch['images'] is not None:
        print("Images:", batch["images"].shape)
    if batch['output_images'] is not None:
        print("Output Images:", batch["output_images"].shape)
    print("Tokens:", batch["tokens"])
    print("Position IDs:", batch["position_ids"])
    print("Labels:", batch["labels"])
    print("Loss Mask:", batch["loss_mask"])
    if batch['input_text'] is not None:
        print("Input text:", batch["input_text"])
