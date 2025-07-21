# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

from typing import Dict, List, Optional

import lightning.pytorch as pl
import numpy as np
import torch
from lightning.pytorch.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from PIL import Image
from torch.utils import data
from torch.utils.data import DataLoader, Dataset

from nemo.collections.vlm.qwen2vl.data.multimodal_tokens import IMAGE_TOKEN_INDEX
from nemo.lightning.pytorch.plugins import MegatronDataSampler
from nemo.collections.vlm.grounding_vlm.model.tokens import generate_extra_grounding_tokens

class ClassificationDetectionMockDataModule(pl.LightningDataModule):
    """
    A mock data module for Qwen2VL training, validation, and testing.
    """

    def __init__(
        self,
        seq_length: int = 2048,
        tokenizer: Optional = None,
        image_processor: Optional = None,
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
        self.seq_length = seq_length
        self.micro_batch_size = micro_batch_size
        self.global_batch_size = global_batch_size
        self.num_train_samples = num_train_samples
        self.num_val_samples = num_val_samples
        self.num_test_samples = num_test_samples
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers

        assert tokenizer is not None and image_processor is not None, 'please assign tokenizer and image_processor'

        # get the extra tokens from the tokenizer
        tokenizer, extra_tokens, extra_tokens_ids, metadata = generate_extra_grounding_tokens(tokenizer)
        extra_token_id_mapping = {k: v for k, v in zip(extra_tokens, extra_tokens_ids)}
        self.extra_tokens_ids = extra_tokens_ids
        self.extra_tokens = extra_tokens
        self.extra_tokens_metadata = metadata
        
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.data_sampler = MegatronDataSampler(
            seq_len=self.seq_length,
            micro_batch_size=micro_batch_size,
            global_batch_size=global_batch_size,
            rampup_batch_size=rampup_batch_size,
        )

    def setup(self, stage: str = "") -> None:
        # pylint: disable=C0115,C0116
        self._train_ds = _Qwen2VLMockDataset(
            self.tokenizer, self.image_processor, "train", self.num_train_samples, self.seq_length, 
            extra_tokens_ids=self.extra_tokens_ids, 
            extra_tokens=self.extra_tokens, 
            extra_tokens_metadata=self.extra_tokens_metadata, 
            extra_token_id_mapping=self.extra_token_id_mapping
        )
        self._validation_ds = _Qwen2VLMockDataset(
            self.tokenizer, self.image_processor, "valid", self.num_val_samples, self.seq_length, 
            extra_tokens_ids=self.extra_tokens_ids, 
            extra_tokens=self.extra_tokens, 
            extra_tokens_metadata=self.extra_tokens_metadata, 
            extra_token_id_mapping=self.extra_token_id_mapping
        )
        self._test_ds = _Qwen2VLMockDataset(
            self.tokenizer, self.image_processor, "test", self.num_test_samples, self.seq_length, 
            extra_tokens_ids=self.extra_tokens_ids, 
            extra_tokens=self.extra_tokens, 
            extra_tokens_metadata=self.extra_tokens_metadata, 
            extra_token_id_mapping=self.extra_token_id_mapping
        )

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        # pylint: disable=C0115,C0116
        if not hasattr(self, "_train_ds"):
            self.setup()
        return self._create_dataloader(self._train_ds)

    def val_dataloader(self) -> EVAL_DATALOADERS:
        # pylint: disable=C0115,C0116
        if not hasattr(self, "_validation_ds"):
            self.setup()
        return self._create_dataloader(self._validation_ds)

    def test_dataloader(self) -> EVAL_DATALOADERS:
        # pylint: disable=C0115,C0116
        if not hasattr(self, "_test_ds"):
            self.setup()
        return self._create_dataloader(self._test_ds)

    def _create_dataloader(self, dataset, **kwargs) -> DataLoader:
        # pylint: disable=C0115,C0116
        return DataLoader(
            dataset,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            collate_fn=dataset.collate_fn,
            **kwargs,
        )


def prepare_image_inputs(num_channels: np.uint8 = 3, width=1024, height=1024):
    """This function prepares a list of PIL images"""
    image_inputs = [np.random.randint(255, size=(num_channels, width, height), dtype=np.uint8)]
    image_inputs = [Image.fromarray(np.moveaxis(x, 0, -1)) for x in image_inputs]
    return image_inputs


class _Qwen2VLMockDataset(Dataset):
    def __init__(
        self,
        tokenizer,
        image_processor,
        name: str,
        num_samples: int,
        seq_length: int,
        seed: int = 42,
        extra_tokens_ids: Optional[List[int]] = None,
        extra_tokens: Optional[List[str]] = None,
        extra_tokens_metadata: Optional[List[Dict[str, Any]]] = None,
        extra_token_id_mapping: Optional[Dict[str, int]] = None,
    ) -> None:
        super().__init__()
        self.name = name
        self.seq_length = seq_length

        self.vocab_size = tokenizer.vocab_size
        # store extra token info here
        self.extra_tokens_ids = extra_tokens_ids
        self.extra_tokens = extra_tokens
        self.extra_tokens_metadata = extra_tokens_metadata
        self.extra_token_id_mapping = extra_token_id_mapping

        self.image_processor = image_processor
        self.image_width, self.image_height = np.random.choice(np.arange(56, 1024), 2)

        self.length = num_samples
        self.seed = seed

        self.loss_mask = torch.ones(self.seq_length, dtype=torch.float)
        self.position_ids = torch.arange(self.seq_length, dtype=torch.int64)
        self.spatial_merge_size = 2

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        # 1) Process image input
        image_inputs = prepare_image_inputs(3, self.image_height, self.image_width)
        process_out = self.image_processor(image_inputs, return_tensors="pt")
        pixel_values = process_out.pixel_values
        image_grid_thw = process_out.image_grid_thw[0]
        image_token_amount = image_grid_thw.prod() // (self.spatial_merge_size**2)

        # Generate mock boxes and labels
        boxes, labels = self._generate_mock_boxes()
        
        # Create prompt with detection information
        prompt = self._format_prompt(boxes, labels)
        
        # Tokenize the prompt
        if self.tokenizer:
            tokenized = self.tokenizer(prompt, return_tensors="pt", padding="max_length", 
                                     max_length=self.seq_length, truncation=True)
            input_ids = tokenized["input_ids"][0]
            labels = input_ids.clone()
        else:
            # Fallback to random tokens if no tokenizer
            np_gen = np.random.default_rng(seed=(self.seed + idx))
            input_ids = torch.from_numpy(np_gen.integers(self.vocab_size, size=[self.seq_length], dtype=np.int64))
            labels = input_ids.clone()

        # Insert image tokens after initial context
        img_start_idx = min(20, len(input_ids) // 4)  # Insert image after initial context
        input_ids[img_start_idx:img_start_idx + image_token_amount] = IMAGE_TOKEN_INDEX

        # Mask labels before the response
        generation_prompt_size = 5
        prompt_end_idx = img_start_idx + image_token_amount + generation_prompt_size
        labels[:prompt_end_idx] = -100

        return {
            "input_ids": input_ids,
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
            "loss_mask": self.loss_mask,
            "labels": labels,
            "boxes": boxes,
            "box_labels": labels,
        }

    def _collate_fn(self, batch):
        """Collate function that handles variable number of boxes per batch."""
        # Standard collation for fixed-size tensors
        input_ids = torch.stack([item["input_ids"] for item in batch])
        pixel_values = torch.stack([item["pixel_values"] for item in batch])
        image_grid_thw = torch.stack([item["image_grid_thw"] for item in batch])
        loss_mask = torch.stack([item["loss_mask"] for item in batch])
        labels = torch.stack([item["labels"] for item in batch])
        
        # Collate boxes and labels (these might have different sizes per batch)
        boxes = [item["boxes"] for item in batch]
        box_labels = [item["box_labels"] for item in batch]
        
        return {
            "input_ids": input_ids,
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
            "loss_mask": loss_mask,
            "labels": labels,
            "boxes": boxes,
            "box_labels": box_labels,
            "attention_mask": None,
        }

    def collate_fn(self, batch):
        return self._collate_fn(batch)
