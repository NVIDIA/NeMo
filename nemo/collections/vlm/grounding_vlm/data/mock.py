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

from typing import Dict, List, Optional, Any

import lightning.pytorch as pl
import numpy as np
import torch
from lightning.pytorch.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from PIL import Image
from torch.utils.data import DataLoader, Dataset

# from nemo.collections.vlm.qwen2vl.data.multimodal_tokens import IMAGE_TOKEN_INDEX
from nemo.lightning.pytorch.plugins import MegatronDataSampler
from nemo.collections.vlm.grounding_vlm.model.tokens import generate_extra_grounding_tokens
from nemo.collections.vlm.grounding_vlm.data.cococlasses import classes as coco_classes

class ClassificationDetectionMockDataModule(pl.LightningDataModule):
    """
    A mock data module for Qwen2VL training, validation, and testing.
    """

    def __init__(
        self,
        seq_length: int = 2048,
        tokenizer: Optional = None,
        image_processor: Optional = None,
        num_thinking_tokens: int = 128,
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
        self.num_thinking_tokens = num_thinking_tokens

        assert tokenizer is not None and image_processor is not None, 'please assign tokenizer and image_processor'

        # get the extra tokens from the tokenizer
        tokenizer, extra_tokens, extra_tokens_ids, metadata = generate_extra_grounding_tokens(tokenizer)
        extra_token_id_mapping = {k: v for k, v in zip(extra_tokens, extra_tokens_ids)}
        self.extra_tokens_ids = extra_tokens_ids
        self.extra_tokens = extra_tokens
        self.extra_tokens_metadata = metadata
        self.extra_token_id_mapping = extra_token_id_mapping
        
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
            num_thinking_tokens=self.num_thinking_tokens,
            extra_tokens_ids=self.extra_tokens_ids, 
            extra_tokens=self.extra_tokens, 
            extra_tokens_metadata=self.extra_tokens_metadata, 
            extra_token_id_mapping=self.extra_token_id_mapping
        )
        self._validation_ds = _Qwen2VLMockDataset(
            self.tokenizer, self.image_processor, "valid", self.num_val_samples, self.seq_length, 
            num_thinking_tokens=self.num_thinking_tokens,
            extra_tokens_ids=self.extra_tokens_ids, 
            extra_tokens=self.extra_tokens, 
            extra_tokens_metadata=self.extra_tokens_metadata, 
            extra_token_id_mapping=self.extra_token_id_mapping
        )
        self._test_ds = _Qwen2VLMockDataset(
            self.tokenizer, self.image_processor, "test", self.num_test_samples, self.seq_length, 
            num_thinking_tokens=self.num_thinking_tokens,
            extra_tokens_ids=self.extra_tokens_ids, 
            extra_tokens=self.extra_tokens, 
            extra_tokens_metadata=self.extra_tokens_metadata, 
            extra_token_id_mapping=self.extra_token_id_mapping
        )

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        # pylint: disable=C0115,C0116
        if not hasattr(self, "_train_ds"):
            self.setup()
        return self._create_dataloader(self._train_ds, batch_size=self.micro_batch_size)

    def val_dataloader(self) -> EVAL_DATALOADERS:
        # pylint: disable=C0115,C0116
        if not hasattr(self, "_validation_ds"):
            self.setup()
        return self._create_dataloader(self._validation_ds, batch_size=self.micro_batch_size)

    def test_dataloader(self) -> EVAL_DATALOADERS:
        # pylint: disable=C0115,C0116
        if not hasattr(self, "_test_ds"):
            self.setup()
        return self._create_dataloader(self._test_ds, batch_size=self.micro_batch_size)

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
        num_thinking_tokens: int,
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
        self.num_thinking_tokens = num_thinking_tokens

        self.image_processor = image_processor
        self.tokenizer = tokenizer
        self.image_width, self.image_height = 448, 448

        self.length = num_samples
        self.seed = seed

        self.loss_mask = torch.ones(self.seq_length, dtype=torch.float)
        self.position_ids = torch.arange(self.seq_length, dtype=torch.int64)
        # self.spatial_merge_size = 2

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        # get random seed
        rng = np.random.default_rng(seed=(self.seed + idx))
        # 1) Process image input
        image_inputs = prepare_image_inputs(3, self.image_height, self.image_width)
        print(f"image_inputs: {image_inputs}")

        # Generate mock boxes and labels
        num_boxes = rng.integers(0, 20) if idx % 3 != 2 else 0 # 1/3 of the time, no boxes
        boxes = rng.random((num_boxes, 4)) * 1000 if num_boxes > 0 else None
        # process them? 

        num_classes = rng.integers(0, 20) if idx % 7 != 4 else 0 # 1/7 of the time, no classes
        classes = list(rng.choice(coco_classes, size=num_classes, replace=False))  # sequence of characters [C',]
        class_label_mask = torch.rand(num_classes) > 0.5 
        class_ids = None
        class_attn_mask = None
        if num_classes > 0:
            classes_tokenized = self.tokenizer(classes, return_tensors='pt', padding=True)
            class_ids = classes_tokenized.input_ids  # [n_classes, seqlen]
            class_attn_mask = classes_tokenized.attention_mask  # [n_classes, seqlen]
        else:
            class_ids = torch.empty(0, 0, dtype=torch.int)  # [0, 0]
            class_attn_mask = torch.empty(0, 0, dtype=torch.int)  # [0, 0]

        prompt = f"Please detect the objects in the image. Also return the correct class(es) from the following list: {', '.join(classes)}"
        # apply chat template to prompt + image
        chat_template_prompt = [
            {
                "role": "user", 
                "content": [
                    {"type": "image", "image": image_inputs},
                    {"type": "text", "text": prompt},
                ]
            }
        ]
        chat_template_prompt = self.image_processor.apply_chat_template(chat_template_prompt, tokenize=False, add_generation_prompt=True)
        print(f"chat_template_prompt: {chat_template_prompt}")
        inputs = self.image_processor(
            text=chat_template_prompt,
            images=image_inputs,
            return_tensors="pt",
        )
        # get the input ids
        input_ids = inputs.input_ids[0].tolist()
        pixel_values = inputs.pixel_values
        image_grid_thw = inputs.image_grid_thw.reshape(-1, 3)

        # generate answers 
        answer = ""
        # generate the answer
        for think_token_idx in range(self.num_thinking_tokens):
            think_token_modidx = think_token_idx % self.extra_tokens_metadata['num_image_think_tokens']
            answer += f"<|img_think_{think_token_modidx}|>"
        answer += "<|img_think_end|>"

        # add number of boxes
        if num_boxes > 0:
            answer += "Here are the boxes:"
            for box_idx in range(num_boxes):
                box_idx_mod = box_idx % self.extra_tokens_metadata['num_bbox_tokens']
                answer += f"<|bbox_{box_idx_mod}|>"
        # add number of classes
        if class_label_mask.sum() > 0:
            answer += "Here are the classes:"
            for class_idx in range(num_classes):
                if class_label_mask[class_idx]:
                    answer += classes[class_idx]  + ", "
            answer = answer[:-2] # delete the last comma

        # tokenize the answer
        print(f"answer: {answer}")
        answer_tokenized = self.tokenizer(answer)
        answer_input_ids = list(answer_tokenized.input_ids)

        # make sure seq len is not exceeded
        final_tokens = input_ids + answer_input_ids + [self.tokenizer.pad_token_id]  # end with pad token
        valid_len = len(final_tokens)
        assert valid_len <= self.seq_length, "Sequence length exceeded"
        final_tokens = final_tokens + ([self.tokenizer.pad_token_id] * (self.seq_length - len(final_tokens))) # pad the rest
        attention_mask = torch.zeros(self.seq_length, dtype=torch.int)
        attention_mask[:valid_len] = 1

        print(f"final token length sequence: {valid_len}")

        return {
            "input_ids": torch.tensor(final_tokens),   # len tokenizer
            "position_ids": self.position_ids,
            "loss_mask": attention_mask,
            "attention_mask": attention_mask,
            "labels": torch.tensor(final_tokens[1:] + [self.tokenizer.pad_token_id]),
            # image
            "pixel_values": torch.tensor(pixel_values),  # [N, H]
            "image_grid_thw": torch.tensor(image_grid_thw),  # [N, 3]
            # detection
            "instance_det_ids": torch.tensor(boxes) if boxes is not None else None,  # [n_boxes, 4]
            # instance_cu_seqlen will be created in the collate fn
            # classification
            "cls_token_ids": torch.tensor(class_ids),  # [n_classes, seqlen]
            "cls_labels": torch.tensor(class_label_mask),  # [n_classes]
            "cls_attention_mask": torch.tensor(class_attn_mask), # [n_classes, seqlen]
            # class loss mask will be created in the collate fn
        }

    def _collate_fn(self, batch):
        """Collate function that handles variable number of boxes and classes per batch."""
        # Standard collation for fixed-size tensors
        input_ids = torch.stack([item["input_ids"] for item in batch])
        position_ids = torch.stack([item["position_ids"] for item in batch])
        loss_mask = torch.stack([item["loss_mask"] for item in batch])
        attention_mask = torch.stack([item["attention_mask"] for item in batch])
        labels = torch.stack([item["labels"] for item in batch])
        
        # Concat image tensors along batch dimension
        pixel_values = torch.cat([item["pixel_values"] for item in batch], dim=0)
        image_grid_thw = torch.cat([item["image_grid_thw"] for item in batch], dim=0)
        
        # Handle classification tokens and labels
        max_classes = max([0 if item["cls_token_ids"] is None else item["cls_token_ids"].shape[0] for item in batch])
        max_seq_len = max([0 if item["cls_token_ids"] is None else item["cls_token_ids"].shape[1] for item in batch])
        
        # Initialize classification tensors
        batch_size = len(batch)
        cls_token_ids = torch.full((batch_size, max_classes, max_seq_len), self.tokenizer.pad_token_id, dtype=torch.long)
        cls_attention_mask = torch.zeros((batch_size, max_classes, max_seq_len), dtype=torch.int)
        cls_labels = torch.zeros((batch_size, max_classes), dtype=torch.bool)
        cls_loss_mask = torch.zeros((batch_size, max_classes), dtype=torch.float)
        
        # Fill classification tensors
        for i, item in enumerate(batch):
            if item["cls_token_ids"] is not None and max_classes > 0:
                n_classes = item["cls_token_ids"].shape[0]
                seq_len = item["cls_token_ids"].shape[1]
                cls_token_ids[i, :n_classes, :seq_len] = item["cls_token_ids"]
                cls_attention_mask[i, :n_classes, :seq_len] = item["cls_attention_mask"]
                cls_labels[i, :n_classes] = item["cls_labels"]
                cls_loss_mask[i, :n_classes] = 1.0  # Mark valid entries
        
        # Handle instance detection boxes
        instance_boxes = [item["instance_det_ids"] for item in batch]
        box_counts = [(boxes.shape[0] if boxes is not None else 0) for boxes in instance_boxes]
        instance_cu_seqlen = (torch.tensor([0] + box_counts).cumsum(0)[1:]).contiguous()
        total_box_count = instance_cu_seqlen[-1]
        
        return {
            "input_ids": input_ids,
            "position_ids": position_ids,
            "loss_mask": loss_mask,
            "attention_mask": attention_mask,
            "labels": labels,
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
            "cls_token_ids": cls_token_ids,
            "cls_labels": cls_labels,
            "cls_attention_mask": cls_attention_mask,
            "cls_loss_mask": cls_loss_mask,
            "instance_det_ids": torch.cat([boxes for boxes in instance_boxes if boxes is not None], dim=0) if total_box_count > 0 else torch.empty(0, 4),
            "instance_cu_seqlen": instance_cu_seqlen,
        }

    def collate_fn(self, batch):
        return self._collate_fn(batch)


if __name__ == "__main__":

    from transformers import AutoTokenizer, AutoProcessor
    model_name = "Qwen/Qwen2.5-VL-7B-Instruct"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    image_processor = AutoProcessor.from_pretrained(model_name)

    print(f"Loading data module...")

    data_module = ClassificationDetectionMockDataModule(
        tokenizer=tokenizer,
        image_processor=image_processor,
        num_train_samples=1000,
        num_val_samples=1000,
        num_test_samples=1000,
        micro_batch_size=4,
        global_batch_size=4,
        num_workers=0,
    )
    data_module.setup()
    train_dataloader = data_module.train_dataloader()
    for batch in train_dataloader:
        print(batch)
        for k, v in batch.items():
            print(k, v.shape)
        input("Press Enter to continue...")