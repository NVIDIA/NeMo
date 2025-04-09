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

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from megatron.energon import VQASample

from nemo.collections.vlm.data.task_encoder import DataBatch, DataSample
from nemo.collections.vlm.data.task_encoder import TaskEncoder as BaseTaskEncoder
from nemo.collections.vlm.data.task_encoder import TaskEncoderConfig as BaseTaskEncoderConfig


@dataclass
class TaskEncoderConfig(BaseTaskEncoderConfig):
    """Configuration for llama4 processing.

    This class consolidates all configuration needed for llama4 processing,
    including model paths, tokenization, image processing, and sequence packing parameters.

    """

    stop_string: Optional[str] = ""
    system_prompt: Optional[str] = None


class TaskEncoder(BaseTaskEncoder):
    """TaskEncoder for llama4 data processing.

    This class handles the processing of different types of llama4 samples,
    including Visual Question Answering (VQA), Captioning, and Interleaved samples.
    It provides functionality for encoding individual samples, batching them together,
    and handling packed sequences for efficient processing.

    The encoder supports:
    - VQA samples: Processing image-question pairs with corresponding answers
    - [In progress] Interleaved samples: Processing alternating image and text content
    - [In progress] Similarity interleaved samples: Processing image-text pairs for similarity tasks
    - [In progress] Packed sequences: Efficient processing of multiple samples in a single sequence

    Args:
        config (TaskEncoderConfig): Configuration object containing processing parameters

    Note:
        When using packed sequences, the micro batch size must be 1, and the global batch
        size and sequence length must be adjusted accordingly.
    """

    def __init__(self, config: TaskEncoderConfig):
        """Initialize the llama4 processor.

        Args:
            config (TaskEncoderConfig): Configuration for processing
        """
        self.config = config
        self.hf_processor = self.config.hf_processor
        self.tokenizer = self.config.tokenizer

        # Initialize encoders with the config
        self.encoders = {
            "VQASample": self.encode_vqa_sample,
            # "InterleavedSample": self.encode_interleaved_sample,
            # "SimilarityInterleavedSample": self.encode_similarity_interleaved_sample,
        }

    def encode_batch(self, batch_data: DataBatch) -> dict:
        """Encode a batched set of samples for model input.

        This method transforms the raw batched data into a format ready for model input, including
        generating position IDs and other necessary fields.

        Parameters:
            batch_data (DataBatch): The raw batch of data to be encoded.

        Returns:
            dict: A dictionary containing the encoded batch data, ready for model input.
        """
        batch_data = super().encode_batch(batch_data)
        batch_data["media"] = batch_data["media"].reshape(-1, *batch_data["media"].shape[2:])
        return batch_data

    def encode_vqa_sample(self, input_sample: VQASample) -> DataSample:
        """Encode a VQA sample into a DataSample format.

        Args:
            input_sample (VQASample): Input VQA sample containing image, context and answers

        Returns:
            DataSample: Encoded sample with processed image, tokens, labels and loss mask
        """
        input_sample.context = input_sample.context.replace("<image>", self.config.image_token)
        messages = []
        if self.config.system_prompt:
            messages.append({'role': 'system', 'content': self.config.system_prompt})

        if isinstance(input_sample.context, list) and isinstance(input_sample.answers, list):
            min_length = min(len(input_sample.context), len(input_sample.answers))
            for i in range(min_length):
                messages.append({'role': self.config.roles[0], 'content': input_sample.context[i]})
                messages.append({'role': self.config.roles[1], 'content': input_sample.answers[i]})
        else:
            messages.append({'role': self.config.roles[0], 'content': input_sample.context})
            messages.append({'role': self.config.roles[1], 'content': input_sample.answers})

        converted_messages = self.hf_processor.apply_chat_template(messages)
        outputs = self.hf_processor(images=input_sample.image, text=converted_messages, return_tensors="pt")
        answers = input_sample.answers if isinstance(input_sample.answers, list) else [input_sample.answers]

        # Get tokens and images from formatter output
        tokens = outputs["input_ids"][0]
        images = outputs["pixel_values"]

        # Compute labels
        labels = torch.ones_like(tokens) * self.config.ignore_place_holder

        search_start = 0
        for answer in answers:
            answer_tokens = self.tokenizer.tokenizer(answer + self.config.stop_string, add_special_tokens=False)
            answer_tokens = answer_tokens["input_ids"]
            # Find answer pattern in tokens
            for i in range(search_start, len(tokens) - len(answer_tokens) + 1):
                if torch.all(tokens[i : i + len(answer_tokens)] == torch.tensor(answer_tokens)):
                    labels[i : i + len(answer_tokens)] = tokens[i : i + len(answer_tokens)]
                    search_start = i + len(answer_tokens)
                    break

        # Prepare final tensors
        tokens = tokens[:-1].contiguous()
        labels = labels[1:].contiguous()

        seq_len = len(tokens)

        # Pad tokens
        if self.config.pad_to_multiple_of:
            seqlen_padded = (
                (seq_len + self.config.pad_to_multiple_of - 1)
                // self.config.pad_to_multiple_of
                * self.config.pad_to_multiple_of
            )
            pad_len = seqlen_padded - seq_len

            if pad_len > 0:
                tokens = F.pad(tokens, (0, pad_len), 'constant', 0)
                labels = F.pad(labels, (0, pad_len), 'constant', self.config.ignore_place_holder)

        # Compute loss mask
        loss_mask = torch.ones_like(labels, dtype=torch.float)
        loss_mask[labels == self.config.ignore_place_holder] = 0.0

        # Process images to match mock data format
        if images is not None:
            processed_images = []
            for img in images:
                processed_img = img.bfloat16()  # Convert to bfloat16 like in mock data
                processed_images.append(processed_img)
            processed_image = torch.stack(processed_images)
        else:
            processed_image = torch.empty(0)

        return DataSample(
            __key__=input_sample.__key__,
            __restore_key__=input_sample.__restore_key__,
            __subflavor__=input_sample.__subflavor__,
            __subflavors__=input_sample.__subflavors__,
            images=processed_image,
            tokens=tokens,
            labels=labels,
            loss_mask=loss_mask,
            seqlen=seq_len,
        )


if __name__ == '__main__':
    import argparse

    from megatron.energon import WorkerConfig, get_loader, get_train_dataset

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True, help='path to the dataset directory')
    args = parser.parse_args()

    model_id = 'meta-llama/Llama-4-Scout-17B-16E-Instruct'

    task_encoder = TaskEncoder(
        config=TaskEncoderConfig(
            hf_path=model_id,
        )
    )

    # Create worker config
    worker_config = WorkerConfig.default_worker_config(0)

    # Create data loader
    train_loader = get_loader(
        get_train_dataset(
            args.data_path,
            batch_size=4,
            shuffle_buffer_size=100,
            max_samples_per_sequence=100,
            task_encoder=task_encoder,
            worker_config=worker_config,
        ),
        worker_config=worker_config,
    )

    print(f"Data loader length: {len(train_loader)}")
    for index, each_batch in enumerate(train_loader):
        print("=" * 50)
        for key in each_batch:
            print(f"batch index {index} '{key}' shape "
                  f"{each_batch[key].shape if isinstance(each_batch[key], torch.Tensor) else each_batch[key]}")
        if index >= 2:
            break
