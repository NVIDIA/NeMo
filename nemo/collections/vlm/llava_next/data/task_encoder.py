from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch
from megatron.energon import SimilarityInterleavedSample, VQASample, batch_list, batch_pad_stack
from torch.nn.utils.rnn import pad_sequence

from nemo.collections.multimodal.data.energon.config import ImageTextRawBatch, ImageTextSample, MultiModalSampleConfig
from nemo.collections.multimodal.data.energon.sample_encoder import SampleEncoder, VQASampleEncoder
from nemo.collections.multimodal.data.energon.task_encoder import MultiModalTaskEncoder
from nemo.collections.vlm.llava_next.data.interleaved_sample_encoder import LlavaNextSimilarityInterleavedSampleEncoder
from nemo.collections.vlm.llava_next.data.sample import LlavaNextTextRawBatch, LlavaNextTextSample
from nemo.collections.vlm.llava_next.data.vqa_sample_encoder import LlavaNextSampleEncoder
from nemo.utils import logging


class LlavaNextTaskEncoder(MultiModalTaskEncoder):
    """LlavaNextTaskEncoder"""

    def __init__(self, tokenizer, image_processor, multimodal_sample_config):
        """
        Initialize the LlavaNextTaskEncoder.

        This encoder extends MultiModalTaskEncoder to specifically handle LlavaNeXT,
        overriding  encoders for VQA sample type.

        Parameters:
        tokenizer (Tokenizer): The tokenizer for processing text data across sample types.
        image_processor (ImageProcessor): The image processor for preprocessing images.
        multimodal_sample_config (MultiModalSampleConfig): Configuration settings for multimodal samples.
        """
        super().__init__(tokenizer, image_processor, multimodal_sample_config)
        self.encoders: Dict[str, SampleEncoder] = {
            VQASample.__name__: LlavaNextSampleEncoder(tokenizer, image_processor, multimodal_sample_config),
            SimilarityInterleavedSample.__name__: LlavaNextSimilarityInterleavedSampleEncoder(
                tokenizer=tokenizer, image_processor=image_processor, multimodal_sample_config=multimodal_sample_config
            ),
        }

    def batch(self, samples: List[LlavaNextTextSample]) -> LlavaNextTextRawBatch:
        """
        Batch multiple encoded samples into a single batch structure for model input.

        This method combines individual sample fields (keys, images, tokens, labels, etc.) and
        pads or stacks them as needed to create a unified batch.

        Parameters:
        samples (List[LlavaNextTextSample]): A list of LlavaNextTextSample instances to be batched.

        Returns:
        LlavaNextTextRawBatch: A batch containing all input samples' images, tokens, labels,
            loss masks, and other metadata prepared for model processing.
        """
        keys, images, tokens, labels, loss_mask, num_media_tiles, image_sizes, attention_mask = (
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
        )
        for sample in samples:
            keys.append(sample.__key__)
            images.append(sample.images)
            tokens.append(sample.tokens)
            labels.append(sample.labels)
            loss_mask.append(sample.loss_mask)
            num_media_tiles.append(sample.num_media_tiles)
            image_sizes.append(sample.image_sizes)
            attention_mask.append(sample.attention_mask)

        batch_keys = batch_list(keys)

        batch_images = torch.cat(images, dim=0)

        batch_tokens = pad_sequence(tokens, batch_first=True)
        batch_labels = pad_sequence(labels, batch_first=True)
        image_sizes = torch.cat(image_sizes, dim=0)
        batch_loss_mask = batch_pad_stack(loss_mask)
        batch_attention_mask = batch_pad_stack(attention_mask)
        batch_list_num_media_tiles = batch_list(num_media_tiles)
        # if batch_list_num_media_tiles is nested lists, each sample has multiple images with different tiles
        # we need to flatten the list so len is num_images (in the batch)
        # image_sizes is also expected to be num_images, 2
        batch_list_num_media_tiles = flatten_if_nested(batch_list_num_media_tiles)
        batch_num_media_tiles = torch.tensor(batch_list_num_media_tiles, dtype=torch.int)

        assert (
            image_sizes.shape[0] == batch_num_media_tiles.shape[0]
        ), "image_sizes and batch_num_media_tiles must have the same length"

        return LlavaNextTextRawBatch(
            __keys__=batch_keys,
            images=batch_images,
            tokens=batch_tokens,
            labels=batch_labels,
            loss_mask=batch_loss_mask,
            num_media_tiles=batch_num_media_tiles,
            image_sizes=image_sizes,
            attention_mask=batch_attention_mask,
        )


from itertools import chain


def flatten_if_nested(lst):
    # Check if the first element is a list (assuming consistent structure)
    if any(isinstance(i, list) for i in lst):
        return list(chain.from_iterable(lst))
    return lst


if __name__ == '__main__':
    import argparse

    from megatron.energon import WorkerConfig, get_loader, get_train_dataset
    from transformers import AutoProcessor

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True, help='path to the dataset directory')
    args = parser.parse_args()
    processor = AutoProcessor.from_pretrained("llava-hf/llava-v1.6-vicuna-7b-hf")
    tokenizer = processor.tokenizer
    multimodal_sample_config = MultiModalSampleConfig()
    multimodal_sample_config.conversation_template_config.system = None

    worker_config = WorkerConfig.default_worker_config(0)
    multimodal_sample_config = MultiModalSampleConfig()
    multimodal_sample_config.image_following_text = False
    train_loader = get_loader(
        get_train_dataset(
            args.data_path,
            batch_size=4,
            shuffle_buffer_size=100,
            max_samples_per_sequence=100,
            task_encoder=LlavaNextTaskEncoder(
                tokenizer=tokenizer,
                image_processor=processor.image_processor,
                multimodal_sample_config=MultiModalSampleConfig(),
            ),
            worker_config=worker_config,
        ),
        worker_config=worker_config,
    )

    print(f"data loader length {len(train_loader)}")
    for index, each_batch in enumerate(train_loader):
        print(f"batch index {index} tokens shape {each_batch['tokens'].shape} ")
