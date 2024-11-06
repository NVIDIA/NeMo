import dataclasses
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Union

import torch
from megatron.energon import (
    CaptioningSample,
    DefaultTaskEncoder,
    VQASample,
    batch_list,
    batch_pad_stack,
    batch_stack,
    get_loader,
    get_train_dataset,
)
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoProcessor

from nemo.collections.multimodal.data.energon.config import ImageTextRawBatch, ImageTextSample, MultiModalSampleConfig
from nemo.collections.multimodal.data.energon.sample_encoder import SampleEncoder, VQASampleEncoder
from nemo.collections.multimodal.data.energon.task_encoder import MultiModalTaskEncoder
from nemo.utils import logging


@dataclass
class MimoCaptioningSample:
    __key__: str = ''
    __restore_key__: Tuple[Union[str, int], ...] = field(default_factory=list)
    input_image: torch.Tensor = field(default_factory=lambda: torch.empty(0))
    tokens: torch.Tensor = field(default_factory=lambda: torch.empty(0, dtype=torch.long))
    labels: torch.Tensor = field(default_factory=lambda: torch.empty(0, dtype=torch.long))
    loss_mask: torch.Tensor = field(default_factory=lambda: torch.empty(0, dtype=torch.float))
    output_image: torch.Tensor = field(default_factory=lambda: torch.empty(0))


@dataclass
class MimoCaptioningRawBatch:
    __keys__: List[str] = field(default_factory=list)
    images: torch.Tensor = field(default_factory=lambda: torch.empty(0))
    output_images: torch.Tensor = field(default_factory=lambda: torch.empty(0))

    tokens: torch.Tensor = field(default_factory=lambda: torch.empty(0, dtype=torch.long))
    labels: torch.Tensor = field(default_factory=lambda: torch.empty(0, dtype=torch.long))
    loss_mask: torch.Tensor = field(default_factory=lambda: torch.empty(0, dtype=torch.float))

    captions: List[str] = field(default_factory=list)
    __subflavors__: List[str] = field(default_factory=list)
    __restore_key__: List[str] = field(default_factory=list)


@dataclass
class CaptioningBatch:
    __keys__: List[str]
    # (n, c, h, w)
    images: torch.Tensor
    # (n, c)
    text_tokens: torch.Tensor
    # (n, c, c)
    text_attn_mask: torch.Tensor


class MimoCaptionSampleEncoder(VQASampleEncoder):
    def __init__(self, tokenizer, image_processor, multimodal_sample_config=MultiModalSampleConfig()):

        super().__init__(tokenizer, image_processor, multimodal_sample_config)
        self.special_tokens = [f"IMG_{i}" for i in range(8)]

    def process_image(self, image):
        """
        Process and prepare an image sample for encoding.
        This method preprocesses the image using the HF image_processor, converting it to
        a tensor.
        Parameters:
        image: The input image to be processed.
        Returns:
        torch.Tensor: The processed image tensor.
        """
        image_array = self.image_processor.preprocess(image, return_tensors='pt', do_rescale=False)['pixel_values'][0]
        return image_array

    def tokenize_text(self, text: str) -> torch.Tensor:
        """Tokenize the input text using the provided tokenizer."""
        tokens = self.tokenizer(text, return_tensors="pt")
        return tokens["input_ids"].squeeze(0)

    def encode(self, input_sample: CaptioningSample, output_sample: MimoCaptioningSample):

        input_tokens = self.tokenize_text(input_sample.caption)
        # label_tokens = self.tokenize_text(self.label_text)

        special_token_ids = [self.tokenizer.convert_tokens_to_ids(token) for token in self.special_tokens]
        special_token_ids = torch.tensor(special_token_ids, dtype=torch.long)
        # label_tokens = torch.cat([label_tokens, torch.tensor(special_token_ids, dtype=torch.long)])

        combined_tokens = torch.cat([input_tokens, special_token_ids])
        labels = torch.ones_like(combined_tokens) * self.multimodal_sample_config.ignore_place_holder
        answer_start = len(input_tokens)
        labels[answer_start:] = combined_tokens[answer_start:]

        tokens = torch.cat([torch.tensor([self.multimodal_sample_config.image_token.token_id]), combined_tokens[:-1]])
        labels = torch.cat([torch.tensor([self.multimodal_sample_config.ignore_place_holder]), labels[1:]])
        loss_mask = (labels != self.multimodal_sample_config.ignore_place_holder).float()

        output_sample.__key__ = input_sample.__key__
        output_sample.__restore_key__ = input_sample.__restore_key__
        output_sample.input_image = torch.zeros((3, 336, 336), dtype=torch.float16)
        output_sample.tokens = tokens
        output_sample.labels = labels
        output_sample.loss_mask = loss_mask
        output_sample.caption = input_sample.caption
        output_sample.output_image = input_sample.image
        # import torchvision.transforms as transforms

        # output_path = "/workspaces/NeMo/nemo/collections/multimodal/mimo/data/debug_image.png"
        # to_pil = transforms.ToPILImage()
        # image = to_pil(input_sample.image)
        # image.save(output_path)
        # print(f"Image saved to {output_path}")
        # from PIL import Image

        return output_sample


class MimoCaptioningTaskEncoder(MultiModalTaskEncoder):
    def __init__(self, tokenizer, image_processor, multimodal_sample_config):
        super().__init__(tokenizer, image_processor, multimodal_sample_config)

        self.encoders: Dict[str, SampleEncoder] = {
            CaptioningSample.__name__: MimoCaptionSampleEncoder(tokenizer, image_processor, multimodal_sample_config)
        }

    def encode_sample(self, sample: CaptioningSample):
        sample_type = type(sample).__name__
        encoder = self.encoders.get(sample_type)
        if not encoder:
            raise NotImplementedError(f"No encoder implemented for sample type {sample_type}")
        encoded_sample = encoder.encode(input_sample=sample, output_sample=MimoCaptioningSample())
        return encoded_sample

    def batch(self, samples: List[MimoCaptioningSample]) -> MimoCaptioningRawBatch:
        keys, images, tokens, labels, loss_mask, output_images, captions = [], [], [], [], [], [], []
        for sample in samples:
            keys.append(sample.__key__)
            images.append(sample.input_image)
            tokens.append(sample.tokens)
            labels.append(sample.labels)
            loss_mask.append(sample.loss_mask)
            output_images.append(sample.output_image)
            captions.append(sample.caption)

        batch_keys = batch_list(keys)
        batch_images = batch_pad_stack(images)
        batch_prompt_tokens = batch_pad_stack(tokens)
        batch_labels = batch_pad_stack(labels)
        batch_loss_mask = batch_pad_stack(loss_mask)
        batch_output_images = batch_pad_stack(output_images)
        batch_captions = batch_list(captions)

        return MimoCaptioningRawBatch(
            __keys__=batch_keys,
            images=batch_images,
            output_images=batch_output_images,
            tokens=batch_prompt_tokens,
            labels=batch_labels,
            loss_mask=batch_loss_mask,
            captions=batch_captions,
        )

    def encode_batch(self, batch_data: MimoCaptioningRawBatch) -> dict:
        batch_dict = dataclasses.asdict(batch_data)
        micro_batch_size, seq_length = batch_dict['tokens'].size()
        position_ids = torch.arange(seq_length, dtype=torch.long)
        position_ids = position_ids.unsqueeze(0).repeat(micro_batch_size, 1)
        batch_dict['position_ids'] = position_ids
        batch_dict['attention_mask'] = None
        if 'captions' in batch_dict:
            batch_dict['input_text'] = batch_dict.pop('captions')

        return batch_dict


if __name__ == '__main__':
    processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")
    tokenizer = processor.tokenizer
    special_tokens = [f"IMG_{i}" for i in range(8)]
    tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
    train_loader = get_loader(
        get_train_dataset(
            '/home/ykarnati/Downloads/datasets/cc3m',
            batch_size=32,
            shuffle_buffer_size=100,
            max_samples_per_sequence=100,
            task_encoder=MimoCaptioningTaskEncoder(
                tokenizer=tokenizer,
                image_processor=processor.image_processor,
                multimodal_sample_config=MultiModalSampleConfig(),
            ),
        )
    )

    one_batch = next(iter(train_loader))
    print(one_batch)
    import torchvision.transforms as transforms
    from PIL import Image

    # Path where the image will be saved
    output_path = "/workspaces/NeMo/nemo/collections/multimodal/mimo/data/debug_image.png"

    # Assuming `one_batch['output_images']` is a batch of images in the form (B, C, H, W)
    first_image_tensor = one_batch['output_images'][0]  # Get the first image in the batch

    # Convert the tensor to a PIL image
    # Assuming the tensor values are in the range [0, 1]
    to_pil = transforms.ToPILImage()
    first_image = to_pil(first_image_tensor)

    # Save the image
    first_image.save(output_path)
    print(f"First image saved to {output_path}")
