import dataclasses
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Union
import random
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
    WorkerConfig,
)
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoProcessor

from nemo.collections.multimodal.data.energon.config import ImageTextRawBatch, ImageTextSample, MultiModalSampleConfig
from nemo.collections.multimodal.data.energon.sample_encoder import SampleEncoder, VQASampleEncoder
from nemo.collections.multimodal.data.energon.task_encoder import MultiModalTaskEncoder
from nemo.utils import logging
from nemo.collections.multimodal.mimo.data.templates import (
    COMPHREHENSION_PROMPTS,
    GENERATION_PROMPTS,
    IMAGE_KEYWORDS,
    GENERATION_KEYWORDS,
    RESPONSES,
)
from nemo.utils import logging
from diffusers.image_processor import VaeImageProcessor

import re


def _fix_sentence(sentence):
    """
    Fixes a sentence by ensuring:
    - Removes any spaces before the final period, if present.
    - Adds a period at the end if missing.
    """
    # Strip any trailing whitespace
    sentence = sentence.rstrip()

    # Remove space before a period at the end, if present
    if sentence.endswith(' .'):
        sentence = sentence[:-2] + '.'  # Remove the space before the period
    elif not sentence.endswith('.'):
        sentence = sentence + '.'  # Add a period if none exists

    return sentence


def _find_pattern_indices(template, pattern, search_start_index=0, allow_first_token_mismatch=False):
    template_len = len(template)
    pattern_len = len(pattern)
    for i in range(search_start_index, template_len - pattern_len + 1):
        match = template[i : i + pattern_len] == pattern
        if torch.all(match) or (allow_first_token_mismatch and torch.all(match[1:])):
            return i, i + pattern_len
    return -1, -1


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
    def __init__(
        self, tokenizer, image_processor, multimodal_sample_config=MultiModalSampleConfig(), is_generation=False
    ):

        super().__init__(tokenizer, image_processor, multimodal_sample_config)
        self.special_tokens = [f"IMG_{i}" for i in range(8)]
        self.is_generation = is_generation
        if is_generation:
            self.output_image_processor = VaeImageProcessor()
        else:
            self.output_image_processor = None

        if self.conversation_template_config.chat_template:
            self.tokenizer.chat_template = self.conversation_template_config.chat_template
        elif self.tokenizer.chat_template is None:
            raise ValueError(
                "Both tokenizer and conversation template does not have chat template defined. Refer to https://huggingface.co/docs/transformers/main/en/chat_templating"
            )

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

    def compute_input_ids_labels(self, input_text, output_text):
        pass
        messages = []
        if self.conversation_template_config.system:
            messages.append({'role': 'system', 'content': self.conversation_template_config.system})
        messages.append({'role': self.conversation_template_config.roles[0], 'content': input_text})
        messages.append({'role': self.conversation_template_config.roles[1], 'content': output_text})

        templated_prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        logging.debug(f"apply prompt template templated_prompt {templated_prompt}")

        input_ids = self.tokenizer(templated_prompt, add_special_tokens=False).input_ids
        input_ids = torch.tensor(input_ids, dtype=torch.long)

        labels = torch.ones_like(input_ids) * self.ignore_place_holder
        stop_str = getattr(self.conversation_template_config, "stop_string", None)
        answer_tokens = self.tokenizer(
            output_text + ("" if stop_str is None else stop_str), add_special_tokens=False
        ).input_ids
        answer_tokens = torch.tensor(answer_tokens, dtype=torch.long)

        answer_start, answer_end = _find_pattern_indices(input_ids, answer_tokens, 0)
        assert answer_start > -1, f"answer_start not found, index {answer_start}"
        assert answer_end > -1, f"answer_end not found, index {answer_end}"
        labels[answer_start:answer_end] = input_ids[answer_start:answer_end]

        return input_ids, labels

    def encode_comphrehension(self, input_sample: CaptioningSample, output_sample: MimoCaptioningSample):

        input_text = random.choice(COMPHREHENSION_PROMPTS).format('image')
        output_text = _fix_sentence(input_sample.caption)

        input_ids, labels = self.compute_input_ids_labels(input_text=input_text, output_text=output_text)

        tokens = torch.cat([torch.tensor([self.multimodal_sample_config.image_token.token_id]), input_ids[:-1]])
        labels = torch.cat([torch.tensor([self.multimodal_sample_config.ignore_place_holder]), labels[1:]])

        logging.debug(f"sample encoder encode_comphrehension after tokenize prompt tokens {tokens}")
        logging.debug(f"sample encoder encodeencode_comphrehension lables {labels}")

        loss_mask = self.compute_loss_mask(labels)

        processed_image = self.process_image(input_sample.image)
        processed_image = processed_image.squeeze()

        output_sample.__key__ = input_sample.__key__
        output_sample.__restore_key__ = input_sample.__restore_key__
        output_sample.input_image = processed_image
        output_sample.tokens = tokens
        output_sample.labels = labels
        output_sample.loss_mask = loss_mask
        output_sample.caption = None
        output_sample.output_image = None

        return output_sample

    def encode_generation(self, input_sample: CaptioningSample, output_sample: MimoCaptioningSample):

        image_caption = _fix_sentence(input_sample.caption)
        input_text = (
            random.choice(GENERATION_PROMPTS).format(random.choice(GENERATION_KEYWORDS), random.choice(IMAGE_KEYWORDS))
            + image_caption
        )

        output_text = random.choice(RESPONSES).format('image')

        input_ids, labels = self.compute_input_ids_labels(input_text=input_text, output_text=output_text)

        special_token_ids = [self.tokenizer.convert_tokens_to_ids(token) for token in self.special_tokens]
        special_token_ids = torch.tensor(special_token_ids, dtype=torch.long)

        input_ids = torch.cat([input_ids, special_token_ids])
        labels = torch.cat([labels, special_token_ids])

        # Add extra stop string after special tokens to stop generation
        stop_str = getattr(self.conversation_template_config, "stop_string", None)
        if stop_str:
            stop_string_token_ids = torch.tensor(self.tokenizer.convert_tokens_to_ids(stop_str), dtype=torch.long)
            if stop_string_token_ids.dim() == 0:
                stop_string_token_ids = stop_string_token_ids.unsqueeze(0)
            # Append stop string token IDs to input_ids and labels
            input_ids = torch.cat([input_ids, stop_string_token_ids])
            labels = torch.cat([labels, stop_string_token_ids])
        # tokens and labels are not shifted by 1 yet
        # TODO: Yash we dont have to append a image_token_id in here, check if llava forward pass can run without an image
        tokens = torch.cat([torch.tensor([self.multimodal_sample_config.image_token.token_id]), input_ids[:-1]])
        labels = torch.cat([torch.tensor([self.multimodal_sample_config.ignore_place_holder]), labels[1:]])

        logging.debug(f"sample encoder encode_comphrehension after tokenize prompt tokens {tokens}")
        logging.debug(f"sample encoder encodeencode_comphrehension lables {labels}")

        loss_mask = self.compute_loss_mask(labels)
        output_image = self.output_image_processor.preprocess(
            image=input_sample.image, height=224, width=224, resize_mode='crop'
        ).squeeze()

        output_sample.__key__ = input_sample.__key__
        output_sample.__restore_key__ = input_sample.__restore_key__
        output_sample.input_image = torch.zeros((3, 336, 336))
        output_sample.tokens = tokens
        output_sample.labels = labels
        output_sample.loss_mask = loss_mask
        output_sample.caption = input_sample.caption
        output_sample.output_image = output_image

        return output_sample

    def encode(self, input_sample: CaptioningSample, output_sample: MimoCaptioningSample):

        if self.is_generation:
            output_sample = self.encode_generation(input_sample, output_sample)
        else:
            output_sample = self.encode_comphrehension(input_sample, output_sample)

        return output_sample


class MimoCaptioningTaskEncoder(MultiModalTaskEncoder):
    def __init__(self, tokenizer, image_processor, multimodal_sample_config, is_generation=False):
        super().__init__(tokenizer, image_processor, multimodal_sample_config)

        self.encoders: Dict[str, SampleEncoder] = {
            CaptioningSample.__name__: MimoCaptionSampleEncoder(
                tokenizer, image_processor, multimodal_sample_config, is_generation
            )
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
        batch_output_images = batch_pad_stack(output_images) if all(img is not None for img in output_images) else None
        batch_captions = batch_list(captions) if captions is all(cap is not None for cap in captions) else None

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
    multimodal_sample_config = MultiModalSampleConfig()
    multimodal_sample_config.conversation_template_config.system = None

    worker_config = WorkerConfig.default_worker_config(0)
    train_loader = get_loader(
        get_train_dataset(
            '/home/ykarnati/Downloads/datasets/cc3m',
            batch_size=128,
            shuffle_buffer_size=100,
            max_samples_per_sequence=100,
            task_encoder=MimoCaptioningTaskEncoder(
                tokenizer=tokenizer,
                image_processor=processor.image_processor,
                multimodal_sample_config=MultiModalSampleConfig(),
                is_generation=False,
            ),
        ),
        worker_config=worker_config,
    )

    print(f"data loader length {len(train_loader)}")
    for index, each_batch in enumerate(train_loader):
        print(f"batch index {index} tokens shape {each_batch['tokens'].shape} ")
