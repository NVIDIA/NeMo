from nemo.collections.multimodal.data.energon.task_encoder import MultiModalTaskEncoder
from nemo.collections.multimodal.data.energon.config import ImageTextRawBatch
from nemo.collections.vlm.llama.data.sample_encoder import Llama3SampleEncoder, LlamaImageTextSample
import dataclasses
from dataclasses import dataclass, field
from typing import Dict, List
from torch.nn.utils.rnn import pad_sequence
import torch
from megatron.energon import (
    VQASample,
    batch_list,
    batch_pad_stack,
)
from nemo.collections.multimodal.data.energon.sample_encoder import SampleEncoder


@dataclass
class LlamaImageTextRawBatch(ImageTextRawBatch):
    vision_mask: torch.Tensor = field(default_factory=lambda: torch.empty(0))


class LlamaTaskEncoder(MultiModalTaskEncoder):
    def __init__(self, tokenizer, image_processor, multimodal_sample_config):
        super().__init__(tokenizer, image_processor, multimodal_sample_config)
        self.encoders: Dict[str, SampleEncoder] = {
            VQASample.__name__: Llama3SampleEncoder(tokenizer, image_processor, multimodal_sample_config)
        }

    def batch(self, samples: List[LlamaImageTextSample]) -> LlamaImageTextRawBatch:
        keys, images, tokens, labels, loss_mask, vision_mask = [], [], [], [], [], []
        for sample in samples:
            keys.append(sample.__key__)
            images.append(sample.images)
            tokens.append(sample.tokens)
            labels.append(sample.labels)
            loss_mask.append(sample.loss_mask)
            vision_mask.append(sample.vision_mask)

        batch_keys = batch_list(keys)
        batch_images = batch_pad_stack(images)

        batch_tokens = pad_sequence(tokens, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        batch_labels = pad_sequence(labels, batch_first=True, padding_value=self.tokenizer.pad_token_id)

        batch_loss_mask = batch_pad_stack(loss_mask)
        batch_vision_mask = batch_pad_stack(vision_mask)
        return LlamaImageTextRawBatch(
            __keys__=batch_keys,
            images=batch_images,
            tokens=batch_tokens,
            labels=batch_labels,
            loss_mask=batch_loss_mask,
            vision_mask=batch_vision_mask,
        )
