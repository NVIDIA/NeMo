"""
data_utils.py

General utilities and classes for facilitating data loading and collation.
"""

from dataclasses import dataclass
from typing import Callable, Dict, Sequence, Tuple

import torch
from torch.nn.utils.rnn import pad_sequence
from nemo.collections.vlm.neva.data.multimodal_tokens import ImageToken

# HuggingFace Default / LLaMa-2 IGNORE_INDEX (for labels)
IGNORE_INDEX = -100

# NeMo special token (image) index
IMAGE_TOKEN_INDEX = ImageToken.token_index

def tree_map(fn: Callable, tree: dict) -> dict:
    """Maps a function over a nested dictionary."""
    return {k: tree_map(fn, v) if isinstance(v, dict) else fn(v) for k, v in tree.items()}


def tree_map_with_key(fn: Callable, tree: dict, keys: Sequence = ()) -> dict:
    """Maps a function over a nested dictionary."""
    return {
        k: tree_map_with_key(fn, v, (*keys, k)) if isinstance(v, dict) else fn((*keys, k), v) for k, v in tree.items()
    }


@dataclass
class PaddedCollatorForLanguageModeling:
    model_max_length: int
    pad_token_id: int
    default_image_resolution: Tuple[int, int, int]
    padding_side: str = "right"
    pixel_values_dtype: torch.dtype = torch.float32

    def __post_init__(self) -> None:
        self.dummy_pixel_values = torch.zeros(self.default_image_resolution, dtype=self.pixel_values_dtype)

    def __call__(self, instances: Sequence[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        pixel_values = [instance["pixel_values"] for instance in instances]

        # For now, we only support Tokenizers with `padding_side = "right"` during Training (but plan to extend!)
        #   => Handle padding via RNN Utils => `pad_sequence`
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.pad_token_id)
        labels = pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)

        # Truncate (if necessary)
        input_ids, labels = input_ids[:, : self.model_max_length], labels[:, : self.model_max_length]

        # Get `attention_mask` by checking for `pad_token_id`
        attention_mask = input_ids.ne(self.pad_token_id)

        # === Handle "unimodal" (language-only) vs. "multimodal" ===

        # Some examples are "language-only" --> build a Tensor of `multimodal_indices` that we can slice into easily
        multimodal_indices = torch.tensor(
            [idx for idx in range(len(pixel_values)) if pixel_values[idx] is not None], dtype=torch.long
        )

        # Stack all `pixel_values` --> depending on type (torch.Tensor, or Dict[str, torch.Tensor]) & presence of None
        if len(multimodal_indices) == 0:
            pixel_values = torch.stack([self.dummy_pixel_values for _ in range(len(input_ids))])
        elif isinstance(pv_example := pixel_values[multimodal_indices[0]], torch.Tensor):
            pixel_values = torch.stack(
                [
                    pixel_values[idx] if idx in multimodal_indices else self.dummy_pixel_values
                    for idx in range(len(input_ids))
                ]
            )
        elif isinstance(pv_example, dict):
            pixel_values = {
                k: torch.stack(
                    [
                        pixel_values[idx][k] if idx in multimodal_indices else self.dummy_pixel_values
                        for idx in range(len(input_ids))
                    ]
                )
                for k in pv_example
            }
        else:
            raise ValueError(f"Unsupported `pixel_values` type = {type(pixel_values)}")

        return dict(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            multimodal_indices=multimodal_indices,
        )


@dataclass
class PaddedCollatorForActionPrediction:
    model_max_length: int
    pad_token_id: int
    padding_side: str = "right"
    pixel_values_dtype: torch.dtype = torch.float32

    def __call__(self, instances: Sequence[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        pixel_values = [instance["pixel_values"] for instance in instances]
        if "dataset_name" in instances[0]:
            dataset_names = [instance["dataset_name"] for instance in instances]
        else:
            dataset_names = None

        # # DEBUGGING
        # # run with one image only
        # pixel_values = [item['dino'] for item in pixel_values]

        # For now, we only support Tokenizers with `padding_side = "right"` during training
        #   => Handle padding via RNN Utils => `pad_sequence`
        assert self.padding_side == "right", f"Invalid Tokenizer `{self.padding_side = }`"
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.pad_token_id)
        labels = pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)

        # DEBUGGING (COMPATIBLE WITH NEVA)
        # check if labels shift in Nemo's model similarly to HF or not
        input_ids = input_ids[..., :-1]
        labels = labels[..., 1:]

        # DEBUGGING (COMPATIBLE WITH NEVA)
        # adding special image token (IMAGE_TOKEN_INDEX) to input_ids and labels
        # we count the number of images for each sample, to add corresponding number of IMAGE_TOKEN_INDEX tokens
        num_images = 1 if isinstance(pixel_values[0], torch.Tensor) else len(pixel_values[0])
        image_tokens = torch.full((input_ids.shape[0], num_images), IMAGE_TOKEN_INDEX, dtype=input_ids.dtype)
        input_ids = torch.cat((image_tokens, input_ids), dim=1)
        labels = torch.cat((image_tokens, labels), dim=1)

        # Truncate (if necessary)
        input_ids, labels = input_ids[:, : self.model_max_length], labels[:, : self.model_max_length]

        # Get `attention_mask` by checking for `pad_token_id`
        attention_mask = input_ids.ne(self.pad_token_id)

        # DEBUGGING (COMPATIBLE WITH NEVA)
        # NeVa pad input_ids with 0s and pad labels with IGNORE_INDEXs
        input_ids[input_ids == self.pad_token_id] = 0


        # DEBUGGING (COMPATIBLE WITH NEVA)
        # compute position_ids
        position_ids = torch.arange(input_ids.shape[1], dtype=torch.long)
        position_ids = position_ids.unsqueeze(0).expand(input_ids.shape[0], -1).contiguous()

        # [Contract] For VLA Training =>> No "Unimodal" Data!
        assert all([pv is not None for pv in pixel_values]), "Invalid VLA Example with `pixel_values = None`!"

        # Stack all `pixel_values` --> depending on type is torch.Tensor or Dict[str, torch.Tensor]
        if isinstance(pixel_values[0], torch.Tensor):
            pixel_values = torch.stack(pixel_values)
        elif isinstance(pixel_values[0], dict):
            pixel_values = {
                k: torch.stack([pixel_values[idx][k] for idx in range(len(input_ids))]) for k in pixel_values[0]
            }
        else:
            raise ValueError(f"Unsupported `pixel_values` type = {type(pixel_values)}")

        # DEBUGGING (COMPATIBLE WITH NEVA)
        # if there are 2 transformered images, concatenate 2 images into one tensor on the channel dimension
        if num_images > 1:
            # DEBUGGING
            concatenated_pixel_values = torch.cat(tuple([pixel_values[k] for k in pixel_values.keys()]), dim=1)
        else:
            concatenated_pixel_values = pixel_values

        # DEBUGGING (COMPATIBLE WITH NEVA)
        # explicitly calculate loss_mask
        # wrong!, mask only False for action tokens
        # loss_mask = (labels==IGNORE_INDEX) | (labels==IMAGE_TOKEN_INDEX)
        loss_mask = labels < 0

        # DEBUGGING
        rank = torch.distributed.get_rank()
        debug_message = (f"==========================================\n"
                         f"[{rank}] concatenated_pixel_values.shape: {concatenated_pixel_values.shape}\n"
                         f"[{rank}] input_ids.shape: {input_ids.shape}\n"
                         f"[{rank}] labels.shape: {labels.shape}\n"
                         f"[{rank}] loss_mask.shape: {loss_mask.shape}\n"
                         f"[{rank}] attention_mask.shape: {attention_mask.shape}\n"
                         f"[{rank}] position_ids.shape: {position_ids.shape}\n"
                         f"[{rank}] concatenated_pixel_values.mean(): {concatenated_pixel_values.mean()}\n"
                         f"[{rank}] concatenated_pixel_values.std(): {concatenated_pixel_values.std()}\n"
                         f"[{rank}] input_ids: {input_ids}\n"
                         f"[{rank}] labels: {labels}\n"
                         f"[{rank}] loss_mask: {loss_mask}\n"
                         f"[{rank}] attention_mask: {attention_mask}\n"
                         f"[{rank}] position_ids: {position_ids}\n"
                         f"==========================================")
        print(debug_message)
        torch.distributed.barrier()
        print(stop_here)


        output = dict(
            media=concatenated_pixel_values,
            tokens=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            loss_mask=loss_mask,
            position_ids=position_ids,
        )

        if dataset_names is not None:
            output["dataset_names"] = dataset_names
        return output