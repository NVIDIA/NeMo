"""
data_utils.py

General utilities and classes for facilitating data loading and collation.
"""

from dataclasses import dataclass
from typing import Callable, Dict, Sequence, Tuple

import torch
from megatron.core import parallel_state
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

        # For now, we only support Tokenizers with `padding_side = "right"` during training
        #   => Handle padding via RNN Utils => `pad_sequence`
        assert self.padding_side == "right", f"Invalid Tokenizer `{self.padding_side = }`"
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.pad_token_id)
        labels = pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)

        # DEBUGGING (COMPATIBLE WITH NEVA)
        # adding special image token (IMAGE_TOKEN_INDEX) to input_ids and labels
        # we count the number of images for each sample, to add corresponding
        # number of IMAGE_TOKEN_INDEX tokens to input_ids and IGNORE_INDEX tokens to labels
        # we add the images tokens after the <bos>
        num_images = 1 if isinstance(pixel_values[0], torch.Tensor) else len(pixel_values[0])
        image_tokens = torch.full((input_ids.shape[0], 1), IMAGE_TOKEN_INDEX, dtype=input_ids.dtype)
        ignored_tokens = torch.full((input_ids.shape[0], 1), IGNORE_INDEX, dtype=input_ids.dtype)
        input_ids = torch.cat(
            (input_ids[:, :1], image_tokens, input_ids[:, 1:]), dim=1
        )  # Concatenate <bos>, image_tokens, and the rest of the sequence along the last dimension
        labels = torch.cat(
            (labels[:, :1], ignored_tokens, labels[:, 1:]), dim=1
        )  # Concatenate <bos>, image_tokens, and the rest of the sequence along the last dimension

        # DEBUGGING (COMPATIBLE WITH NEVA)
        # labels are automatically shifted in HF model, but not in NeVa, so we manually shift it here
        input_ids = input_ids[..., :-1]
        labels = labels[..., 1:]

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
            concatenated_pixel_values = torch.cat(tuple([pixel_values[k] for k in pixel_values.keys()]), dim=1)
        else:
            concatenated_pixel_values = pixel_values

        # DEBUGGING (COMPATIBLE WITH NEVA)
        # explicitly calculate loss_mask
        loss_mask = labels > 0

        # DEBUGGING
        rank = torch.distributed.get_rank()
        data_rank = parallel_state.get_data_parallel_rank()
        debug_message = (f"==========================================\n"
                         f"[{rank}: Data Rank = {data_rank}] concatenated_pixel_values.shape: {concatenated_pixel_values.shape}\n"
                         f"[{rank}: {data_rank = }] input_ids.shape: {input_ids.shape}\n"
                         f"[{rank}: {data_rank = }] concatenated_pixel_values.stddev: {concatenated_pixel_values.std()}\n"
                         f"[{rank}: {data_rank = }] concatenated_pixel_values.mean: {concatenated_pixel_values.mean()}\n"
                         # f"[{rank}] input_ids: {input_ids}\n"
                         # f"[{rank}] labels: {labels}\n"
                         # f"[{rank}] loss_mask: {loss_mask}\n"
                         # f"[{rank}] attention_mask: {attention_mask}\n"
                         # f"[{rank}] position_ids: {position_ids}\n"
                         f"==========================================")
        print(debug_message)
        # torch.distributed.barrier()
        # print(stop_here)


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
