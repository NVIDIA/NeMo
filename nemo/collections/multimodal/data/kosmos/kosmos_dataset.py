# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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
import io
import json
import re
from functools import partial
from typing import Any, Dict, List, Optional, Union

import torch
from einops import rearrange
from PIL import Image
from torch.utils.data import Dataset, default_collate

from nemo.collections.multimodal.data.clip.augmentations.augmentations import image_transform
from nemo.collections.multimodal.data.clip.imagenet_zeroshot_data import imagenet_classnames, openai_imagenet_template
from nemo.collections.multimodal.data.common.webdataset import WebDatasetCommon
from nemo.collections.nlp.data.language_modeling.megatron.data_samplers import (
    MegatronPretrainingRandomSampler,
    MegatronPretrainingSampler,
)
from nemo.collections.nlp.data.language_modeling.megatron.gpt_dataset import _create_ltor_masks_and_position_ids
from nemo.collections.vision.data.megatron.image_folder import ImageFolder
from nemo.collections.vision.data.megatron.vit_dataset import RandomSeedDataset

try:
    from megatron.core import parallel_state

    HAVE_MEGATRON_CORE = True

except (ImportError, ModuleNotFoundError):

    HAVE_MEGATRON_CORE = False

MIN_KB = 10
MAX_NUM_IMAGES = 6
Image.MAX_IMAGE_PIXELS = 933120000
_IMG_EXTENSIONS = "jpg jpeg png ppm pgm pbm pnm".split()
_DATASET_TYPES = ["image_caption", "image_interleaved"]


def pil_loader(key, data):
    r"""
    Function to load an image.
    If the image is corrupt, it returns a black image.
    Args:
        key: Image key.
        data: Image data stream.
    """
    extension = re.sub(r".*[.]", "", key)
    if extension.lower() not in _IMG_EXTENSIONS:
        return None
    if len(data) // 1000 <= MIN_KB:
        return None

    with io.BytesIO(data) as stream:
        img = Image.open(stream)
        img.load()
        img = img.convert("RGB")

    return img


def tokenize_and_insert_media_tokens(
    texts: Union[str, List[str]],
    tokenizer: Any,
    context_length: int,
    num_media_tokens: int,
    add_extra_token: int,
    media_start_id: str,
    media_end_id: str,
) -> torch.LongTensor:
    """
    Returns the tokenized representation of given input string(s) with media tokens inserted.

    Parameters
    ----------
    texts : Union[str, List[str]]
        An input string or a list of input strings to tokenize.
    tokenizer : Any
        A tokenizer to be used for tokenization.
    context_length : int
        The context length to be used for the output tensor.
    num_media_tokens : int
        The number of media latents to insert between media tokens.

    Returns
    -------
    torch.LongTensor
        A two-dimensional tensor containing the resulting tokens, shape = [number of input strings, context_length].
    """
    assert add_extra_token == 0 or add_extra_token == 1, "`add_extra_token` should be either 0 or 1."

    texts_is_str = False
    if isinstance(texts, str):
        texts = [texts]
        texts_is_str = True

    # bos token is never used
    # bos_id = tokenizer.bos_id
    eos_id = tokenizer.eos_id

    all_tokens = []
    for text in texts:
        tokens = tokenizer.text_to_ids(text)
        media_positions = [i for i, x in enumerate(tokens) if x == media_start_id]
        for media_pos in media_positions[::-1]:
            tokens[media_pos : media_pos + 1] = [media_start_id] + [-1] * num_media_tokens + [media_end_id]
        tokens = tokens + [eos_id]
        all_tokens.append(tokens)

    # truncate and padding
    result = torch.zeros(len(all_tokens), context_length + add_extra_token, dtype=torch.long)

    for i, tokens in enumerate(all_tokens):
        if len(tokens) > context_length + add_extra_token:
            tokens = tokens[: context_length + add_extra_token]  # Truncate
        result[i, : len(tokens)] = torch.tensor(tokens)

    if texts_is_str:
        result = result[0]
    return result


def get_preprocess_fns(
    model_cfg, data_type, tokenizer=None, is_train=True, add_extra_token=1, media_start_id=None, media_end_id=None,
):
    assert (
        media_start_id is not None and media_end_id is not None
    ), "`media_start_id` and `media_end_id` should be provided."

    # Define transforms
    img_size = (model_cfg.vision.get("img_h"), model_cfg.vision.get("img_w"))
    img_mean = model_cfg.vision.get("img_mean")
    img_std = model_cfg.vision.get("img_std")
    img_transform = image_transform(img_size, is_train=is_train, mean=img_mean, std=img_std,)

    text_transform = lambda x: x
    if tokenizer is not None:
        text_transform = partial(
            tokenize_and_insert_media_tokens,
            tokenizer=tokenizer,
            context_length=model_cfg.per_type_sequence_length[data_type],
            num_media_tokens=model_cfg.num_media_latents,
            add_extra_token=add_extra_token,
            media_start_id=media_start_id,
            media_end_id=media_end_id,
        )
    else:
        raise ValueError("tokenizer should not be None here!")

    return img_transform, text_transform


def transform_fn_for_image_caption(sample, img_transform, text_transform, media_start_token):
    image, text = sample["jpg"], sample["txt"]
    caption_template = lambda x: f"{media_start_token}{x.strip()}"
    text = caption_template(text)
    return img_transform(image), text_transform(text)


def transform_fn_for_image_interleaved(sample, img_transform, text_transform, media_start_token, sim_threshold=0.3):
    info = sample["json"]
    sentences = info["text_list"]

    images, sentence_ixs = [], []
    for sample_image in info["image_info"]:
        image = sample[sample_image["image_name"]]
        # filter to images >= 10KB
        if isinstance(image, bytes):
            continue
        if sample_image["matched_sim"] < sim_threshold:
            continue

        images.append(image)
        sentence_ixs.append(sample_image["matched_text_index"])

    if len(images) == 0:
        raise ValueError("No images in sample")

    keep_ixs = min(len(images), MAX_NUM_IMAGES)
    images = images[:keep_ixs]
    sentence_ixs = sentence_ixs[:keep_ixs]

    def interleaved_template(sentences, sentence_ixs):
        for ix in sentence_ixs:
            sentences[ix] = f"{media_start_token}{sentences[ix]}"
        text = " ".join(sentences)
        return text

    text = interleaved_template(sentences, sentence_ixs)
    images_tensors = torch.stack([img_transform(image) for image in images])
    image_size = images_tensors.shape[1:]
    if len(images_tensors) < MAX_NUM_IMAGES:
        zero_padding = torch.zeros((MAX_NUM_IMAGES - len(images_tensors), *image_size), dtype=torch.float)
        images_tensors = torch.cat((images_tensors, zero_padding), dim=0)

    return images_tensors, text_transform(text)


def compose_batch(inp, model_cfg, tokenizer, add_extra_token, media_start_id, media_end_id, newline_id):
    pad_id = tokenizer.pad_id
    for input in inp:
        media = input[0]

        # vision_x should be of shape (b, T_img, F, C, H, W)
        if len(media.shape) == 3:  # image_caption
            media = rearrange(media, "c h w -> 1 1 c h w")
        elif len(media.shape) == 4:  # image_interleaved
            media = rearrange(media, "T c h w -> T 1 c h w")
        else:
            raise ValueError(f"Media shape length is not expected: {media.shape}.")

        tokens = input[1]
        if add_extra_token:
            tokens = input[1][:-1].contiguous()
            labels = input[1][1:].contiguous().clone().detach()
        else:
            labels = torch.roll(tokens, shifts=-1, dims=0)
            labels[-1] = -1

        labels[labels == media_start_id] = newline_id
        labels[labels == media_end_id] = -1
        labels[labels == pad_id] = -1

        attention_mask, loss_mask, position_ids = _create_ltor_masks_and_position_ids(
            tokens=tokens,
            eod_token=tokenizer.eos_id,
            eod_mask_loss=model_cfg.data.get("eod_mask_loss", False),
            reset_attention_mask=False,
            reset_position_ids=False,
        )

        loss_mask[labels == -1] = 0.0
        tokens[tokens == -1] = 0
        labels[labels == -1] = 0

        yield {
            'tokens': tokens,
            'labels': labels,
            'attention_mask': attention_mask,
            'loss_mask': loss_mask,
            'position_ids': position_ids,
            'media': media,
        }


def build_train_valid_datasets(
    model_cfg, consumed_samples, tokenizer=None, data_type='image_caption',
):
    assert data_type in _DATASET_TYPES, f"`data_type={data_type}` is not available: {_DATASET_TYPES}."

    media_start_token = model_cfg.media_start_token
    media_end_token = model_cfg.media_end_token
    assert (
        media_start_token in tokenizer.vocab and media_end_token in tokenizer.vocab
    ), f"Cannot find media tokens in tokenizer vocab: {media_start_token} {media_end_token}"
    media_start_id = tokenizer.token_to_id(media_start_token)
    media_end_id = tokenizer.token_to_id(media_end_token)
    newline_id = tokenizer.text_to_ids("\n")[-1]

    data_cfg = model_cfg.data.get(data_type)

    no_seqlen_plus_one_input_tokens = model_cfg.data.get('no_seqlen_plus_one_input_tokens', False)
    add_extra_token = 0 if no_seqlen_plus_one_input_tokens else 1

    compose_fn = compose_batch
    if data_type == 'image_caption':
        transform_fn = transform_fn_for_image_caption
    elif data_type == 'image_interleaved':
        transform_fn = transform_fn_for_image_interleaved

    train_img_transform, text_transform = get_preprocess_fns(
        model_cfg,
        data_type=data_type,
        tokenizer=tokenizer,
        is_train=True,
        add_extra_token=add_extra_token,
        media_start_id=media_start_id,
        media_end_id=media_end_id,
    )
    train_data = WebDatasetCommon(
        dataset_cfg=data_cfg,
        consumed_samples=consumed_samples,
        decode_fn=pil_loader if data_type == 'interleaved' else None,
        map_fn=partial(
            transform_fn,
            img_transform=train_img_transform,
            text_transform=text_transform,
            media_start_token=media_start_token,
        ),
        compose_fn=partial(
            compose_fn,
            model_cfg=model_cfg,
            tokenizer=tokenizer,
            add_extra_token=add_extra_token,
            media_start_id=media_start_id,
            media_end_id=media_end_id,
            newline_id=newline_id,
        ),
        is_train=True,
    )

    val_data = None
    if data_cfg.get("validation") is not None and data_cfg.validation.get("dataset_path"):
        val_img_transform, text_transform = get_preprocess_fns(
            model_cfg,
            data_type=data_type,
            tokenizer=tokenizer,
            is_train=False,
            add_extra_token=add_extra_token,
            media_start_id=media_start_id,
            media_end_id=media_end_id,
        )
        val_data = WebDatasetCommon(
            dataset_cfg=data_cfg,
            consumed_samples=0,
            decode_fn=pil_loader if data_type == 'interleaved' else None,
            map_fn=partial(
                transform_fn,
                img_transform=train_img_transform,
                text_transform=text_transform,
                media_start_token=media_start_token,
            ),
            compose_fn=partial(
                compose_fn,
                model_cfg=model_cfg,
                tokenizer=tokenizer,
                add_extra_token=add_extra_token,
                media_start_id=media_start_id,
                media_end_id=media_end_id,
                newline_id=newline_id,
            ),
            is_train=False,
        )

    return train_data, val_data


class MergedKosmosDataLoader:
    def __init__(self, dataloaders):
        self.dataloaders = dataloaders
        self.dataloader_iters = {type: iter(dataloader) for type, dataloader in dataloaders.items()}
        self.lengths = {type: len(dataloader) for type, dataloader in dataloaders.items()}
        self.min_length = min(self.lengths.values())

    def __iter__(self):
        while True:
            try:
                batch = {type: next(iter) for type, iter in self.dataloader_iters.items()}
            except StopIteration:
                return
            yield batch

    def __len__(self):
        return self.min_length
