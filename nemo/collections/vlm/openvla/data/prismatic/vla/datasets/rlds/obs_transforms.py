"""
obs_transforms.py

Contains observation-level transforms used in the orca data pipeline.

These transforms operate on the "observation" dictionary, and are applied at a per-frame level.
"""

from typing import Dict, Tuple, Union

import dlimp as dl
import tensorflow as tf
from absl import logging


# ruff: noqa: B023
def augment(obs: Dict, seed: tf.Tensor, augment_kwargs: Union[Dict, Dict[str, Dict]]) -> Dict:
    """Augments images, skipping padding images."""
    image_names = {key[6:] for key in obs if key.startswith("image_")}

    # "augment_order" is required in augment_kwargs, so if it's there, we can assume that the user has passed
    # in a single augmentation dict (otherwise, we assume that the user has passed in a mapping from image
    # name to augmentation dict)
    if "augment_order" in augment_kwargs:
        augment_kwargs = {name: augment_kwargs for name in image_names}

    for i, name in enumerate(image_names):
        if name not in augment_kwargs:
            continue
        kwargs = augment_kwargs[name]
        logging.debug(f"Augmenting image_{name} with kwargs {kwargs}")
        obs[f"image_{name}"] = tf.cond(
            obs["pad_mask_dict"][f"image_{name}"],
            lambda: dl.transforms.augment_image(
                obs[f"image_{name}"],
                **kwargs,
                seed=seed + i,  # augment each image differently
            ),
            lambda: obs[f"image_{name}"],  # skip padding images
        )

    return obs


def decode_and_resize(
    obs: Dict,
    resize_size: Union[Tuple[int, int], Dict[str, Tuple[int, int]]],
    depth_resize_size: Union[Tuple[int, int], Dict[str, Tuple[int, int]]],
) -> Dict:
    """Decodes images and depth images, and then optionally resizes them."""
    image_names = {key[6:] for key in obs if key.startswith("image_")}
    depth_names = {key[6:] for key in obs if key.startswith("depth_")}

    if isinstance(resize_size, tuple):
        resize_size = {name: resize_size for name in image_names}
    if isinstance(depth_resize_size, tuple):
        depth_resize_size = {name: depth_resize_size for name in depth_names}

    for name in image_names:
        if name not in resize_size:
            logging.warning(
                f"No resize_size was provided for image_{name}. This will result in 1x1 "
                "padding images, which may cause errors if you mix padding and non-padding images."
            )
        image = obs[f"image_{name}"]
        if image.dtype == tf.string:
            if tf.strings.length(image) == 0:
                # this is a padding image
                image = tf.zeros((*resize_size.get(name, (1, 1)), 3), dtype=tf.uint8)
            else:
                image = tf.io.decode_image(image, expand_animations=False, dtype=tf.uint8)
        elif image.dtype != tf.uint8:
            raise ValueError(f"Unsupported image dtype: found image_{name} with dtype {image.dtype}")
        if name in resize_size:
            image = dl.transforms.resize_image(image, size=resize_size[name])
        obs[f"image_{name}"] = image

    for name in depth_names:
        if name not in depth_resize_size:
            logging.warning(
                f"No depth_resize_size was provided for depth_{name}. This will result in 1x1 "
                "padding depth images, which may cause errors if you mix padding and non-padding images."
            )
        depth = obs[f"depth_{name}"]

        if depth.dtype == tf.string:
            if tf.strings.length(depth) == 0:
                depth = tf.zeros((*depth_resize_size.get(name, (1, 1)), 1), dtype=tf.float32)
            else:
                depth = tf.io.decode_image(depth, expand_animations=False, dtype=tf.float32)[..., 0]
        elif depth.dtype != tf.float32:
            raise ValueError(f"Unsupported depth dtype: found depth_{name} with dtype {depth.dtype}")

        if name in depth_resize_size:
            depth = dl.transforms.resize_depth_image(depth, size=depth_resize_size[name])

        obs[f"depth_{name}"] = depth

    return obs
