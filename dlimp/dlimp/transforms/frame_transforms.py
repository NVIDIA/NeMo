from functools import partial
from typing import Any, Callable, Dict, Sequence, Tuple, Union

import tensorflow as tf

from dlimp.augmentations import augment_image
from dlimp.utils import resize_depth_image, resize_image

from .common import selective_tree_map


def decode_images(
    x: Dict[str, Any], match: Union[str, Sequence[str]] = "image"
) -> Dict[str, Any]:
    """Can operate on nested dicts. Decodes any leaves that have `match` anywhere in their path."""
    if isinstance(match, str):
        match = [match]

    return selective_tree_map(
        x,
        lambda keypath, value: any([s in keypath for s in match])
        and value.dtype == tf.string,
        partial(tf.io.decode_image, expand_animations=False),
    )


def resize_images(
    x: Dict[str, Any],
    match: Union[str, Sequence[str]] = "image",
    size: Tuple[int, int] = (128, 128),
) -> Dict[str, Any]:
    """Can operate on nested dicts. Resizes any leaves that have `match` anywhere in their path. Takes uint8 images
    as input and returns float images (still in [0, 255]).
    """
    if isinstance(match, str):
        match = [match]

    return selective_tree_map(
        x,
        lambda keypath, value: any([s in keypath for s in match])
        and value.dtype == tf.uint8,
        partial(resize_image, size=size),
    )


def resize_depth_images(
    x: Dict[str, Any],
    match: Union[str, Sequence[str]] = "depth",
    size: Tuple[int, int] = (128, 128),
) -> Dict[str, Any]:
    """Can operate on nested dicts. Resizes any leaves that have `match` anywhere in their path. Takes float32 images
    as input and returns float images (in arbitrary range).
    """
    if isinstance(match, str):
        match = [match]

    return selective_tree_map(
        x,
        lambda keypath, value: any([s in keypath for s in match])
        and value.dtype == tf.float32,
        partial(resize_depth_image, size=size),
    )


def augment(
    x: Dict[str, Any],
    match: Union[str, Callable[[str, Any], bool]] = "*image",
    traj_identical: bool = True,
    keys_identical: bool = True,
    augment_kwargs: dict = {},
) -> Dict[str, Any]:
    """
    Augments the input dictionary `x` by applying image augmentation to all values whose keypath contains `match`.

    Args:
        x (Dict[str, Any]): The input dictionary to augment.
        match (str or Callable[[str, Any], bool]): See documentation for `selective_tree_map`.
            Defaults to "*image", which matches all leaves whose key ends in "image".
        traj_identical (bool, optional): Whether to use the same random seed for all images in a trajectory.
        keys_identical (bool, optional): Whether to use the same random seed for all keys that are augmented.
        augment_kwargs (dict, optional): Additional keyword arguments to pass to the `augment_image` function.
    """
    toplevel_seed = tf.random.uniform([2], 0, 2**31 - 1, dtype=tf.int32)

    def map_fn(value):
        if keys_identical and traj_identical:
            seed = [x["_traj_index"], x["_traj_index"]]
        elif keys_identical and not traj_identical:
            seed = toplevel_seed
        elif not keys_identical and traj_identical:
            raise NotImplementedError()
        else:
            seed = None

        return augment_image(value, seed=seed, **augment_kwargs)

    return selective_tree_map(
        x,
        match,
        map_fn,
    )
