from typing import Callable, Tuple

import tensorflow as tf


def tensor_feature(value):
    return tf.train.Feature(
        bytes_list=tf.train.BytesList(value=[tf.io.serialize_tensor(value).numpy()])
    )


def resize_image(image: tf.Tensor, size: Tuple[int, int]) -> tf.Tensor:
    """Resizes an image using Lanczos3 interpolation. Expects & returns uint8."""
    assert image.dtype == tf.uint8
    image = tf.image.resize(image, size, method="lanczos3", antialias=True)
    image = tf.cast(tf.clip_by_value(tf.round(image), 0, 255), tf.uint8)
    return image


def resize_depth_image(depth_image: tf.Tensor, size: Tuple[int, int]) -> tf.Tensor:
    """Resizes a depth image using bilinear interpolation. Expects & returns float32 in arbitrary range."""
    assert depth_image.dtype == tf.float32
    if len(depth_image.shape) < 3:
        depth_image = tf.image.resize(
            depth_image[..., None], size, method="bilinear", antialias=True
        )[..., 0]
    else:
        depth_image = tf.image.resize(
            depth_image, size, method="bilinear", antialias=True
        )
    return depth_image


def read_resize_encode_image(path: str, size: Tuple[int, int]) -> tf.Tensor:
    """Reads, decodes, resizes, and then re-encodes an image."""
    data = tf.io.read_file(path)
    image = tf.image.decode_jpeg(data)
    image = resize_image(image, size)
    image = tf.cast(tf.clip_by_value(tf.round(image), 0, 255), tf.uint8)
    return tf.io.encode_jpeg(image, quality=95)


def vmap(fn: Callable) -> Callable:
    """
    Vmap a function over the first dimension of a tensor (or nested structure of tensors). This
    version does NOT parallelize the function; however, it fuses the function calls in a way that
    appears to be more performant than tf.map_fn or tf.vectorized_map (when falling back to
    while_loop) for certain situations.

    Requires the first dimension of the input to be statically known.
    """

    def wrapped(structure):
        return tf.nest.map_structure(
            lambda *x: tf.stack(x),
            *[
                fn(tf.nest.pack_sequence_as(structure, x))
                for x in zip(*map(tf.unstack, tf.nest.flatten(structure)))
            ],
        )

    return wrapped


def parallel_vmap(fn: Callable, num_parallel_calls=tf.data.AUTOTUNE) -> Callable:
    """
    Vmap a function over the first dimension of a tensor (or nested structure of tensors). This
    version attempts to parallelize the function using the tf.data API. I found this to be more
    performant than tf.map_fn or tf.vectorized_map (when falling back to while_loop), but the batch
    call appears to add significant overhead that may make it slower for some situations.
    """

    def wrapped(structure):
        return (
            tf.data.Dataset.from_tensor_slices(structure)
            .map(fn, deterministic=True, num_parallel_calls=num_parallel_calls)
            .batch(
                tf.cast(tf.shape(tf.nest.flatten(structure)[0])[0], tf.int64),
            )
            .get_single_element()
        )

    return wrapped
