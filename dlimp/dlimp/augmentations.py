from typing import Optional

import tensorflow as tf


def random_resized_crop(image, scale, ratio, seed):
    assert image.shape.ndims == 3 or image.shape.ndims == 4
    if image.shape.ndims == 3:
        image = tf.expand_dims(image, axis=0)
    batch_size = tf.shape(image)[0]
    # taken from https://keras.io/examples/vision/nnclr/#random-resized-crops
    log_ratio = (tf.math.log(ratio[0]), tf.math.log(ratio[1]))
    height = tf.shape(image)[1]
    width = tf.shape(image)[2]

    random_scales = tf.random.stateless_uniform((batch_size,), seed, scale[0], scale[1])
    random_ratios = tf.exp(
        tf.random.stateless_uniform((batch_size,), seed, log_ratio[0], log_ratio[1])
    )

    new_heights = tf.clip_by_value(tf.sqrt(random_scales / random_ratios), 0, 1)
    new_widths = tf.clip_by_value(tf.sqrt(random_scales * random_ratios), 0, 1)
    height_offsets = tf.random.stateless_uniform(
        (batch_size,), seed, 0, 1 - new_heights
    )
    width_offsets = tf.random.stateless_uniform((batch_size,), seed, 0, 1 - new_widths)

    bounding_boxes = tf.stack(
        [
            height_offsets,
            width_offsets,
            height_offsets + new_heights,
            width_offsets + new_widths,
        ],
        axis=1,
    )

    image = tf.image.crop_and_resize(
        image, bounding_boxes, tf.range(batch_size), (height, width)
    )

    if image.shape[0] == 1:
        return image[0]
    else:
        return image


def random_rot90(image, seed):
    k = tf.random.stateless_uniform((), seed, 0, 4, dtype=tf.int32)
    return tf.image.rot90(image, k=k)


AUGMENT_OPS = {
    "random_resized_crop": random_resized_crop,
    "random_brightness": tf.image.stateless_random_brightness,
    "random_contrast": tf.image.stateless_random_contrast,
    "random_saturation": tf.image.stateless_random_saturation,
    "random_hue": tf.image.stateless_random_hue,
    "random_flip_left_right": tf.image.stateless_random_flip_left_right,
    "random_flip_up_down": tf.image.stateless_random_flip_up_down,
    "random_rot90": random_rot90,
}


def augment_image(
    image: tf.Tensor, seed: Optional[tf.Tensor] = None, **augment_kwargs
) -> tf.Tensor:
    """Unified image augmentation function for TensorFlow.

    This function is primarily configured through `augment_kwargs`. There must be one kwarg called "augment_order",
    which is a list of strings specifying the augmentation operations to apply and the order in which to apply them. See
    the `AUGMENT_OPS` dictionary above for a list of available operations.

    For each entry in "augment_order", there may be a corresponding kwarg with the same name. The value of this kwarg
    can be a dictionary of kwargs or a sequence of positional args to pass to the corresponding augmentation operation.
    This additional kwarg is required for all operations that take additional arguments other than the image and random
    seed. For example, the "random_resized_crop" operation requires a "scale" and "ratio" argument that can be specified
    either positionally or by name. "random_flip_left_right", on the other hand, does not take any additional arguments
    and so does not require an additional kwarg to configure it.

    Here is an example config:

    ```
    augment_kwargs = {
        "augment_order": ["random_resized_crop", "random_brightness", "random_contrast", "random_flip_left_right"],
        "random_resized_crop": {
            "scale": [0.8, 1.0],
            "ratio": [3/4, 4/3],
        },
        "random_brightness": [0.1],
        "random_contrast": [0.9, 1.1],
    ```

    Args:
        image: A `Tensor` of shape [height, width, channels] with the image. May be uint8 or float32 with values in [0, 255].
        seed (optional): A `Tensor` of shape [2] with the seed for the random number generator.
        **augment_kwargs: Keyword arguments for the augmentation operations. The order of operations is determined by
            the "augment_order" keyword argument.  Other keyword arguments are passed to the corresponding augmentation
            operation. See above for a list of operations.
    """
    if "augment_order" not in augment_kwargs:
        raise ValueError("augment_kwargs must contain an 'augment_order' key.")

    # convert to float at the beginning to avoid each op converting back and
    # forth between uint8 and float32 internally
    orig_dtype = image.dtype
    image = tf.image.convert_image_dtype(image, tf.float32)

    if seed is None:
        seed = tf.random.uniform([2], maxval=tf.dtypes.int32.max, dtype=tf.int32)

    for op in augment_kwargs["augment_order"]:
        seed = tf.random.stateless_uniform([2], seed, maxval=tf.dtypes.int32.max, dtype=tf.int32)
        if op in augment_kwargs:
            if hasattr(augment_kwargs[op], "items"):
                image = AUGMENT_OPS[op](image, seed=seed, **augment_kwargs[op])
            else:
                image = AUGMENT_OPS[op](image, seed=seed, *augment_kwargs[op])
        else:
            image = AUGMENT_OPS[op](image, seed=seed)
        # float images are expected to be in [0, 1]
        image = tf.clip_by_value(image, 0, 1)

    # convert back to original dtype and scale
    image = tf.image.convert_image_dtype(image, orig_dtype, saturate=True)

    return image
