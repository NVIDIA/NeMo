"""
Converts data from a raw somethingsomething format to TFRecord format.
"""

import json
import os
from email.mime import image

import imageio
import numpy as np
import tensorflow as tf
import tqdm
from absl import app, flags, logging
from tqdm_multiprocess import TqdmMultiProcessPool

from dlimp.utils import resize_image, tensor_feature

FLAGS = flags.FLAGS

flags.DEFINE_string("input_path", None, "Input path", required=True)
flags.DEFINE_string("label_path", None, "Labels to filter by", required=True)
flags.DEFINE_string("output_path", None, "Output path", required=True)
flags.DEFINE_bool("overwrite", False, "Overwrite existing files")
flags.DEFINE_integer("num_workers", 8, "Number of threads to use")
flags.DEFINE_integer("shard_size", 200, "Maximum number of trajectories per shard")


# create a tfrecord for a group of trajectories
def create_tfrecord(shard, output_path, tqdm_func, global_tqdm):
    writer = tf.io.TFRecordWriter(output_path)
    for item in shard:
        video = imageio.mimread(item["path"])

        # center crop to 4:3 aspect ratio (same as bridge)
        # height is always 240, width varies
        width = video[0].shape[1]
        if width > 320:
            # center crop horizontally to 320
            video = [
                image[:, (width - 320) // 2 : (width + 320) // 2] for image in video
            ]
        elif width < 320:
            # center crop vertically
            desired_height = int(np.round(3 / 4 * width))
            video = [
                image[(240 - desired_height) // 2 : (240 + desired_height) // 2, :]
                for image in video
            ]
        assert all(
            np.isclose(image.shape[1] / image.shape[0], 4 / 3, atol=0.01)
            for image in video
        )

        # now resize to square 240x240
        video = [resize_image(image, (240, 240)) for image in video]
        video = [
            tf.cast(tf.clip_by_value(tf.round(image), 0, 255), tf.uint8)
            for image in video
        ]

        # encode
        video = [tf.io.encode_jpeg(image, quality=95) for image in video]

        example = tf.train.Example(
            features=tf.train.Features(
                feature={
                    "obs": tensor_feature(video),
                    "lang": tensor_feature(item["lang"]),
                }
            )
        )
        writer.write(example.SerializeToString())

        global_tqdm.update(1)

    writer.close()
    global_tqdm.write(f"Finished {output_path}")


def main(_):
    if tf.io.gfile.exists(FLAGS.output_path):
        if FLAGS.overwrite:
            logging.info(f"Deleting {FLAGS.output_path}")
            tf.io.gfile.rmtree(FLAGS.output_path)
        else:
            logging.info(f"{FLAGS.output_path} exists, exiting")
            return

    # load annotations
    with open(os.path.join(FLAGS.input_path, "annotations", "train.json"), "r") as f:
        train_annotations = json.load(f)
    with open(
        os.path.join(FLAGS.input_path, "annotations", "validation.json"), "r"
    ) as f:
        val_annotations = json.load(f)

    # load labels to filter by
    with open(FLAGS.label_path, "r") as f:
        labels = set(json.load(f).keys())

    # filter
    train_annotations = [
        x
        for x in train_annotations
        if x["template"].replace("[", "").replace("]", "") in labels
    ]
    val_annotations = [
        x
        for x in val_annotations
        if x["template"].replace("[", "").replace("]", "") in labels
    ]

    print(f"------ Train: {len(train_annotations)}, Val: {len(val_annotations)} ------")

    # get video paths
    train = [
        {
            "path": os.path.join(
                FLAGS.input_path, "20bn-something-something-v2", x["id"]
            )
            + ".webm",
            "lang": x["label"],
        }
        for x in train_annotations
    ]
    val = [
        {
            "path": os.path.join(
                FLAGS.input_path, "20bn-something-something-v2", x["id"]
            )
            + ".webm",
            "lang": x["label"],
        }
        for x in val_annotations
    ]

    # shard
    train_shards = np.array_split(train, np.ceil(len(train) / FLAGS.shard_size))
    val_shards = np.array_split(val, np.ceil(len(val) / FLAGS.shard_size))

    # create output paths
    tf.io.gfile.makedirs(os.path.join(FLAGS.output_path, "train"))
    tf.io.gfile.makedirs(os.path.join(FLAGS.output_path, "val"))
    train_output_paths = [
        os.path.join(FLAGS.output_path, "train", f"{i}.tfrecord")
        for i in range(len(train_shards))
    ]
    val_output_paths = [
        os.path.join(FLAGS.output_path, "val", f"{i}.tfrecord")
        for i in range(len(val_shards))
    ]

    # create tasks (see tqdm_multiprocess documenation)
    tasks = [
        (create_tfrecord, (train_shards[i], train_output_paths[i]))
        for i in range(len(train_shards))
    ] + [
        (create_tfrecord, (val_shards[i], val_output_paths[i]))
        for i in range(len(val_shards))
    ]

    # run tasks
    pool = TqdmMultiProcessPool(FLAGS.num_workers)
    with tqdm.tqdm(
        total=len(train) + len(val),
        dynamic_ncols=True,
        position=0,
        desc="Total progress",
    ) as pbar:
        pool.map(pbar, tasks, lambda _: None, lambda _: None)


if __name__ == "__main__":
    app.run(main)
