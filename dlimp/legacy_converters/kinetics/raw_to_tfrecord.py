"""
Converts data from the kinetics raw mp4 format to TFRecord format.

The assumptions of the data format are as follows:

    k400/
        annotations/
            train.csv
            val.csv
        train/
            id{time_start}_{time_end}.mp4
            ...
        val/
            id{time_start}_{time_end}.mp4
            ...

The --input_path should be the path to the k400 directory, and the --output_path
should be the path to the directory where the TFRecord files will be written.
--aspect_ratio controls whether the videos are first center cropped to 4:3 aspect before being resized to 240x240.

Follow instructions in https://github.com/cvdfoundation/kinetics-dataset for downloading the specific kinetics dataset of your choosing.
"""

import os

import imageio
import numpy as np
import pandas as pd
import tensorflow as tf
import tqdm
from absl import app, flags, logging
from tqdm_multiprocess import TqdmMultiProcessPool

from dlimp.utils import resize_image, tensor_feature

FLAGS = flags.FLAGS

flags.DEFINE_string("input_path", None, "Input path", required=True)
flags.DEFINE_string("output_path", None, "Output path", required=True)
flags.DEFINE_bool("overwrite", False, "Overwrite existing files")
flags.DEFINE_bool("aspect_ratio", False, "Whether to preserve aspect ratio")
flags.DEFINE_integer("num_workers", 8, "Number of threads to use")
flags.DEFINE_integer("shard_size", 400, "Maximum number of trajectories per shard")


# create a tfrecord for a group of trajectories
def create_tfrecord(shard, output_path, tqdm_func, global_tqdm):
    writer = tf.io.TFRecordWriter(output_path)
    for item in shard:
        try:
            video = imageio.mimread(item["path"], format="mp4", memtest=False)
        except OSError:
            # corrupted video
            global_tqdm.update(1)
            continue

        if FLAGS.aspect_ratio:
            # height and width both varies in this dataset
            height = video[0].shape[0]
            width = video[0].shape[1]
            ratio = width / height

            if ratio > 4 / 3:
                # center crop horizontally
                desired_width = int(np.round(4 / 3 * height))
                video = [
                    image[
                        :, (width - desired_width) // 2 : (width + desired_width) // 2
                    ]
                    for image in video
                ]
            elif ratio < 4 / 3:
                # center crop vertically
                desired_height = int(np.round(3 / 4 * width))
                video = [
                    image[
                        (height - desired_height) // 2 : (height + desired_height) // 2,
                        :,
                    ]
                    for image in video
                ]

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
                    "label": tensor_feature(item["label"]),
                }
            )
        )
        writer.write(example.SerializeToString())
        global_tqdm.update(1)

    writer.close()
    global_tqdm.write(f"Finished {output_path}")


def main(_):
    tf.config.set_visible_devices(
        [], "GPU"
    )  # TF might look for GPUs and crash out if it finds one

    if tf.io.gfile.exists(FLAGS.output_path):
        if FLAGS.overwrite:
            logging.info(f"Deleting {FLAGS.output_path}")
            tf.io.gfile.rmtree(FLAGS.output_path)
        else:
            logging.info(f"{FLAGS.output_path} exists, exiting")
            return

    # load annotations
    with open(os.path.join(FLAGS.input_path, "annotations", "train.csv"), "r") as f:
        train_annotations = pd.read_csv(f)
    with open(os.path.join(FLAGS.input_path, "annotations", "val.csv"), "r") as f:
        val_annotations = pd.read_csv(f)

    # filter
    train_annotations = [
        (row["label"], row["youtube_id"], row["time_start"], row["time_end"])
        for idx, row in train_annotations.iterrows()
    ]
    val_annotations = [
        (row["label"], row["youtube_id"], row["time_start"], row["time_end"])
        for idx, row in val_annotations.iterrows()
    ]

    print(f"------ Train: {len(train_annotations)}, Val: {len(val_annotations)} ------")

    train = []
    count = 0
    for label, youtube_id, time_start, time_end in train_annotations:
        # Downloader was made by geniuses as you can tell
        path = (
            f"{FLAGS.input_path}/train/{youtube_id}_{time_start:06d}_{time_end:06d}.mp4"
        )
        if not os.path.exists(path):
            count += 1
            continue
        train.append({"path": path, "label": label})

    print("Number of train files not found: ", count)

    val = []
    count = 0
    for label, youtube_id, time_start, time_end in val_annotations:
        path = (
            f"{FLAGS.input_path}/val/{youtube_id}_{time_start:06d}_{time_end:06d}.mp4"
        )
        if not os.path.exists(path):
            count += 1
            continue
        val.append({"path": path, "label": label})

    print("Number of val files not found: ", count)

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
