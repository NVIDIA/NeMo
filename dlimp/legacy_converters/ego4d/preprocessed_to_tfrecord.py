"""
Converts data from a preprocessed Ego4D format to TFRecord format.

Expects a manifest.csv file with paths to directories containing JPEG files. Images should be 224x224.
"""

import os

import numpy as np
import pandas as pd
import tensorflow as tf
import tqdm
from absl import app, flags, logging
from tqdm_multiprocess import TqdmMultiProcessPool

from dlimp.utils import read_resize_encode_image, tensor_feature

FLAGS = flags.FLAGS

flags.DEFINE_string("input_path", None, "Input path", required=True)
flags.DEFINE_string("output_path", None, "Output path", required=True)
flags.DEFINE_bool("overwrite", False, "Overwrite existing files")
flags.DEFINE_float(
    "train_proportion", 0.9, "Proportion of data to use for training (rather than val)"
)
flags.DEFINE_integer("num_workers", 8, "Number of threads to use")
flags.DEFINE_integer("shard_size", 200, "Maximum number of trajectories per shard")

IMAGE_SIZE = (224, 224)


# create a tfrecord for a group of trajectories
def create_tfrecord(manifest, output_path, tqdm_func, global_tqdm):
    writer = tf.io.TFRecordWriter(output_path)
    for _, row in manifest.iterrows():
        # left-zero-pad the frame indices to length 6; this weird way of doing it is left over from when this had to be
        # done in a tf graph; I'm leaving it in case I need it again someday
        indices = tf.as_string(tf.range(row["num_frames"]))
        indices = tf.strings.bytes_split(indices)
        n = 6 - indices.row_lengths()
        zeros = tf.fill([tf.reduce_sum(n)], "0")
        zeros = tf.RaggedTensor.from_row_lengths(zeros, n)
        padded = tf.concat([zeros, indices], axis=1)
        padded = tf.strings.reduce_join(padded, axis=1)

        # get the paths to all of the frames
        paths = tf.strings.join([row["directory"], "/", padded, ".jpg"])

        # read, resize, and re-encode the images
        images = [read_resize_encode_image(path, IMAGE_SIZE) for path in paths]

        example = tf.train.Example(
            features=tf.train.Features(
                feature={
                    "obs": tensor_feature(images),
                    "lang": tensor_feature(row["text"]),
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

    # get the manifest
    manifest = pd.read_csv(os.path.join(FLAGS.input_path, "manifest.csv"))

    assert list(manifest.columns) == ["index", "directory", "num_frames", "text"]

    # get rid of the invalid path prefixes and replace them with the actual
    # dataset path prefix
    manifest["directory"] = manifest["directory"].apply(
        lambda x: os.path.join(FLAGS.input_path, *x.strip("/").split("/")[-2:])
    )

    # train/val split
    manifest = manifest.sample(frac=1.0, random_state=0)
    train_manifest = manifest.iloc[: int(len(manifest) * FLAGS.train_proportion)]
    val_manifest = manifest.iloc[int(len(manifest) * FLAGS.train_proportion) :]

    # shard paths
    train_shards = np.array_split(
        train_manifest, np.ceil(len(train_manifest) / FLAGS.shard_size)
    )
    val_shards = np.array_split(
        val_manifest, np.ceil(len(val_manifest) / FLAGS.shard_size)
    )

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
        total=len(manifest),
        dynamic_ncols=True,
        position=0,
        desc="Total progress",
    ) as pbar:
        pool.map(pbar, tasks, lambda _: None, lambda _: None)


if __name__ == "__main__":
    app.run(main)
