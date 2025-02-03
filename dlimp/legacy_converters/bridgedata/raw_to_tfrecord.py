"""
Converts data from the BridgeData raw format to TFRecord format.

Consider the following directory structure for the input data:

    bridgedata_raw/
        rss/
            toykitchen2/
                set_table/
                    00/
                        2022-01-01_00-00-00/
                            collection_metadata.json
                            config.json
                            diagnostics.png
                            raw/
                                traj_group0/
                                    traj0/
                                        obs_dict.pkl
                                        policy_out.pkl
                                        agent_data.pkl
                                        images0/
                                            im_0.jpg
                                            im_1.jpg
                                            ...
                                    ...
                                ...
                    01/
                    ...

The --depth parameter controls how much of the data to process at the
--input_path; for example, if --depth=5, then --input_path should be
"bridgedata_raw", and all data will be processed. If --depth=3, then
--input_path should be "bridgedata_raw/rss/toykitchen2", and only data
under "toykitchen2" will be processed.

Can write directly to Google Cloud Storage, but not read from it.
"""

import glob
import logging
import os
import pickle
import random
from datetime import datetime
from functools import partial
from multiprocessing import Pool

import numpy as np
import tensorflow as tf
import tqdm
from absl import app, flags
from tqdm_multiprocess import TqdmMultiProcessPool

import dlimp as dl
from dlimp.utils import read_resize_encode_image, tensor_feature

FLAGS = flags.FLAGS

flags.DEFINE_string("input_path", None, "Input path", required=True)
flags.DEFINE_string("output_path", None, "Output path", required=True)
flags.DEFINE_integer(
    "depth",
    5,
    "Number of directories deep to traverse to the dated directory. Looks for"
    "{input_path}/dir_1/dir_2/.../dir_{depth-1}/2022-01-01_00-00-00/...",
)
flags.DEFINE_bool("overwrite", False, "Overwrite existing files")
flags.DEFINE_float(
    "train_proportion", 0.9, "Proportion of data to use for training (rather than val)"
)
flags.DEFINE_integer("num_workers", 8, "Number of threads to use")
flags.DEFINE_integer("shard_size", 200, "Maximum number of trajectories per shard")

IMAGE_SIZE = (256, 256)
CAMERA_VIEWS = {"images0", "images1", "images2"}


def process_images(path):  # processes images at a trajectory level
    image_dirs = set(os.listdir(str(path))).intersection(CAMERA_VIEWS)
    image_paths = [
        sorted(
            glob.glob(os.path.join(path, image_dir, "im_*.jpg")),
            key=lambda x: int(x.split("_")[-1].split(".")[0]),
        )
        for image_dir in image_dirs
    ]

    filenames = [[path.split("/")[-1] for path in x] for x in image_paths]
    assert all(x == filenames[0] for x in filenames)

    d = {
        image_dir: [read_resize_encode_image(path, IMAGE_SIZE) for path in p]
        for image_dir, p in zip(image_dirs, image_paths)
    }

    for missing in CAMERA_VIEWS - set(d.keys()):
        d[missing] = [""] * len(
            image_paths[0]
        )  # empty string is a placeholder for missing images

    return d


def process_state(path):
    fp = os.path.join(path, "obs_dict.pkl")
    with open(fp, "rb") as f:
        x = pickle.load(f)
    return x["full_state"]


def process_actions(path):
    fp = os.path.join(path, "policy_out.pkl")
    with open(fp, "rb") as f:
        act_list = pickle.load(f)
    if isinstance(act_list[0], dict):
        act_list = [x["actions"] for x in act_list]
    return act_list


def process_lang(path):
    fp = os.path.join(path, "lang.txt")
    text = ""  # empty string is a placeholder for missing text
    if os.path.exists(fp):
        with open(fp, "r") as f:
            text = f.readline().strip()

    return text


# create a tfrecord for a group of trajectories
def create_tfrecord(paths, output_path, tqdm_func, global_tqdm):
    writer = tf.io.TFRecordWriter(output_path)
    for path in paths:
        try:
            # Data collected prior to 7-23 has a delay of 1, otherwise a delay of 0
            date_time = datetime.strptime(path.split("/")[-4], "%Y-%m-%d_%H-%M-%S")
            latency_shift = date_time < datetime(2021, 7, 23)

            out = dict()

            out["obs"] = process_images(path)
            out["obs"]["state"] = process_state(path)
            out["actions"] = process_actions(path)
            out["lang"] = process_lang(path)

            # shift the actions according to camera latency
            if latency_shift:
                out["obs"] = {k: v[1:] for k, v in out["obs"].items()}
                out["actions"] = out["actions"][:-1]

            # append a null action to the end
            out["actions"].append(np.zeros_like(out["actions"][0]))

            assert (
                len(out["actions"])
                == len(out["obs"]["state"])
                == len(out["obs"]["images0"])
            )

            example = tf.train.Example(
                features=tf.train.Features(
                    feature={
                        k: tensor_feature(v)
                        for k, v in dl.transforms.flatten_dict(out).items()
                    }
                )
            )
            writer.write(example.SerializeToString())
        except Exception as e:
            import sys
            import traceback

            traceback.print_exc()
            logging.error(f"Error processing {path}")
            sys.exit(1)

        global_tqdm.update(1)

    writer.close()
    global_tqdm.write(f"Finished {output_path}")


def get_traj_paths(path, train_proportion):
    train_traj = []
    val_traj = []
    for dated_folder in os.listdir(path):
        # a mystery left by the greats of the past
        if "lmdb" in dated_folder:
            continue

        search_path = os.path.join(path, dated_folder, "raw", "traj_group*", "traj*")
        all_traj = glob.glob(search_path)
        if not all_traj:
            logging.info(f"no trajs found in {search_path}")
            continue

        random.shuffle(all_traj)
        train_traj += all_traj[: int(len(all_traj) * train_proportion)]
        val_traj += all_traj[int(len(all_traj) * train_proportion) :]
    return train_traj, val_traj


def main(_):
    assert FLAGS.depth >= 1

    if tf.io.gfile.exists(FLAGS.output_path):
        if FLAGS.overwrite:
            logging.info(f"Deleting {FLAGS.output_path}")
            tf.io.gfile.rmtree(FLAGS.output_path)
        else:
            logging.info(f"{FLAGS.output_path} exists, exiting")
            return

    # each path is a directory that contains dated directories
    paths = glob.glob(os.path.join(FLAGS.input_path, *("*" * (FLAGS.depth - 1))))

    # get trajecotry paths in parallel
    with Pool(FLAGS.num_workers) as p:
        train_paths, val_paths = zip(
            *p.map(
                partial(get_traj_paths, train_proportion=FLAGS.train_proportion), paths
            )
        )

    train_paths = [x for y in train_paths for x in y]
    val_paths = [x for y in val_paths for x in y]
    random.shuffle(train_paths)
    random.shuffle(val_paths)

    # shard paths
    train_shards = np.array_split(
        train_paths, np.ceil(len(train_paths) / FLAGS.shard_size)
    )
    val_shards = np.array_split(val_paths, np.ceil(len(val_paths) / FLAGS.shard_size))

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
        total=len(train_paths) + len(val_paths),
        dynamic_ncols=True,
        position=0,
        desc="Total progress",
    ) as pbar:
        pool.map(pbar, tasks, lambda _: None, lambda _: None)


if __name__ == "__main__":
    app.run(main)
