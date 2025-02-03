import glob
import json
import os
import pickle
from datetime import datetime

import numpy as np
import tensorflow_datasets as tfds
from absl import logging
from dataset_builder import MultiThreadedDatasetBuilder
from PIL import Image

# we ignore the small amount of data that contains >4 views
N_VIEWS = 4
IMAGE_SIZE = (480, 640)
DEPTH = 5
TRAIN_PROPORTION = 0.9

ORIG_NAMES = [f"images{i}" for i in range(N_VIEWS)]
NEW_NAMES = [f"image_{i}" for i in range(N_VIEWS)]


def read_image(path: str) -> np.ndarray:
    with Image.open(path) as im:
        # depth should be uint16 (I;16), but PIL has a bug where it reads as int32 (I)
        # there are also few trajectories where it's uint8 (L) for some reason
        # we just cast to uint16 in both cases
        assert im.mode == "RGB" or im.mode == "I" or im.mode == "L", (path, im.mode)
        assert im.size == (640, 480), (path, im.size)
        arr = np.array(im)
        if arr.ndim == 2:
            return arr[..., None].astype(np.uint16)
        else:
            assert arr.ndim == 3 and arr.shape[-1] == 3, (path, arr.shape)
            assert arr.dtype == np.uint8, (path, arr.dtype)
            return arr

    # you can speed things up significantly by skipping image decoding/re-encoding by using the line below,
    # but then you also need to skip the checks
    # return open(path, "rb").read()


def process_images(path):  # processes images at a trajectory level
    image_dirs = set(os.listdir(str(path))).intersection(set(ORIG_NAMES))
    image_paths = [
        sorted(
            glob.glob(os.path.join(path, image_dir, "im_*.jpg")),
            key=lambda x: int(x.split("_")[-1].split(".")[0]),
        )
        for image_dir in image_dirs
    ]

    filenames = [[path.split("/")[-1] for path in x] for x in image_paths]
    assert all(x == filenames[0] for x in filenames), (path, filenames)

    d = {
        image_dir: [read_image(path) for path in p]
        for image_dir, p in zip(image_dirs, image_paths)
    }

    return d


def process_depth(path):
    depth_path = os.path.join(path, "depth_images0")
    if os.path.exists(depth_path):
        image_paths = sorted(
            glob.glob(os.path.join(depth_path, "im_*.png")),
            key=lambda x: int(x.split("_")[-1].split(".")[0]),
        )
        return [read_image(path) for path in image_paths]
    else:
        return None


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


class BridgeDataset(MultiThreadedDatasetBuilder):
    """DatasetBuilder for bridge dataset."""

    VERSION = tfds.core.Version("1.0.0")
    RELEASE_NOTES = {
        "1.0.0": "Initial release.",
    }
    MANUAL_DOWNLOAD_INSTRUCTIONS = "You can download the raw BridgeData from https://rail.eecs.berkeley.edu/datasets/bridge_release/data/."

    NUM_WORKERS = 16
    CHUNKSIZE = 1000

    def _info(self) -> tfds.core.DatasetInfo:
        """Dataset metadata (homepage, citation,...)."""
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict(
                {
                    "steps": tfds.features.Dataset(
                        {
                            "observation": tfds.features.FeaturesDict(
                                {
                                    "image_0": tfds.features.Image(
                                        shape=IMAGE_SIZE + (3,),
                                        dtype=np.uint8,
                                        encoding_format="jpeg",
                                        doc="Main camera RGB observation (fixed position).",
                                    ),
                                    "image_1": tfds.features.Image(
                                        shape=IMAGE_SIZE + (3,),
                                        dtype=np.uint8,
                                        encoding_format="jpeg",
                                        doc="Side camera RGB observation (varied position).",
                                    ),
                                    "image_2": tfds.features.Image(
                                        shape=IMAGE_SIZE + (3,),
                                        dtype=np.uint8,
                                        encoding_format="jpeg",
                                        doc="Side camera RGB observation (varied position)",
                                    ),
                                    "image_3": tfds.features.Image(
                                        shape=IMAGE_SIZE + (3,),
                                        dtype=np.uint8,
                                        encoding_format="jpeg",
                                        doc="Wrist camera RGB observation.",
                                    ),
                                    "depth_0": tfds.features.Image(
                                        shape=IMAGE_SIZE + (1,),
                                        dtype=np.uint16,
                                        encoding_format="png",
                                        doc="Main camera depth observation (fixed position).",
                                    ),
                                    "state": tfds.features.Tensor(
                                        shape=(7,),
                                        dtype=np.float32,
                                        doc="Robot end effector state, consists of [3x XYZ, 3x roll-pitch-yaw, 1x gripper]",
                                    ),
                                }
                            ),
                            "action": tfds.features.Tensor(
                                shape=(7,),
                                dtype=np.float32,
                                doc="Robot action, consists of [3x XYZ delta, 3x roll-pitch-yaw delta, 1x gripper absolute].",
                            ),
                            "is_first": tfds.features.Scalar(
                                dtype=np.bool_, doc="True on first step of the episode."
                            ),
                            "is_last": tfds.features.Scalar(
                                dtype=np.bool_, doc="True on last step of the episode."
                            ),
                            "language_instruction": tfds.features.Text(
                                doc="Language Instruction."
                            ),
                        }
                    ),
                    "episode_metadata": tfds.features.FeaturesDict(
                        {
                            "file_path": tfds.features.Text(
                                doc="Path to the original data file."
                            ),
                            "has_image_0": tfds.features.Scalar(
                                dtype=np.bool_,
                                doc="True if image0 exists in observation, otherwise dummy value.",
                            ),
                            "has_image_1": tfds.features.Scalar(
                                dtype=np.bool_,
                                doc="True if image1 exists in observation, otherwise dummy value.",
                            ),
                            "has_image_2": tfds.features.Scalar(
                                dtype=np.bool_,
                                doc="True if image2 exists in observation, otherwise dummy value.",
                            ),
                            "has_image_3": tfds.features.Scalar(
                                dtype=np.bool_,
                                doc="True if image3 exists in observation, otherwise dummy value.",
                            ),
                            "has_depth_0": tfds.features.Scalar(
                                dtype=np.bool_,
                                doc="True if depth0 exists in observation, otherwise dummy value.",
                            ),
                            "has_language": tfds.features.Scalar(
                                dtype=np.bool_,
                                doc="True if language exists in observation, otherwise empty string.",
                            ),
                        }
                    ),
                }
            )
        )

    @classmethod
    def _process_example(cls, example_input):
        """Process a single example."""
        path, camera_topics = example_input

        out = dict()

        out["images"] = process_images(path)
        out["depth"] = process_depth(path)
        out["state"] = process_state(path)
        out["actions"] = process_actions(path)
        out["lang"] = process_lang(path)

        # data collected prior to 7-23 has a delay of 1, otherwise a delay of 0
        date_time = datetime.strptime(path.split("/")[-4], "%Y-%m-%d_%H-%M-%S")
        latency_shift = date_time < datetime(2021, 7, 23)

        # shift the actions according to camera latency
        if latency_shift:
            out["images"] = {k: v[1:] for k, v in out["images"].items()}
            out["state"] = out["state"][1:]
            out["actions"] = out["actions"][:-1]
            if out["depth"] is not None:
                out["depth"] = out["depth"][1:]

        # append a null action to the end
        out["actions"].append(np.zeros_like(out["actions"][0]))

        assert len(out["actions"]) == len(out["state"]) == len(out["images"]["images0"])

        # assemble episode
        episode = []
        episode_metadata = dict()

        # map original image name to correct image name according to logged camera topics
        orig_to_new = dict()
        for image_idx in range(len(out["images"])):
            orig_key = ORIG_NAMES[image_idx]

            if camera_topics[image_idx] in [
                "/cam0/image_raw",
                "/camera0/color/image_raw",
                "/D435/color/image_raw",
            ]:
                # fixed cam should always be image_0
                new_key = "image_0"
                # assert new_key[-1] == orig_key[-1], episode_path
            elif camera_topics[image_idx] == "/wrist/image_raw":
                # wrist cam should always be image_3
                new_key = "image_3"
            elif camera_topics[image_idx] in [
                "/cam1/image_raw",
                "/cam2/image_raw",
                "/cam3/image_raw",
                "/cam4/image_raw",
                "/camera1/color/image_raw",
                "/camera3/color/image_raw",
                "/camera2/color/image_raw",
                "/camera4/color/image_raw",
                "/blue/image_raw",
                "/yellow/image_raw",
            ]:
                # other cams can be either image_1 or image_2
                if "image_1" in list(orig_to_new.values()):
                    new_key = "image_2"
                else:
                    new_key = "image_1"
            else:
                raise ValueError(f"Unexpected camera topic {camera_topics[image_idx]}")

            orig_to_new[orig_key] = new_key
            episode_metadata[f"has_{new_key}"] = True

        # record which images are missing
        missing_keys = set(NEW_NAMES) - set(orig_to_new.values())
        for missing in missing_keys:
            episode_metadata[f"has_{missing}"] = False

        episode_metadata["has_depth_0"] = out["depth"] is not None

        instruction = out["lang"]

        for i in range(len(out["actions"])):
            observation = {
                "state": out["state"][i].astype(np.float32),
            }

            for orig_key in out["images"]:
                new_key = orig_to_new[orig_key]
                observation[new_key] = out["images"][orig_key][i]
            for missing in missing_keys:
                observation[missing] = np.zeros(IMAGE_SIZE + (3,), dtype=np.uint8)
            if episode_metadata["has_depth_0"]:
                observation["depth_0"] = out["depth"][i]
            else:
                observation["depth_0"] = np.zeros(IMAGE_SIZE + (1,), dtype=np.uint16)

            episode.append(
                {
                    "observation": observation,
                    "action": out["actions"][i].astype(np.float32),
                    "is_first": i == 0,
                    "is_last": i == (len(out["actions"]) - 1),
                    "language_instruction": instruction,
                }
            )

        episode_metadata["file_path"] = path
        episode_metadata["has_language"] = bool(instruction)

        # create output data sample
        sample = {"steps": episode, "episode_metadata": episode_metadata}

        # use episode path as key
        return path, sample

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        # each path is a directory that contains dated directories
        paths = glob.glob(os.path.join(dl_manager.manual_dir, *("*" * (DEPTH - 1))))

        train_inputs, val_inputs = [], []

        for path in paths:
            for dated_folder in os.listdir(path):
                # a mystery left by the greats of the past
                if "lmdb" in dated_folder:
                    continue

                search_path = os.path.join(
                    path, dated_folder, "raw", "traj_group*", "traj*"
                )
                all_traj = glob.glob(search_path)
                if not all_traj:
                    print(f"no trajs found in {search_path}")
                    continue

                config_path = os.path.join(path, dated_folder, "config.json")
                if os.path.exists(config_path):
                    with open(config_path, "rb") as f:
                        config = json.load(f)
                    camera_topics = config["agent"]["env"][1]["camera_topics"]
                else:
                    # assumed camera topics if no config.json exists
                    camera_topics = [
                        "/D435/color/image_raw",
                        "/blue/image_raw",
                        "/yellow/image_raw",
                        "/wrist/image_raw",
                    ]
                all_inputs = [(t, camera_topics) for t in all_traj]

                train_inputs += all_inputs[: int(len(all_inputs) * TRAIN_PROPORTION)]
                val_inputs += all_inputs[int(len(all_inputs) * TRAIN_PROPORTION) :]

        logging.info(
            "Converting %d training and %d validation files.",
            len(train_inputs),
            len(val_inputs),
        )
        return {
            "train": iter(train_inputs),
            "val": iter(val_inputs),
        }
