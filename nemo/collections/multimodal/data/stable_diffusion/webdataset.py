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
import os
import pickle
import random
import re

import boto3
import torch.distributed as dist
from botocore.config import Config
from PIL import Image
from torch.utils.data import IterableDataset
from webdataset.utils import pytorch_worker_info

from nemo.collections.multimodal.data.stable_diffusion.augmentation.augmentations import (
    construct_image_augmentations,
    identical_transform,
)
from nemo.collections.multimodal.data.stable_diffusion.webdataset_utils import WebDataset
from nemo.core.classes import IterableDataset as NeMoIterableDataset

Image.MAX_IMAGE_PIXELS = 933120000
_IMG_EXTENSIONS = "jpg jpeg png ppm pgm pbm pnm".split()
from webdataset import warn_and_continue


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

    with io.BytesIO(data) as stream:
        img = Image.open(stream)
        img.load()
        img = img.convert("RGB")

    return img


def get_world_size():
    r"""Get world size. How many GPUs are available in this job."""
    world_size = 1
    if dist.is_available():
        if dist.is_initialized():
            world_size = dist.get_world_size()
    return world_size


def repeat_list(x, n):
    r"""
    Function to repeat the list to a fixed shape.
    n is the desired length of the extended list.
    Args:
        x (list): Input list
        n (int): Desired length
    """
    if n == 0:
        return []
    assert len(x) > 0

    x_extended = []
    while len(x_extended) < n:
        x_extended = x_extended + x
    x_extended = x_extended[0:n]

    return x_extended


def build_resolution_filter(value=None, method='larger', image_idx=0):
    assert method == 'larger' or method == 'smaller'
    if method == 'larger':
        print(f'Only Selecting images with resolution >= {value}')
        return lambda x: x[image_idx].size[0] >= value and x[image_idx].size[1] >= value

    print(f'Only Selecting images with resolution <= {value}')
    return lambda x: x[image_idx].size[0] <= value and x[image_idx].size[1] <= value


class ShardListWithResumes(IterableDataset):
    r"""
    An iterable dataset that is compatible with custom resets.
    Can be restored from an iteration number and index number.
    """

    def __init__(
        self,
        urls,
        epoch_shuffle=False,
        shuffle=True,
        split_by_node=True,
        split_by_worker=True,
        chunk_size=1,
        resume_flag=True,
        verbose=False,
    ):
        r"""Create a ShardList.
        Args:
            urls (list): a list of URLs as a Python list or brace notation string
            epoch_shuffle (bool): Shuffles the whole epoch. If disabled, each node will see the same set of urls.
            shuffle (bool): shuffle samples before iterating.
            split_by_node (bool): split shards by node if True
            chunk_size (int): chunk size used in webdataset creation
            resume_flag (bool): If enabled, resumes from a specific iteration and epoch number.
            verbose (bool): Prints some logs if true
        """
        super().__init__()

        self.verbose = verbose
        self.epoch = 0
        self.start_index = 0
        self.epoch_shuffle = epoch_shuffle
        self.shuffle = shuffle
        self.split_by_node = split_by_node
        self.split_by_worker = split_by_worker
        self.chunk_size = chunk_size
        self.resume_flag = resume_flag
        self.urls = urls

    def set_epoch(self, epoch, start_index):
        r"""Set the current epoch. Used for per-node shuffling.
        Args:
            epoch (int): Epoch number
            start_index (int): iteraton number
        """
        self.epoch = epoch
        self.start_index = start_index

    def __iter__(self):
        r"""Return an iterator over the shards."""

        rank, world_size, worker_id, num_workers = pytorch_worker_info()

        # Setting epoch and start index
        if self.resume_flag:
            self.epoch = int(os.environ['WDS_EPOCH_NUM'])

            # This tells us number of chunks that have been seen by one GPU
            self.start_index = int(os.environ['WDS_START_INDEX']) // self.chunk_size

        urls = self.urls

        # Shuffling the entire epoch before splitting among nodes and workers.
        if self.epoch_shuffle:
            if self.shuffle:
                raise ValueError("If epoch_shuffle is used, do not use shuffle.")

            if self.verbose:
                print("PytorchShardList epochshuffle {}".format(self.epoch))
            random.Random(self.epoch).shuffle(urls)

        num_urls = len(urls)

        # Splitting the shards by worker and node

        # Extending urls so that each workers receive the same number of batches.
        # This serves the job of ddp_equalize.
        nworkers_all = world_size * num_workers
        if num_urls % nworkers_all > 0:
            num_urls_per_process = (num_urls // nworkers_all) + 1
        else:
            num_urls_per_process = num_urls // nworkers_all
        extended_url_list_size = num_urls_per_process * nworkers_all
        urls = repeat_list(urls, extended_url_list_size)

        # print(f'Total Number of URLS before spliting: {num_urls}')
        if self.split_by_node:
            urls = urls[rank::world_size]

        if self.split_by_worker:
            urls = urls[worker_id::num_workers]

        if self.verbose:
            print(
                f'Number of URLs after splitting: {len(urls)}. rank/world_size={rank}/{world_size} worker_id/num_workers={worker_id}/{num_workers}'
            )

        if self.shuffle:
            random.Random(self.epoch + 17).shuffle(urls)

        # This tells us the number of chunks seen by one worker.
        # Do not iterate over the seen chunks.
        start_index_per_worker = self.start_index // (num_workers * world_size)
        urls = urls[start_index_per_worker:]

        if self.verbose:
            print(
                f'Number of URLS after using start_index_per_worker: {len(urls)}. self.start_index={self.start_index} start_index_per_worker={start_index_per_worker}'
            )
            print(
                f'PytorchShardList Rank=<{rank}/{world_size}> Worker=<{worker_id}/{num_workers}> receives {len(urls)} URLs (TARs)'
            )

        for url in urls:
            yield dict(url=url)


class WebDatasetBase(NeMoIterableDataset):
    def __init__(self, cfg, is_train=True):
        r"""
        Webdataloader class
        Args:
            cfg: Dataset Config
            is_train (bool): Is the dataset used in training mode?
        """
        super().__init__()

        self.cfg = cfg
        self.num_workers = self.cfg.num_workers
        self.world_size = get_world_size()
        self.webdata_cfg = self.cfg.webdataset
        self.infinite_sampler = self.webdata_cfg.infinite_sampler
        if is_train:
            dataset_path = cfg.train.dataset_path
            self.batch_size = self.cfg.train.batch_size
            self.augmentations = self.cfg.train.augmentations
            self.filterings = self.cfg.train.filterings
        else:
            dataset_path = cfg.val.dataset_path
            self.batch_size = self.val.batch_size
            self.augmentations = self.cfg.val.augmentations
            self.filterings = self.cfg.val.filterings

        if getattr(self.webdata_cfg, 'object_store', False):
            # Initializing PBSS
            print(f'Init PBSS using credentials file at {self.webdata_cfg.pbss_credentials_file}')
            self.use_object_store = True
            assert self.webdata_cfg.pbss_credentials_file is not None
            with open(self.webdata_cfg.pbss_credentials_file) as fin:
                self.credentials = json.load(fin)
            config = Config(connect_timeout=30, signature_version="s3", retries={"max_attempts": 999999})
            self.s3 = boto3.client('s3', **self.credentials, config=config)
            self.bucket = self.webdata_cfg.bucket
            self.local_root_path = None
        else:
            self.use_object_store = False
            self.s3 = None
            self.bucket = None
            self.local_root_path = self.webdata_cfg.local_root_path
            print(f'Read Webdataset locally. Data stores at {self.local_root_path}')

        # Concatenate all dataset infos

        # wdinfo in a dict containing webdata information
        self.wdinfo = dict()
        for dset_info_path in dataset_path:
            with open(dset_info_path, 'rb') as fp:
                dset_info = pickle.load(fp)
                if 'tar_files' not in self.wdinfo:
                    self.wdinfo['tar_files'] = dset_info['tar_files']
                    self.wdinfo['total_key_count'] = dset_info['total_key_count']
                    self.wdinfo['chunk_size'] = dset_info['chunk_size']
                else:
                    self.wdinfo['tar_files'].extend(dset_info['tar_files'])
                    self.wdinfo['total_key_count'] += dset_info['total_key_count']

    def build_dataset(self, **kwargs):
        raise ValueError('build_dataset function not implemented')


class WebDatasetWithRawText(WebDatasetBase):
    def __init__(self, dataset_cfg, is_train=True):
        r"""
        Webdataloader class
        Args:
            dataset_cfg: Dataset config
            is_train (bool): Is the dataset used in training mode?
        """
        super().__init__(dataset_cfg, is_train=is_train)
        # For adding corruptions and obtaining image pyramid
        # TODO Add this for SR256/SR1024 training
        # self.corruption_gen = ImagePyramidWithCorruptions(
        #     cfg=cfg, is_inference=is_inference, is_test=is_test
        # )

        # Construct augmentations
        self.img_transform = construct_image_augmentations(self.augmentations)
        self.text_transform = identical_transform
        self.verbose = dataset_cfg.get("verbose", False)
        self.build_dataset()

    def build_dataset(self):
        """See base class."""

        train_info = self.wdinfo

        shards_train_list = train_info["tar_files"]
        num_shards = len(shards_train_list)
        assert num_shards > 0, "Did not find any training data."

        chunk_size = train_info["chunk_size"]

        # Shuffle buffer:
        shuffle_buffer_size = train_info["chunk_size"]

        # This function maps data that are tuples to dictionary.
        def tuple_to_dict(inp):
            for input in inp:
                out_dict = dict()
                out_dict['images'] = input[0].permute(1, 2, 0)

                out_dict['captions'] = input[1]
                yield out_dict

        # Train dataset object
        from webdataset import warn_and_continue

        if self.infinite_sampler:
            rank, world_size, worker_id, num_workers = pytorch_worker_info()
            epoch_length = train_info["total_key_count"] // self.batch_size // world_size
            print(f'Using infinite sampler, world_size={world_size}. The epoch length will be set to: {epoch_length}')
        else:
            print(f'Initiating ShardListWithResumes..')
            shards_train_list = ShardListWithResumes(
                urls=shards_train_list,
                epoch_shuffle=True,
                shuffle=False,
                split_by_node=True,
                split_by_worker=True,
                chunk_size=chunk_size,
                resume_flag=True,
                verbose=self.verbose,
            )

        train_dataset = (
            WebDataset(
                shards_train_list,
                load_from_object_store=self.use_object_store,
                s3_client=self.s3,
                s3_bucket_name=self.bucket,
                local_root_path=self.local_root_path,
                handler=warn_and_continue,
                resampled=self.infinite_sampler,
            )
            .shuffle(shuffle_buffer_size)  # Shuffling the buffer
            .decode(pil_loader, handler=warn_and_continue)  # Decoding the data
            .to_tuple("jpg txt")  # Splitting into tuple
        )
        if self.filterings is not None:
            if self.filterings.resolution is not None:
                train_dataset = train_dataset.select(
                    build_resolution_filter(**self.filterings.resolution, image_idx=0)
                )

        # Add additional augmentation
        train_dataset = train_dataset.map_tuple(self.img_transform, self.text_transform).compose(  # Augmentation
            tuple_to_dict
        )  # Converting tuple to data dict

        train_dataset.total_images = train_info["total_key_count"]
        # Set epoch length if using infinite sampler
        if self.infinite_sampler:
            rank, world_size, worker_id, num_workers = pytorch_worker_info()
            nbatches = train_dataset.total_images // world_size // self.num_workers
            print(f'Setting nbatches={nbatches} for infinite sampler. world_size={world_size}')
            train_dataset = train_dataset.with_epoch(nbatches=nbatches)

        print("Total number of training shards: %d", num_shards)
        print("Total training key count: %d", train_dataset.total_images)

        self._dataset = train_dataset

    def __iter__(self):
        return self._dataset.__iter__()

    def __len__(self):
        world_size = get_world_size()
        # In Webdataset multi-gpu training settings, each GPU will be assigned with different portions of
        # training data, therefore divde the dataset size by the number of GPUs.
        return self._dataset.total_images // world_size
