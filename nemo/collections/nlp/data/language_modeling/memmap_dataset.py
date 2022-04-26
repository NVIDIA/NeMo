# Copyright (c) 2022, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

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

import os
import time
import pickle

from typing import Optional
from dataclasses import dataclass

import torch
import numpy as np

from nemo.collections.nlp.data.language_modeling.megatron.megatron_dataset import MegatronDataset
from nemo.utils import logging
from ..utils.data_index import build_index_files

__all__ = ['MoleculeCsvDatasetConfig', 'MoleculeCsvDataset']


@dataclass
class MoleculeCsvDatasetConfig():
    dataset_path: str = ''
    dataset_files: str = 'data.csv'
    dataset_type: str = 'zinc_csv'
    newline_int: int = 10
    header_lines: int = 1
    skip_lines: int = 0
    data_col: int = 1
    data_sep: str = ','
    micro_batch_size: int = 1
    use_iterable: bool = False
    map_data: bool = False
    encoder_augment: bool = True
    encoder_mask: bool = False
    decoder_augment: bool = False
    canonicalize_input: bool = True # TODO remove when CSV data processing updated
    drop_last: bool = False
    shuffle: bool = False
    num_workers: Optional[int] = None
    pin_memory: bool = True # TODO: remove this if value is fixed
    dataloader_type: str = 'single'


class MoleculeCsvDataset(MegatronDataset):
    """
    Allow per-line lazy access to multiple text files using numpy memmap.
    """
    def __init__(self,
                 dataset_paths,
                 cfg=None,
                 trainer=None,
                 newline_int=10,
                 header_lines=1,
                 skip_lines=0,
                 workers=None,
                 data_col=1,
                 data_sep=','):
        if len(dataset_paths) < 1:
            raise ValueError("files_list must contain at leat one file name")

        super().__init__(cfg, trainer)

        # load values from cfg
        if cfg is not None:
            newline_int = cfg.get('newline_int', 10)
            header_lines = cfg.get('header_lines', 1)
            skip_lines = cfg.get('skip_lines', 0)
            data_col = cfg.get('data_col', 1)
            data_sep = cfg.get('data_sep', ',')

        self.newline_int = newline_int
        # skip first N lines
        self._header_lines = header_lines
        self._skip_lines = skip_lines
        self.files_list = dataset_paths
        self._data_col = data_col
        self._data_sep = data_sep
        self.mdata_midx_size_list = None
        self.worker = workers

        logging.info(f"Building data files")
        # load all files into memmap
        start_time = time.time()
        is_ditributed = torch.distributed.is_available() and \
            torch.distributed.is_initialized()

        if  not is_ditributed or \
            (is_ditributed and torch.distributed.get_rank() == 0):
            build_index_files(dataset_paths, newline_int, workers=self.worker)

        if is_ditributed:
            torch.distributed.barrier()
        logging.info(f'Time building mem-mapped file: {time.time() - start_time}')

        logging.info(f"Loading data files")
        mdata_midx_size_list = [self.load_file(fn) for fn in self.files_list]

        logging.info("Computing global indices")
        joint_midx = [0]
        for i in range(len(mdata_midx_size_list)):
            midx = mdata_midx_size_list[i][1]
            joint_midx.append(joint_midx[-1] + (len(midx) - header_lines))

        self.joint_midx = joint_midx
        self.mdata_midx_size_list = mdata_midx_size_list

    def __del__(self):
        if self.mdata_midx_size_list:
            for mdata, midx, size in self.mdata_midx_size_list:
                mdata._mmap.close()

    def __len__(self):
        return self.joint_midx[-1]

    def __getitem__(self, idx):
        """
        Return a string
        """
        # Identify the file containing the record
        file_id = 0
        for end_idx in self.joint_midx[1:]:
            if idx < end_idx:
                break
            file_id += 1
        file_row = idx - self.joint_midx[file_id]

        rec_start = self.mdata_midx_size_list[file_id][1][file_row]
        rec_end = self.mdata_midx_size_list[file_id][1][file_row + 1 + self._skip_lines]
        data = self.mdata_midx_size_list[file_id][0][rec_start:rec_end].tobytes().decode("ascii")
        return data.split(self._data_sep)[self._data_col]

    def load_file(self, fn):
        """
        Loads a text file as np.int8.

        Returns:
            mdata - memorymap of np.int8
            midx - indices pointing to the end-of-line (or end of file) position
        """
        logging.info(f"Loading {fn}")
        idx_fn = fn + ".idx"

        # create data map
        mdata = np.memmap(fn, dtype=np.uint8, mode='r')

        if os.path.exists(idx_fn):
            idx_dict = pickle.load(open(idx_fn, 'rb'))
            midx = idx_dict['midx']
            size = idx_dict['size']
        else:
            raise ValueError(f'Memory Map for {fn} is not found')

        return (mdata, midx, size)