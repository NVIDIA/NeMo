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

from configparser import _section
import os
import time
import pickle
import tokenize

from typing import Optional
from dataclasses import dataclass

import torch
import numpy as np
import multiprocessing as mp
from functools import partial

from nemo.core import Dataset
from nemo.utils import logging



__all__ = ['TextMemMapDatasetConfig', 'TextMemMapDataset', 'build_index_files']

@dataclass
class TextMemMapDatasetConfig():
    dataset_path: str = ''
    dataset_files: str = 'data.csv'
    dataset_type: str = 'zinc_csv'
    newline_int: int = 10
    header_lines: int = 1
    skip_lines: int = 0
    data_col: int = 1
    data_sep: str = ','

class TextMemMapDataset(Dataset):
    """
    Allow per-line lazy access to multiple text files using numpy memmap.
    """
    def __init__(self,
                 dataset_paths,
                 newline_int=10,
                 header_lines=1,
                 skip_lines=0,
                 workers=None,
                 tokenizer=None,
        ):
        super().__init__()

        if len(dataset_paths) < 1:
            raise ValueError("files_list must contain at leat one file name")


        self._newline_int = newline_int
        # skip first N lines
        self._header_lines = header_lines
        self._skip_lines = skip_lines
        self._files_list = dataset_paths
        self._worker = workers
        self.tokenizer = tokenizer

        logging.info(f"Building data files")
        # load all files into memmap
        start_time = time.time()
        is_ditributed = torch.distributed.is_available() and \
            torch.distributed.is_initialized()

        if  not is_ditributed or \
            (is_ditributed and torch.distributed.get_rank() == 0):
            build_index_files(dataset_paths, newline_int, workers=self._worker)

        if is_ditributed:
            torch.distributed.barrier()

        logging.info(f"Loading data files")
        mdata_midx_size_list = [self.load_file(fn) for fn in self._files_list]

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
        Return a string from binary memmap
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
        text = self.mdata_midx_size_list[file_id][0][rec_start:rec_end].tobytes().decode("ascii")
        
        # parse raw text (e.g., tokenize)
        data = self._build_data_from_text(text)

        return data

    def _build_data_from_text(self, text):
        """Allows child-classes to modify the parsing of raw text, prior to tokenization"""
        # tokenize text if tokenizer is given
        if self.tokenizer is not None:
            data = self.tokenizer.text_to_ids(text)
        else:
            data = text
        
        return data

    def load_file(self, fn):
        """
        Loads a text file as np.int8.

        Returns:
            mdata - memorymap of np.int8
            midx - indices pointing to the end-of-line (or end of file) position
            size - number of lines in file
        """
        logging.info(f"Loading {fn}")
        idx_fn = fn + ".idx"

        # create data map
        mdata = np.memmap(fn, dtype=np.uint8, mode='r')

        if os.path.exists(idx_fn):
            idx_dict = pickle.load(open(idx_fn, 'rb'))
            midx = idx_dict['midx']
            size = idx_dict['size']
            # test for mismatch in expected newline_int
            if 'newline_int' in idx_dict:
                newline_int = idx_dict['newline_int']
                if self._newline_int != newline_int:
                    logger.warning(f"Mismatch in newline_int, expected = {self._newline_int} but loaded {newline_int}")
        else:
            raise ValueError(f'Memory Map for {fn} is not found')

        return (mdata, midx, size)    

class CSVMemMapDataset(TextMemMapDataset):
    """
    Allow per-line lazy access to multiple text files using numpy memmap.
    """
    def __init__(self,
                 dataset_paths,
                 newline_int=10,
                 header_lines=1,
                 skip_lines=0,
                 workers=None,
                 data_col=1,
                 data_sep=','):
        super().__init__(
                 dataset_paths=dataset_paths,
                 newline_int=newline_int,
                 header_lines=header_lines,
                 skip_lines=skip_lines,
                 workers=workers,            
        )
        self._data_col = data_col
        self._data_sep = data_sep


    def _build_data_from_text(self, text):
        """Return a CSV field from text"""
        # get CSV field
        text = text.split(self._data_sep)[self._data_col]
        # tokenize
        return super()._build_data_from_text(text)        


def _build_memmap_index_files(newline_int, fn):
    """Helper function to build an index .idx file"""
    idx_fn = fn + ".idx"

    # create data map
    mdata = np.memmap(fn, dtype=np.uint8, mode='r')
    if os.path.exists(idx_fn):
        return None
    else:
        logging.info(f"Building idx file = {idx_fn}")
        midx = np.where(mdata == newline_int)[0]
        # add last item in case there is no new-line
        if (len(midx) == 0) or (midx[-1]+1 != len(mdata)):
            midx = np.asarray(midx.tolist() + [len(midx)], dtype=midx.dtype)

        size = len(mdata)
        data = dict(midx=midx, size=size, newline_int=newline_int)
        pickle.dump(data, open(idx_fn, "wb"))
        mdata._mmap.close()
        del mdata

        return True


def build_index_files(dataset_paths,
                      newline_int,
                      workers=None):
    """Auxiliary method to build multiple index .idx files"""
    if len(dataset_paths) < 1:
        raise ValueError("files_list must contain at leat one file name")

    if workers is None:
        workers = min(1, os.cpu_count() // 2)

    logging.info(f"Building data files using {workers} workers")
    # load all files into memmap
    start_time = time.time()
    with mp.Pool(workers) as p:
        build_status = p.map(partial(_build_memmap_index_files, newline_int),
                                     dataset_paths)

    logging.info(f'Time building {sum(build_status)} mem-mapped file: {time.time() - start_time}')
