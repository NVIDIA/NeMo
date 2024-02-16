# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Most of the code here has been copied from:
#   fairseq/fairseq/data/indexed_dataset.py

# with some modifications:

# Removed IndexedRawTextDataset since it relied on Fairseq dictionary
# other slight modifications to remove fairseq dependencies
# Added document index to index file and made it accessible.
#    An empty sentence no longer separates documents.

import os
import shutil
import struct
from functools import lru_cache
from itertools import accumulate

import boto3
import numpy as np
import torch

from nemo.collections.nlp.data.language_modeling.megatron.indexed_retrieval_dataset import (
    MMapRetrievalIndexedDataset,
    MMapRetrievalIndexedDatasetBuilder,
)
from nemo.collections.nlp.data.language_modeling.text_memmap_dataset import CSVMemMapDataset, TextMemMapDataset
from nemo.utils import AppState, logging
from nemo.utils.s3 import is_s3_path, object_exists, parse_s3_path


def __best_fitting_dtype(vocab_size=None):
    if vocab_size is not None and vocab_size < 65500:
        return np.uint16
    else:
        return np.int32


def get_available_dataset_impl():
    return ['lazy', 'cached', 'mmap', "retmmap"]


def infer_dataset_impl(path):
    if IndexedDataset.exists(path):
        with open(index_file_path(path), 'rb') as f:
            magic = f.read(8)
            if magic == IndexedDataset._HDR_MAGIC:
                return 'cached'
            elif magic == MMapIndexedDataset.Index._HDR_MAGIC[:8]:
                return 'mmap'
            elif magic == MMapRetrievalIndexedDataset.Index._HDR_MAGIC[:8]:
                return 'retmmap'
            else:
                return None
    else:
        print(f"Dataset does not exist: {path}")
        print("Path should be a basename that both .idx and .bin can be appended to get full filenames.")
        return None


def make_builder(out_file, impl, vocab_size=None, chunk_size=64, pad_id=0, retrieval_db=False, stride=64):
    if impl == 'mmap':
        return MMapIndexedDatasetBuilder(out_file, dtype=__best_fitting_dtype(vocab_size))
    elif impl == 'retmmap':
        return MMapRetrievalIndexedDatasetBuilder(
            out_file,
            chunk_size=chunk_size,
            pad_id=pad_id,
            retrieval_db=retrieval_db,
            dtype=__best_fitting_dtype(vocab_size),
            stride=stride,
        )
    else:
        return IndexedDatasetBuilder(out_file)


def make_indexed_dataset_compatibility(ds):
    """Make any dataset compatible with IndexedDataset for Megatron samples mapping."""
    if (getattr(ds, 'doc_idx', None) is not None) or (getattr(ds, 'sizes', None) is not None):
        raise AttributeError("Dataset already has doc_idx or sizes attributes.")

    ds.doc_idx = np.arange(len(ds) + 1, dtype=np.int64)
    ds.sizes = np.ones(len(ds), dtype=np.int32)

    return ds


def deallocate_indexed_dataset_memory(indexed_dataset):
    """Deallocate memory of an IndexedDataset."""
    if isinstance(indexed_dataset, MMapIndexedDataset):
        # for MMapIndexedDataset we cannot release any memory of sizes
        indexed_dataset._index._doc_idx = None
    elif isinstance(indexed_dataset, S3IndexedDataset):
        # for S3IndexedDataset we cannot release any memory of sizes
        indexed_dataset._index._doc_idx = None
    else:
        indexed_dataset.sizes = None
        indexed_dataset.doc_idx = None


def make_dataset(
    path, impl, skip_warmup=False, impl_kwargs={}, delay_data_mmap=False, index_cache_dir=None, data_cache_nbytes=None
):
    # first handle text memap
    if impl == 'text_mmap':
        return TextMemMapDataset(path, **impl_kwargs)
    elif impl == 'csv_mmap':
        return CSVMemMapDataset(path, **impl_kwargs)

    # now handle bin memap
    if (not IndexedDataset.exists(path)) and (not S3IndexedDataset.exists(path)):
        print(f"Dataset does not exist: {path}")
        print("Path should be a basename that both .idx and .bin can be appended to get full filenames.")
        return None
    if impl == 'infer':
        impl = infer_dataset_impl(path)
    if impl == 'lazy' and IndexedDataset.exists(path):
        return IndexedDataset(path)
    elif impl == 'cached' and IndexedDataset.exists(path):
        return IndexedCachedDataset(path)
    elif impl == 'mmap' and (not is_s3_path(path)) and MMapIndexedDataset.exists(path):
        return MMapIndexedDataset(path, skip_warmup, delay_data_mmap)
    elif impl == 'mmap' and is_s3_path(path) and S3IndexedDataset.exists(path):
        assert skip_warmup, "S3IndexedDataset only supports skip_warmup == True"
        assert not delay_data_mmap, "S3IndexedDataset only supports delay_data_mmap == False"
        return S3IndexedDataset(path, index_cache_dir, data_cache_nbytes)
    elif impl == 'retmmap':
        return MMapRetrievalIndexedDataset(path, skip_warmup)
    raise ValueError(f"Unknown dataset implementation: {impl}")


def dataset_exists(path, impl):
    if impl == 'mmap':
        return MMapIndexedDataset.exists(path)
    elif impl == 'retmmap':
        return MMapRetrievalIndexedDataset.exists(path)
    else:
        return IndexedDataset.exists(path)


def read_longs(f, n):
    a = np.empty(n, dtype=np.int64)
    f.readinto(a)
    return a


def write_longs(f, a):
    f.write(np.array(a, dtype=np.int64))


dtypes = {1: np.uint8, 2: np.int8, 3: np.int16, 4: np.int32, 5: np.int64, 6: np.float64, 7: np.double, 8: np.uint16}


def code(dtype):
    for k in dtypes.keys():
        if dtypes[k] == dtype:
            return k
    raise ValueError(dtype)


def index_file_path(prefix_path):
    return prefix_path + '.idx'


def local_index_file_path(prefix_path, index_cache_dir):
    return os.path.join(index_cache_dir, os.path.basename(index_file_path(prefix_path)))


def data_file_path(prefix_path):
    return prefix_path + '.bin'


def create_doc_idx(sizes):
    doc_idx = [0]
    for i, s in enumerate(sizes):
        if s == 0:
            doc_idx.append(i + 1)
    return doc_idx


class IndexedDataset(torch.utils.data.Dataset):
    """Loader for IndexedDataset"""

    _HDR_MAGIC = b'TNTIDX\x00\x00'

    def __init__(self, path):
        super().__init__()
        self.path = path
        self.data_file = None
        self.read_index(path)

    def read_index(self, path):
        with open(index_file_path(path), 'rb') as f:
            magic = f.read(8)
            assert magic == self._HDR_MAGIC, (
                'Index file doesn\'t match expected format. ' 'Make sure that --dataset-impl is configured properly.'
            )
            version = f.read(8)
            assert struct.unpack('<Q', version) == (1,)
            code, self.element_size = struct.unpack('<QQ', f.read(16))
            self.dtype = dtypes[code]
            self._len, self.s = struct.unpack('<QQ', f.read(16))
            self.doc_count = struct.unpack('<Q', f.read(8))
            self.dim_offsets = read_longs(f, self._len + 1)
            self.data_offsets = read_longs(f, self._len + 1)
            self.sizes = read_longs(f, self.s)
            self.doc_idx = read_longs(f, self.doc_count)

    def read_data(self, path):
        self.data_file = open(data_file_path(path), 'rb', buffering=0)

    def check_index(self, i):
        if i < 0 or i >= self._len:
            raise IndexError('index out of range')

    def __del__(self):
        if self.data_file:
            self.data_file.close()

    # @lru_cache(maxsize=8)
    def __getitem__(self, idx):
        if not self.data_file:
            self.read_data(self.path)
        if isinstance(idx, int):
            i = idx
            self.check_index(i)
            tensor_size = self.sizes[self.dim_offsets[i] : self.dim_offsets[i + 1]]
            a = np.empty(tensor_size, dtype=self.dtype)
            self.data_file.seek(self.data_offsets[i] * self.element_size)
            self.data_file.readinto(a)
            return a
        elif isinstance(idx, slice):
            start, stop, step = idx.indices(len(self))
            if step != 1:
                raise ValueError("Slices into indexed_dataset must be contiguous")
            sizes = self.sizes[self.dim_offsets[start] : self.dim_offsets[stop]]
            size = sum(sizes)
            a = np.empty(size, dtype=self.dtype)
            self.data_file.seek(self.data_offsets[start] * self.element_size)
            self.data_file.readinto(a)
            offsets = list(accumulate(sizes))
            sents = np.split(a, offsets[:-1])
            return sents

    def __len__(self):
        return self._len

    def num_tokens(self, index):
        return self.sizes[index]

    def size(self, index):
        return self.sizes[index]

    @staticmethod
    def exists(path):
        return os.path.exists(index_file_path(path)) and os.path.exists(data_file_path(path))

    @property
    def supports_prefetch(self):
        return False  # avoid prefetching to save memory


class IndexedCachedDataset(IndexedDataset):
    def __init__(self, path):
        super().__init__(path)
        self.cache = None
        self.cache_index = {}

    @property
    def supports_prefetch(self):
        return True

    def prefetch(self, indices):
        if all(i in self.cache_index for i in indices):
            return
        if not self.data_file:
            self.read_data(self.path)
        indices = sorted(set(indices))
        total_size = 0
        for i in indices:
            total_size += self.data_offsets[i + 1] - self.data_offsets[i]
        self.cache = np.empty(total_size, dtype=self.dtype)
        ptx = 0
        self.cache_index.clear()
        for i in indices:
            self.cache_index[i] = ptx
            size = self.data_offsets[i + 1] - self.data_offsets[i]
            a = self.cache[ptx : ptx + size]
            self.data_file.seek(self.data_offsets[i] * self.element_size)
            self.data_file.readinto(a)
            ptx += size
        if self.data_file:
            # close and delete data file after prefetch so we can pickle
            self.data_file.close()
            self.data_file = None

    # @lru_cache(maxsize=8)
    def __getitem__(self, idx):
        if isinstance(idx, int):
            i = idx
            self.check_index(i)
            tensor_size = self.sizes[self.dim_offsets[i] : self.dim_offsets[i + 1]]
            a = np.empty(tensor_size, dtype=self.dtype)
            ptx = self.cache_index[i]
            np.copyto(a, self.cache[ptx : ptx + a.size])
            return a
        elif isinstance(idx, slice):
            # Hack just to make this work, can optimizer later if necessary
            sents = []
            for i in range(*idx.indices(len(self))):
                sents.append(self[i])
            return sents


class IndexedDatasetBuilder(object):
    element_sizes = {np.uint8: 1, np.int8: 1, np.int16: 2, np.int32: 4, np.int64: 8, np.float64: 4, np.double: 8}

    def __init__(self, out_file, dtype=np.int32):
        self.out_file = open(out_file, 'wb')
        self.dtype = dtype
        self.data_offsets = [0]
        self.dim_offsets = [0]
        self.sizes = []
        self.element_size = self.element_sizes[self.dtype]
        self.doc_idx = [0]

    def add_item(self, tensor):
        bytes = self.out_file.write(np.array(tensor.numpy(), dtype=self.dtype))
        self.data_offsets.append(self.data_offsets[-1] + bytes / self.element_size)
        for s in tensor.size():
            self.sizes.append(s)
        self.dim_offsets.append(self.dim_offsets[-1] + len(tensor.size()))

    def end_document(self):
        self.doc_idx.append(len(self.sizes))

    def merge_file_(self, another_file):
        index = IndexedDataset(another_file)
        assert index.dtype == self.dtype

        begin = self.data_offsets[-1]
        for offset in index.data_offsets[1:]:
            self.data_offsets.append(begin + offset)
        self.sizes.extend(index.sizes)
        begin = self.dim_offsets[-1]
        for dim_offset in index.dim_offsets[1:]:
            self.dim_offsets.append(begin + dim_offset)

        with open(data_file_path(another_file), 'rb') as f:
            while True:
                data = f.read(1024)
                if data:
                    self.out_file.write(data)
                else:
                    break

    def finalize(self, index_file):
        self.out_file.close()
        index = open(index_file, 'wb')
        index.write(b'TNTIDX\x00\x00')
        index.write(struct.pack('<Q', 1))
        index.write(struct.pack('<QQ', code(self.dtype), self.element_size))
        index.write(struct.pack('<QQ', len(self.data_offsets) - 1, len(self.sizes)))
        index.write(struct.pack('<Q', len(self.doc_idx)))
        write_longs(index, self.dim_offsets)
        write_longs(index, self.data_offsets)
        write_longs(index, self.sizes)
        write_longs(index, self.doc_idx)
        index.close()


def _warmup_mmap_file(path):
    with open(path, 'rb') as stream:
        while stream.read(100 * 1024 * 1024):
            pass


class MMapIndexedDataset(torch.utils.data.Dataset):
    class Index(object):
        _HDR_MAGIC = b'MMIDIDX\x00\x00'

        @classmethod
        def writer(cls, path, dtype):
            class _Writer(object):
                def __enter__(self):
                    self._file = open(path, 'wb')

                    self._file.write(cls._HDR_MAGIC)
                    self._file.write(struct.pack('<Q', 1))
                    self._file.write(struct.pack('<B', code(dtype)))

                    return self

                @staticmethod
                def _get_pointers(sizes):
                    dtype_size = dtype().itemsize
                    address = 0
                    pointers = []

                    for size in sizes:
                        pointers.append(address)
                        address += size * dtype_size

                    return pointers

                def write(self, sizes, doc_idx):
                    pointers = self._get_pointers(sizes)

                    self._file.write(struct.pack('<Q', len(sizes)))
                    self._file.write(struct.pack('<Q', len(doc_idx)))

                    sizes = np.array(sizes, dtype=np.int32)
                    self._file.write(sizes.tobytes(order='C'))
                    del sizes

                    pointers = np.array(pointers, dtype=np.int64)
                    self._file.write(pointers.tobytes(order='C'))
                    del pointers

                    doc_idx = np.array(doc_idx, dtype=np.int64)
                    self._file.write(doc_idx.tobytes(order='C'))

                def __exit__(self, exc_type, exc_val, exc_tb):
                    self._file.close()

            return _Writer()

        def __init__(self, path, skip_warmup=False):
            with open(path, 'rb') as stream:
                magic_test = stream.read(9)
                assert self._HDR_MAGIC == magic_test, (
                    'Index file doesn\'t match expected format. '
                    'Make sure that --dataset-impl is configured properly.'
                )
                version = struct.unpack('<Q', stream.read(8))
                assert (1,) == version

                (dtype_code,) = struct.unpack('<B', stream.read(1))
                self._dtype = dtypes[dtype_code]
                self._dtype_size = self._dtype().itemsize

                self._len = struct.unpack('<Q', stream.read(8))[0]
                self._doc_count = struct.unpack('<Q', stream.read(8))[0]
                offset = stream.tell()

            if not skip_warmup:
                logging.info("    warming up index mmap file...")
                _warmup_mmap_file(path)

            self._bin_buffer_mmap = np.memmap(path, mode='r', order='C')
            self._bin_buffer = memoryview(self._bin_buffer_mmap)
            logging.info("    reading sizes...")
            self._sizes = np.frombuffer(self._bin_buffer, dtype=np.int32, count=self._len, offset=offset)
            logging.info("    reading pointers...")
            self._pointers = np.frombuffer(
                self._bin_buffer, dtype=np.int64, count=self._len, offset=offset + self._sizes.nbytes
            )
            logging.info("    reading document index...")
            self._doc_idx = np.frombuffer(
                self._bin_buffer,
                dtype=np.int64,
                count=self._doc_count,
                offset=offset + self._sizes.nbytes + self._pointers.nbytes,
            )

        def __del__(self):
            self._bin_buffer_mmap._mmap.close()
            del self._bin_buffer_mmap

        @property
        def dtype(self):
            return self._dtype

        @property
        def sizes(self):
            return self._sizes

        @property
        def doc_idx(self):
            return self._doc_idx

        @lru_cache(maxsize=8)
        def __getitem__(self, i):
            return self._pointers[i], self._sizes[i]

        def __len__(self):
            return self._len

    def __init__(self, path, skip_warmup=False, delay_data_mmap=False):
        super().__init__()

        self._path = None
        self._index = None
        self._bin_buffer = None
        self._delay_data_mmap = delay_data_mmap
        self._skip_warmup = skip_warmup

        self._do_init(path, skip_warmup, delay_data_mmap)

    def __getstate__(self):
        return self._path

    def __setstate__(self, state):
        self._do_init(state)

    def _do_init(self, path, skip_warmup=True, delay_data_mmap=False):
        self._path = path
        self._index = self.Index(index_file_path(self._path), skip_warmup)

        if not delay_data_mmap:
            self._create_data_mmap(skip_warmup)
        else:
            logging.info("    skip creating data numpy buffer of mmap...")
            self._bin_buffer_mmap = None
            self._bin_buffer = None

    def _create_data_mmap(self, skip_warmup):
        if not skip_warmup:
            logging.info("    warming up data mmap file...")
            _warmup_mmap_file(data_file_path(self._path))
        logging.info("    creating numpy buffer of mmap...")
        self._bin_buffer_mmap = np.memmap(data_file_path(self._path), mode='r', order='C')
        logging.info("    creating memory view of numpy buffer...")
        self._bin_buffer = memoryview(self._bin_buffer_mmap)

    def __del__(self):
        if self._bin_buffer_mmap is not None:
            self._bin_buffer_mmap._mmap.close()
        del self._bin_buffer_mmap
        del self._index

    def __len__(self):
        return len(self._index)

    # @lru_cache(maxsize=8)
    def __getitem__(self, idx):
        if isinstance(idx, int):
            ptr, size = self._index[idx]
            np_array = np.frombuffer(self._bin_buffer, dtype=self._index.dtype, count=size, offset=ptr)
            return np_array
        elif isinstance(idx, slice):
            start, stop, step = idx.indices(len(self))
            if step != 1:
                raise ValueError("Slices into indexed_dataset must be contiguous")
            ptr = self._index._pointers[start]
            sizes = self._index._sizes[idx]
            offsets = list(accumulate(sizes))
            total_size = sum(sizes)
            np_array = np.frombuffer(self._bin_buffer, dtype=self._index.dtype, count=total_size, offset=ptr)
            sents = np.split(np_array, offsets[:-1])
            return sents

    def get(self, idx, offset=0, length=None):
        """ Retrieves a single item from the dataset with the option to only
        return a portion of the item.

        get(idx) is the same as [idx] but get() does not support slicing.
        """
        ptr, size = self._index[idx]
        if length is None:
            length = size - offset
        ptr += offset * np.dtype(self._index.dtype).itemsize
        np_array = np.frombuffer(self._bin_buffer, dtype=self._index.dtype, count=length, offset=ptr)
        return np_array

    def create_data_mmap(self):
        self._create_data_mmap(self._skip_warmup)

    @property
    def sizes(self):
        return self._index.sizes

    @property
    def doc_idx(self):
        return self._index.doc_idx

    def get_doc_idx(self):
        return self._index._doc_idx

    def set_doc_idx(self, doc_idx_):
        self._index._doc_idx = doc_idx_

    @property
    def supports_prefetch(self):
        return False

    @staticmethod
    def exists(path):
        return os.path.exists(index_file_path(path)) and os.path.exists(data_file_path(path))


class MMapIndexedDatasetBuilder(object):
    def __init__(self, out_file, dtype=np.int64):
        self._data_file = open(out_file, 'wb')
        self._dtype = dtype
        self._sizes = []
        self._doc_idx = [0]

    def add_item(self, tensor):
        np_array = np.array(tensor.numpy(), dtype=self._dtype)
        self._data_file.write(np_array.tobytes(order='C'))
        self._sizes.append(np_array.size)

    def end_document(self):
        self._doc_idx.append(len(self._sizes))

    def merge_file_(self, another_file):
        # Concatenate index
        index = MMapIndexedDataset.Index(index_file_path(another_file))
        assert index.dtype == self._dtype

        for size in index.sizes:
            self._sizes.append(size)

        # Concatenate data
        with open(data_file_path(another_file), 'rb') as f:
            shutil.copyfileobj(f, self._data_file)

    def finalize(self, index_file):
        self._data_file.close()

        with MMapIndexedDataset.Index.writer(index_file, self._dtype) as index:
            index.write(self._sizes, self._doc_idx)


class _S3Agent:
    """Retrieve byte ranges from S3 for the S3IndexedDataset."""

    def __init__(self, path, cache_nbytes):
        self._client = boto3.client('s3')
        assert path.startswith('s3://')
        path = path[len('s3://') :]
        self._bucket, self._key = path.split('/', 1)
        self._cache = None
        self._cache_bytes_start = None
        self._cache_bytes_end = None
        self._cache_nbytes = cache_nbytes

    def _extract_from_cache(self, offset, size):
        start = offset - self._cache_bytes_start
        assert start >= 0
        end = start + size
        assert end <= len(self._cache)
        return self._cache[start:end]

    def get_bytes(self, offset, size):
        """Get `size` bytes starting at `offset`.

        If the requested span of bytes [`offset`, `offset` + `size`) is covered
        by the in-memory cache maintained by this class, then this function
        extracts the requested span from that cache and returns it.

        Otherwise, this function first refreshes the cache and then extracts the
        requested span from the refreshed cache and returns it.

        The cache is refreshed based on `offset` and `size`. In particular, we
        divide all the bytes in an object into blocks, where each block contains
        `cache_size` bytes. We assign each block an index starting from 0.
        We take the block with index (`offset` // `cache_size`) to refresh the
        cache. If this new block still does not cover the requested span, we extend
        it just enough to include `offset` + `size`.
        """
        if self._cache is not None and offset >= self._cache_bytes_start and offset + size <= self._cache_bytes_end:
            return self._extract_from_cache(offset, size)

        bytes_start = (offset // self._cache_nbytes) * self._cache_nbytes
        assert bytes_start >= 0
        assert offset >= bytes_start
        bytes_end = max(bytes_start + self._cache_nbytes, offset + size)
        assert bytes_end >= 1
        self._cache = self._client.get_object(
            Bucket=self._bucket,
            Key=self._key,
            # Subtract 1, because the end of Range is inclusive.
            Range=f'bytes={bytes_start}-{bytes_end-1}',
        )['Body'].read()
        self._cache_bytes_start = bytes_start
        self._cache_bytes_end = bytes_end
        return self._extract_from_cache(offset, size)

    def close(self):
        self._client.close()


class S3IndexedDataset(torch.utils.data.Dataset):
    """Load a dataset stored in the same format as the MMapIndexedDataset from S3."""

    class Index(object):
        _HDR_MAGIC = b'MMIDIDX\x00\x00'

        def __init__(self, local_path):
            with open(local_path, 'rb') as stream:
                magic_test = stream.read(9)
                assert self._HDR_MAGIC == magic_test, (
                    'Index file doesn\'t match expected format. '
                    'Make sure that --dataset-impl is configured properly.'
                )
                version = struct.unpack('<Q', stream.read(8))
                assert (1,) == version

                (dtype_code,) = struct.unpack('<B', stream.read(1))
                self._dtype = dtypes[dtype_code]

                self._len = struct.unpack('<Q', stream.read(8))[0]
                self._doc_count = struct.unpack('<Q', stream.read(8))[0]
                offset = stream.tell()

            self._bin_buffer_mmap = np.memmap(local_path, mode='r', order='C')
            self._bin_buffer = memoryview(self._bin_buffer_mmap)
            logging.info("    reading sizes...")
            self._sizes = np.frombuffer(self._bin_buffer, dtype=np.int32, count=self._len, offset=offset)
            logging.info("    reading pointers...")
            self._pointers = np.frombuffer(
                self._bin_buffer, dtype=np.int64, count=self._len, offset=offset + self._sizes.nbytes
            )
            logging.info("    reading document index...")
            self._doc_idx = np.frombuffer(
                self._bin_buffer,
                dtype=np.int64,
                count=self._doc_count,
                offset=offset + self._sizes.nbytes + self._pointers.nbytes,
            )

        def __del__(self):
            self._bin_buffer_mmap._mmap.close()
            del self._bin_buffer_mmap

        @property
        def dtype(self):
            return self._dtype

        @property
        def sizes(self):
            return self._sizes

        @property
        def doc_idx(self):
            return self._doc_idx

        @lru_cache(maxsize=8)
        def __getitem__(self, i):
            return self._pointers[i], self._sizes[i]

        def __len__(self):
            return self._len

    def __init__(self, path, index_cache_dir, data_cache_nbytes):
        """Initialize the S3IndexedDataset.

        Args:
        * path: The path to a .bin file and a .idx file in S3 excluding the file extension.
        * index_cache_dir: Download the .idx file to this local directory so that we can memory map it.
        * data_cache_nbytes: Stream the .bin file into memory in blocks of this number of bytes.
          If the cache size is too small, then we send a request to S3 at each call of `get_bytes`,
          which is slow, because each request has a fixed cost independent of the size of the byte range
          requested. If the cache size is too large, then we only rarely have to send requests to S3,
          but it takes a lot of time to complete the request when we do, which can block training.
          We have found that a cache size of 128 * 1024 * 1024 (i.e., 128 MiB) has worked well
          (though we have not put that much effort into tuning it).
        """
        super().__init__()

        assert path is not None, "S3IndexDataset only supports path is not None"
        assert index_cache_dir is not None, "S3IndexDataset only supports index_cache_dir is not None"
        assert data_cache_nbytes is not None, "S3IndexDataset only supports data_cache_nbytes is not None"

        # The arguments are populated in the `_do_init` method.
        self._path = None
        self._index_cache_dir = None
        self._data_cache_nbytes = None
        self._s3_agent = None
        self._index = None

        # Download .idx file to the index cache.
        #
        # Download in sequential order by rank rather than concurrently to avoid
        # race conditions. If `index_cache_dir` is in a filesystem shared across
        # nodes, then only global rank 0 will download the file. Otherwise, exactly
        # 1 rank per node will download the file. This assumes that a download does
        # not take that long, because the wall time scales linearly in the number of
        # nodes.
        #
        # Download here rather than in self._do_init, because __init__ is called in
        # a training process, which has a rank, while self._do_init may be called inside
        # the process spawned for a dataloader worker, which is not a training process and
        # does not have a rank.
        client = boto3.client("s3")
        assert torch.distributed.is_available() and torch.distributed.is_initialized()
        for rank in range(torch.distributed.get_world_size()):
            if torch.distributed.get_rank() == rank and AppState().local_rank == 0:
                parsed_s3_path = parse_s3_path(index_file_path(path))
                filename = local_index_file_path(path, index_cache_dir)
                dirname = os.path.dirname(filename)
                os.makedirs(dirname, exist_ok=True)
                client.download_file(parsed_s3_path.bucket, parsed_s3_path.key, filename)
            torch.distributed.barrier()

        self._do_init(path, index_cache_dir, data_cache_nbytes)

    def __getstate__(self):
        return (self._path, self._index_cache_dir, self._data_cache_nbytes)

    def __setstate__(self, state):
        self._do_init(state[0], state[1], state[2])

    def _do_init(self, path, index_cache_dir, data_cache_nbytes):
        self._path = path
        self._index_cache_dir = index_cache_dir
        self._data_cache_nbytes = data_cache_nbytes
        self._index = self.Index(local_index_file_path(self._path, self._index_cache_dir))
        self._s3_agent = _S3Agent(data_file_path(self._path), data_cache_nbytes)

    def __del__(self):
        if self._s3_agent is not None:
            self._s3_agent.close()
            del self._s3_agent
        del self._index

    def __len__(self):
        return len(self._index)

    def get(self, idx, offset=0, length=None):
        """ Retrieves a single item from the dataset with the option to only
        return a portion of the item.
        """
        ptr, size = self._index[idx]
        if length is None:
            length = size - offset
        ptr += offset * np.dtype(self._index.dtype).itemsize
        np_array = np.frombuffer(
            self._s3_agent.get_bytes(ptr, length * self._index.dtype().itemsize), dtype=self._index.dtype
        )
        return np_array

    @property
    def sizes(self):
        return self._index.sizes

    @property
    def dtype(self):
        return self._index.dtype

    @property
    def data_cache_nbytes(self):
        return self._data_cache_nbytes

    @property
    def supports_prefetch(self):
        return False

    @staticmethod
    def exists(path):
        client = boto3.client("s3")
        return object_exists(client, index_file_path(path)) and object_exists(client, data_file_path(path))
