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

"""GPT style dataset."""

import os
import time

import numpy as np
import torch
from omegaconf.dictconfig import DictConfig

from nemo.collections.nlp.data.language_modeling.megatron.base_dataset_utils import (
    get_datasets_weights_and_num_samples,
    get_train_valid_test_split_,
)
from nemo.collections.nlp.data.language_modeling.megatron.blendable_dataset import BlendableDataset
from nemo.collections.nlp.data.language_modeling.megatron.indexed_dataset import deallocate_indexed_dataset_memory, IndexedCachedDataset, MMapIndexedDataset, IndexedDataset
from nemo.collections.nlp.data.language_modeling.megatron.indexed_dataset import make_dataset as make_indexed_dataset
from nemo.core import Dataset
from nemo.utils import logging

try:
    from megatron.core import parallel_state

    HAVE_MEGATRON_CORE = True

except (ImportError, ModuleNotFoundError):

    HAVE_MEGATRON_CORE = False


def build_dataset(cfg, data_prefix, data_impl, num_samples, seq_length, seed, skip_warmup, tokenizer, name):
    def _build_dataset(current_data_prefix, current_num_samples):
        delay_data_mmap = cfg.data.get('delay_data_mmap', False)
        indexed_dataset = get_indexed_dataset_(current_data_prefix, data_impl, skip_warmup, delay_data_mmap)
        total_num_of_documents = len(indexed_dataset)
        # Print stats about the splits.
        logging.info(' > dataset split:')
        logging.info('     Total {} documents is : {} '.format(name, total_num_of_documents))
        drop_last = True
        if name == "valid":
            drop_last = cfg.data.get("validation_drop_last", True)
        dataset = SpeechLM_T5dataset(
            cfg,
            tokenizer,
            name,
            current_data_prefix,
            np.arange(start=0, stop=total_num_of_documents, step=1, dtype=np.int32),
            indexed_dataset,
            current_num_samples,
            seq_length,
            seed,
            drop_last=drop_last,
        )
        return dataset

    if len(data_prefix) == 1:
        return _build_dataset(data_prefix[0], num_samples)

    else:
        output = get_datasets_weights_and_num_samples(data_prefix, num_samples)
        prefixes, weights, datasets_num_samples = output
        datasets = []
        for i in range(len(prefixes)):
            dataset = _build_dataset(prefixes[i], datasets_num_samples[i])
            datasets.append(dataset)
        return BlendableDataset(datasets, weights, num_samples)


def build_train_valid_test_datasets(
    cfg,
    data_prefix,
    data_impl,
    splits_string,
    train_valid_test_num_samples,
    seq_length,
    seed,
    skip_warmup,
    tokenizer,
):
    
    if isinstance(data_prefix, DictConfig):
        assert (
            data_prefix.get('train') is not None
            and data_prefix.get('test') is not None
            and data_prefix.get('validation') is not None
        ), f"Data prefix dictionary should have train, test and validation keys.  data_prefix currently has only {data_prefix.keys()}"
        if cfg.data.splits_string is not None:
            logging.warning(cfg.data.splits_string + " ignored since data prefix is of type dictionary.")
        train_ds = build_dataset(
            cfg,
            data_prefix["train"],
            data_impl,
            int(train_valid_test_num_samples[0]),
            seq_length,
            seed,
            skip_warmup,
            tokenizer,
            "train",
        )
        validation_ds = build_dataset(
            cfg,
            data_prefix["validation"],
            data_impl,
            int(train_valid_test_num_samples[1]),
            seq_length,
            seed,
            skip_warmup,
            tokenizer,
            "valid",
        )
        test_ds = build_dataset(
            cfg,
            data_prefix["test"],
            data_impl,
            int(train_valid_test_num_samples[2]),
            seq_length,
            seed,
            skip_warmup,
            tokenizer,
            "test",
        )
        return train_ds, validation_ds, test_ds

    else:
        # Single dataset.
        if len(data_prefix) == 1:
            return _build_train_valid_test_datasets(
                cfg,
                data_prefix[0],
                data_impl,
                splits_string,
                train_valid_test_num_samples,
                seq_length,
                seed,
                skip_warmup,
                tokenizer,
            )

        # Blending dataset.
        # Parse the values.
        output = get_datasets_weights_and_num_samples(data_prefix, train_valid_test_num_samples)
        prefixes, weights, datasets_train_valid_test_num_samples = output

        # Build individual datasets.
        train_datasets = []
        valid_datasets = []
        test_datasets = []
        for i in range(len(prefixes)):
            train_ds, valid_ds, test_ds = _build_train_valid_test_datasets(
                cfg,
                prefixes[i],
                data_impl[i],
                splits_string,
                datasets_train_valid_test_num_samples[i],
                seq_length,
                seed,
                skip_warmup,
                tokenizer,
            )
            if train_ds:
                train_datasets.append(train_ds)
            if valid_ds:
                valid_datasets.append(valid_ds)
            if test_ds:
                test_datasets.append(test_ds)

        train_n, valid_n, test_n = map(sum, zip(*datasets_train_valid_test_num_samples))

        # Blend.
        blending_train_dataset = None
        if train_datasets:
            blending_train_dataset = BlendableDataset(train_datasets, weights, train_n)
        blending_valid_dataset = None
        if valid_datasets:
            blending_valid_dataset = BlendableDataset(valid_datasets, weights, valid_n)
        blending_test_dataset = None
        if test_datasets:
            blending_test_dataset = BlendableDataset(test_datasets, weights, test_n)

        return (blending_train_dataset, blending_valid_dataset, blending_test_dataset)


def _build_train_valid_test_datasets(
    cfg,
    data_prefix,
    data_impl,
    splits_string,
    train_valid_test_num_samples,
    seq_length,
    seed,
    skip_warmup,
    tokenizer,
):
    """Build train, valid, and test datasets."""

    # Indexed dataset.
    delay_data_mmap = cfg.data.get('delay_data_mmap', False)
    indexed_dataset = get_indexed_dataset_(data_prefix, data_impl, skip_warmup, delay_data_mmap)

    total_num_of_documents = len(indexed_dataset)
    splits = get_train_valid_test_split_(splits_string, total_num_of_documents)

    # Print stats about the splits.
    logging.info(' > dataset split:')

    def print_split_stats(name, index):
        logging.info('    {}:'.format(name))
        logging.info(
            '     document indices in [{}, {}) total of {} '
            'documents'.format(splits[index], splits[index + 1], splits[index + 1] - splits[index])
        )

    print_split_stats('train', 0)
    print_split_stats('validation', 1)
    print_split_stats('test', 2)

    def build_dataset(index, name):
        dataset = None
        if splits[index + 1] > splits[index]:
            documents = np.arange(start=splits[index], stop=splits[index + 1], step=1, dtype=np.int32)
            drop_last = True
            if name == "valid":
                drop_last = cfg.data.get("validation_drop_last", True)
            dataset = SpeechLM_T5dataset(
                cfg,
                tokenizer,
                name,
                data_prefix,
                documents,
                indexed_dataset,
                train_valid_test_num_samples[index],
                seq_length,
                seed,
                drop_last=drop_last,
            )
        return dataset

    train_dataset = build_dataset(0, 'train')
    valid_dataset = build_dataset(1, 'valid')
    test_dataset = build_dataset(2, 'test')
    if isinstance(indexed_dataset, MMapIndexedDataset):
        deallocate_indexed_dataset_memory(indexed_dataset)

    return (train_dataset, valid_dataset, test_dataset)


def get_indexed_dataset_(data_prefix, data_impl, skip_warmup, delay_data_mmap=False):
    """Build indexed dataset."""
    logging.info(' > building dataset index ...')

    start_time = time.time()
    indexed_dataset = make_indexed_dataset(data_prefix, data_impl, skip_warmup, delay_data_mmap=delay_data_mmap)
    logging.info(' > finished creating indexed dataset in {:4f} ' 'seconds'.format(time.time() - start_time))
    logging.info('    number of documents: {}'.format(len(indexed_dataset)))

    return indexed_dataset


class SpeechLM_T5dataset(Dataset):
    def __init__(
        self,
        cfg,
        tokenizer,
        name,
        data_prefix,
        documents,
        indexed_dataset,
        num_samples,
        seq_length,
        seed,
        drop_last=True,
    ):
        if not HAVE_MEGATRON_CORE:
            raise ImportError(
                "megatron-core was not found. Please see the NeMo README for installation instructions: https://github.com/NVIDIA/NeMo#megatron-gpt."
            )

        super().__init__()
        self.name = name
        self.indexed_dataset = indexed_dataset
        self.drop_last = drop_last
        self.seq_length = seq_length

        # Checks
        assert np.min(documents) >= 0
        assert np.max(documents) < len(indexed_dataset)

        self.reset_position_ids = cfg.data.get('reset_position_ids', False)
        self.reset_attention_mask = cfg.data.get('reset_attention_mask', False)
        self.eod_mask_loss = cfg.data.get('eod_mask_loss', False)
        self.create_inputs = any([self.reset_position_ids, self.reset_attention_mask, self.eod_mask_loss])
        self.cached_inputs = False
        self.eos_id = tokenizer.eos_id
        self.pad_id = tokenizer.pad_id
        self.mask_id = tokenizer.mask_id
        self.no_seqlen_plus_one_input_tokens = cfg.data.get('no_seqlen_plus_one_input_tokens', False)
        self.add_extra_token = 1
        if self.no_seqlen_plus_one_input_tokens:
            self.add_extra_token = 0
        self.shuffle_documents = cfg.data.get('shuffle_documents', True)
        self.exchange_indices_distributed = cfg.data.get('exchange_indices_distributed', False)
        self.speech_offset = cfg.data.get('speech_offset', 29184)
        self.speech_codebook_size = cfg.data.get('speech_codebook_size', 1024)
        self.mask_context_prob = cfg.data.get('mask_context_prob', 0.3)
        # save index mappings to a configurable dir
        self.index_mapping_dir = cfg.data.get('index_mapping_dir', None)

        # create index_mapping_dir on rank 0
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            if torch.distributed.get_rank() == 0:
                if self.index_mapping_dir is not None and not os.path.isdir(self.index_mapping_dir):
                    os.makedirs(self.index_mapping_dir)
            torch.distributed.barrier()

        splits = self.indexed_dataset.sizes
        if isinstance(self.indexed_dataset, IndexedDataset):
            splits = self.indexed_dataset.sizes[1::2]
        
        self.num_samples = num_samples
        # Build index mappings.
        self.doc_idx, self.sample_idx, self.shuffle_idx = _build_index_mappings(
            self.name,
            data_prefix,
            documents,
            splits,
            num_samples,
            seq_length,
            seed,
            index_mapping_dir=self.index_mapping_dir,
            drop_last=drop_last,
            add_extra_token=self.add_extra_token,
            shuffle_documents=self.shuffle_documents,
            exchange_indices_distributed=self.exchange_indices_distributed,
        )
        # deallocate_indexed_dataset_memory(self.indexed_dataset)

    def create_data_mmap(self):
        self.indexed_dataset.create_data_mmap()

    def __len__(self):
        # -1 is due to data structure used to retieve the index:
        #    sample i --> [sample_idx[i], sample_idx[i+1])
        return min(self.num_samples, self.sample_idx.shape[0] - 1)

    def _get_tokens(self, idx: int) -> np.ndarray:

        # Get the shuffled index.
        idx = self.shuffle_idx[idx]
        # Start and end documents and offsets.
        doc_index_f = self.sample_idx[idx][0]
        doc_index_l = self.sample_idx[idx + 1][0]
        offset_f = self.sample_idx[idx][1]
        offset_l = self.sample_idx[idx + 1][1]
        # If we are within the same document, just extract the chunk.
        if doc_index_f == doc_index_l:
            sample = self.indexed_dataset.get(
                self.doc_idx[doc_index_f], offset=offset_f, length=offset_l - offset_f + self.add_extra_token
            )
        else:
            # Otherwise, get the rest of the initial document.
            sample_list = [self.indexed_dataset.get(self.doc_idx[doc_index_f], offset=offset_f)]
            # Loop over all in between documents and add the entire document.
            for i in range(doc_index_f + 1, doc_index_l):
                sample_list.append(self.indexed_dataset.get(self.doc_idx[i]))
            # And finally add the relevant portion of last document.
            sample_list.append(
                self.indexed_dataset.get(self.doc_idx[doc_index_l], length=offset_l + self.add_extra_token)
            )
            if sample_list[0].ndim == 2:
                sample = np.concatenate(sample_list, axis=1) # (8, 513)
            else:
                sample = np.concatenate(sample_list) #(513)
        
        if sample.ndim == 1:
            sample_len = len(sample)
        else:
            sample_len = sample.shape[1]

        if sample_len != (self.seq_length + self.add_extra_token):
            logging.info(
                F' > WARNING: Got sample of length: {len(sample)} for sequence length={self.seq_length+self.add_extra_token}, padding the sample to match sequence length'
            )
            if sample.ndim == 1:
                sample = np.array(sample, dtype=np.int64)
                sample = np.pad(
                    sample, (0, self.seq_length + self.add_extra_token - len(sample)), mode='constant', constant_values=0
                )
            else:
                sample = np.array(sample, dtype=np.int64)
                sample = np.pad(
                    sample,
                    ((0, 0), (0, self.seq_length + self.add_extra_token - sample.shape[1])),
                    mode='constant',
                    constant_values=0,
                )
        return sample.astype(np.int64)

    def _mask_encoder_input(self, enc_input):
        span_length = torch.poisson(torch.tensor([3.5]))
        span_length = int(span_length.item())
        span_length = max(span_length, 1)

        n_timesteps = enc_input.shape[1]
        span_length = min(span_length, n_timesteps)
        n_spans = int(n_timesteps // span_length)
        n_masked_spans = int(n_spans * self.mask_context_prob)
        masked_spans = torch.randperm(n_spans)[:n_masked_spans]
        for i in masked_spans:
            enc_input[:, i * span_length : (i + 1) * span_length] = self.mask_id
        
        return enc_input

    def _getitem_from_speech(self, tokens):
        assert tokens.ndim == 2

        for _i in range(tokens.shape[0]):
            tokens[_i] = tokens[_i] + self.speech_offset + (_i * self.speech_codebook_size)
        
        enc_input = tokens[:, 1:] * 1 # to avoid changing the original tensor
        dec_input = tokens[:, :-1] * 1
        labels = tokens[:, 1:] * 1
        for _i in range(1, tokens.shape[0]):
            # bring other layers back in range (0, 1024)
            labels[_i] = labels[_i] - self.speech_offset - (_i * self.speech_codebook_size)
            # dec_input[_i] = dec_input[_i] - self.speech_offset - (_i * self.speech_codebook_size)

        enc_input = self._mask_encoder_input(enc_input)

        # TODO add pad id condition as well for enc_input?
        enc_mask = (enc_input[0] != self.mask_id).long()
        dec_mask = (labels[0] != self.pad_id ).long()
        # loss_mask = (enc_input[0] == self.mask_id ).long()
        loss_mask = torch.ones_like(dec_mask)

        return {
            'enc_input': enc_input,
            'dec_input': dec_input,
            'labels': labels,
            'enc_mask': enc_mask,
            'dec_mask': dec_mask,
            'loss_mask': loss_mask,
        }

    def _getitem_from_text(self, tokens):
        # tokens is of shape (seq_len, )
        # change tokens to (8, seq_len) by padding with 0
        assert tokens.ndim == 1
        tokens = torch.tile(tokens, (8, 1))
        tokens[1:] = 0
        
        enc_input = tokens[:, 1:] * 1 # to avoid changing the original tensor
        dec_input = tokens[:, :-1] * 1
        labels = tokens[:, 1:] * 1
        enc_input = self._mask_encoder_input(enc_input)
        enc_input[1:] = 0
        enc_mask = (enc_input[0] != self.mask_id).long()
        dec_mask = (labels[0] != self.pad_id ).long()
        # loss_mask = (enc_input[0] == self.mask_id ).long()
        loss_mask = torch.ones_like(dec_mask)

        return {
            'enc_input': enc_input,
            'dec_input': dec_input,
            'labels': labels,
            'enc_mask': enc_mask,
            'dec_mask': dec_mask,
            'loss_mask': loss_mask,
        }

    def __getitem__(self, idx):
        tokens = torch.from_numpy(self._get_tokens(idx))
        is_speech = tokens.ndim == 2
        if is_speech:
            item = self._getitem_from_speech(tokens)
        else:
            item = self._getitem_from_text(tokens)
        
        item['speech_mask'] = torch.ones_like(item['enc_input'][0]) * is_speech
        item['position_ids'] = torch.arange(item['enc_input'].shape[1], dtype=torch.long)

        return item
        


def _build_index_mappings(
    name,
    data_prefix,
    documents,
    sizes,
    num_samples,
    seq_length,
    seed,
    index_mapping_dir: str = None,
    drop_last: bool = True,
    add_extra_token: int = 1,
    shuffle_documents: bool = True,
    exchange_indices_distributed: bool = False,
):
    """Build doc-idx, sample-idx, and shuffle-idx.
    doc-idx: is an array (ordered) of documents to be used in training.
    sample-idx: is the start document index and document offset for each
       training sample.
    shuffle-idx: maps the sample index into a random index into sample-idx.
    """
    # Number of tokens in each epoch and number of required epochs.
    tokens_per_epoch = _num_tokens(documents, sizes)
    num_epochs = _num_epochs(tokens_per_epoch, seq_length, num_samples, add_extra_token)
    # rng state
    np_rng = np.random.RandomState(seed=seed)

    # Filename of the index mappings.
    if index_mapping_dir is not None:
        _filename = os.path.join(index_mapping_dir, os.path.basename(data_prefix))
    else:
        _filename = data_prefix
    _filename += '_{}_indexmap'.format(name)
    _filename += '_{}ns'.format(num_samples)
    _filename += '_{}sl'.format(seq_length)
    _filename += '_{}s'.format(seed)
    doc_idx_filename = _filename + '_doc_idx.npy'
    sample_idx_filename = _filename + '_sample_idx.npy'
    shuffle_idx_filename = _filename + '_shuffle_idx.npy'

    # Build the indexed mapping if not exist.
    if torch.distributed.get_rank() == 0:
        using_cached_indices = True
        if (
            (not os.path.isfile(doc_idx_filename))
            or (not os.path.isfile(sample_idx_filename))
            or (not os.path.isfile(shuffle_idx_filename))
        ):
            using_cached_indices = False
            logging.info(' > WARNING: could not find index map files, building ' 'the indices on rank 0 ...')

            # For the last epoch, decide whether include the entire epoch
            # in the global shuffle or not.

            # If we need only one epoch, then separating last epoch  does
            # not mean anything.
            if num_epochs == 1:
                separate_last_epoch = False
                print(' > only one epoch required, setting ' 'separate_last_epoch to False', flush=True)

            else:
                # Get the number of samples for the last epoch
                num_samples_from_epochs_minus_one = (
                    (num_epochs - 1) * tokens_per_epoch - add_extra_token
                ) // seq_length
                last_epoch_num_samples = num_samples - num_samples_from_epochs_minus_one
                assert last_epoch_num_samples >= 0, 'last epoch number of samples should be non-negative.'
                num_samples_per_epoch = (tokens_per_epoch - add_extra_token) // seq_length
                assert last_epoch_num_samples <= (
                    num_samples_per_epoch + 1
                ), 'last epoch number of samples exceeded max value.'
                # If we have less than 80% of the samples for the last epoch,
                # seperate out the epoch and treat it differently.
                # Note: the 80% number is just based on common sense and can
                # be adjusted if needed.
                separate_last_epoch = last_epoch_num_samples < int(0.80 * num_samples_per_epoch)
                if separate_last_epoch:
                    string = (
                        ' > last epoch number of samples ({}) is smaller '
                        'than 80% of number of samples per epoch ({}), '
                        'setting separate_last_epoch to True'
                    )
                else:
                    string = (
                        ' > last epoch number of samples ({}) is larger '
                        'than 80% of number of samples per epoch ({}), '
                        'setting separate_last_epoch to False'
                    )
                print(string.format(last_epoch_num_samples, num_samples_per_epoch), flush=True)

            # doc-idx.
            start_time = time.time()
            doc_idx = _build_doc_idx(documents, num_epochs, np_rng, separate_last_epoch, shuffle_documents)
            np.save(doc_idx_filename, doc_idx, allow_pickle=True)
            logging.info(
                ' > elasped time to build and save doc-idx mapping '
                '(seconds): {:4f}'.format(time.time() - start_time)
            )
            # sample-idx.
            start_time = time.time()
            # Use C++ implementation for speed.
            # First compile and then import.
            assert doc_idx.dtype == np.int32
            if sizes.dtype != np.int32:
                if np.max(np.abs(sizes)) < 2**31-1:
                    sizes = sizes.astype(np.int32)
                else:
                    raise NotImplementedError("Sizes needs to be int32?")
            # assert sizes.dtype == np.int32
            try:
                from nemo.collections.nlp.data.language_modeling.megatron.dataset_utils import compile_helper

                compile_helper()
                from nemo.collections.nlp.data.language_modeling.megatron import helpers
            except ImportError:
                raise ImportError(
                    f'Could not compile megatron dataset C++ helper functions and therefore cannot import helpers python file.'
                )

            sample_idx = helpers.build_sample_idx(
                sizes, doc_idx, seq_length, num_epochs, tokens_per_epoch, drop_last, add_extra_token
            )
            # sample_idx = _build_sample_idx(sizes, doc_idx, seq_length,
            #                              num_epochs, tokens_per_epoch, drop_last, add_extra_token)
            np.save(sample_idx_filename, sample_idx, allow_pickle=True)
            logging.info(
                ' > elasped time to build and save sample-idx mapping '
                '(seconds): {:4f}'.format(time.time() - start_time)
            )
            # shuffle-idx.
            start_time = time.time()
            # -1 is due to data structure used to retieve the index:
            #    sample i --> [sample_idx[i], sample_idx[i+1])
            if separate_last_epoch:
                num_samples_ = num_samples_from_epochs_minus_one
            else:
                num_samples_ = sample_idx.shape[0] - 1
            shuffle_idx = _build_shuffle_idx(num_samples_, sample_idx.shape[0] - 1, np_rng)
            np.save(shuffle_idx_filename, shuffle_idx, allow_pickle=True)
            logging.info(
                ' > elasped time to build and save shuffle-idx mapping'
                ' (seconds): {:4f}'.format(time.time() - start_time)
            )

    torch.distributed.barrier()
    counts = torch.cuda.LongTensor([1])
    torch.distributed.all_reduce(counts, group=parallel_state.get_data_parallel_group())
    torch.distributed.all_reduce(counts, group=parallel_state.get_pipeline_model_parallel_group())
    assert counts[0].item() == (
        torch.distributed.get_world_size()
        // torch.distributed.get_world_size(group=parallel_state.get_tensor_model_parallel_group())
    )

    if not exchange_indices_distributed or (torch.distributed.get_rank() == 0 and using_cached_indices):
        # Load mappings.
        start_time = time.time()
        logging.info(' > loading doc-idx mapping from {}'.format(doc_idx_filename))
        doc_idx = np.load(doc_idx_filename, allow_pickle=True, mmap_mode='r')
        logging.info(' > loading sample-idx mapping from {}'.format(sample_idx_filename))
        sample_idx = np.load(sample_idx_filename, allow_pickle=True, mmap_mode='r')
        logging.info(' > loading shuffle-idx mapping from {}'.format(shuffle_idx_filename))
        shuffle_idx = np.load(shuffle_idx_filename, allow_pickle=True, mmap_mode='r')
        logging.info('    loaded indexed file in {:3.3f} seconds'.format(time.time() - start_time))
        logging.info('    total number of samples: {}'.format(sample_idx.shape[0]))
        logging.info('    total number of epochs: {}'.format(num_epochs))

    if exchange_indices_distributed:
        if torch.distributed.get_rank() == 0:
            indices = [(doc_idx, sample_idx, shuffle_idx)]
        else:
            indices = [None]
        torch.distributed.broadcast_object_list(indices)
        doc_idx, sample_idx, shuffle_idx = indices[0]

    return doc_idx, sample_idx, shuffle_idx


def _num_tokens(documents, sizes):
    """Total number of tokens in the dataset."""
    return np.sum(sizes[documents])


def _num_epochs(tokens_per_epoch, seq_length, num_samples, add_extra_token=1):
    """Based on number of samples and sequence lenght, calculate how many
    epochs will be needed."""
    num_epochs = 0
    total_tokens = 0
    while True:
        num_epochs += 1
        total_tokens += tokens_per_epoch
        # -1 is because we need to retrieve seq_length + 1 token each time
        # but the last token will overlap with the first token of the next
        # sample except for the last sample.
        if ((total_tokens - add_extra_token) // seq_length) >= num_samples:
            return num_epochs


def _build_doc_idx(documents, num_epochs, np_rng, separate_last_epoch, shuffle=True):
    """Build an array with length = number-of-epochs * number-of-dcuments.
    Each index is mapped to a corresponding document."""
    if not separate_last_epoch or num_epochs == 1:
        doc_idx = np.mgrid[0:num_epochs, 0 : len(documents)][1]
        doc_idx[:] = documents
        doc_idx = doc_idx.reshape(-1)
        doc_idx = doc_idx.astype(np.int32)
        if shuffle:
            np_rng.shuffle(doc_idx)
        else:
            logging.info('Document shuffling disabled')
        return doc_idx

    doc_idx_first = _build_doc_idx(documents, num_epochs - 1, np_rng, False, shuffle)
    doc_idx_last = _build_doc_idx(documents, 1, np_rng, False, shuffle)
    return np.concatenate((doc_idx_first, doc_idx_last))


def _build_sample_idx(sizes, doc_idx, seq_length, num_epochs, tokens_per_epoch, drop_last=True, add_extra_token=1):
    """Sample index mapping is a 2D array with sizes
    [number-of-samples + 1, 2] where [..., 0] contains
    the index into `doc_idx` and [..., 1] is the
    starting offset in that document."""

    # Total number of samples. For -1 see comments in `_num_epochs`.
    if not drop_last:
        num_samples = -(-(num_epochs * tokens_per_epoch - add_extra_token) // seq_length)
    else:
        num_samples = (num_epochs * tokens_per_epoch - add_extra_token) // seq_length
    sample_idx = np.zeros([num_samples + 1, 2], dtype=np.int32)

    # Index into sample_idx.
    sample_index = 0
    # Index into doc_idx.
    doc_idx_index = 0
    # Begining offset for each document.
    doc_offset = 0
    # Start with first document and no offset.
    sample_idx[sample_index][0] = doc_idx_index
    sample_idx[sample_index][1] = doc_offset
    sample_index += 1
    while sample_index <= num_samples:
        # Start with a fresh sequence.
        remaining_seq_length = seq_length + add_extra_token
        while remaining_seq_length != 0:
            # Get the document length.
            doc_id = doc_idx[doc_idx_index]
            doc_length = sizes[doc_id] - doc_offset
            # And add it to the current sequence.
            remaining_seq_length -= doc_length
            # If we have more than a full sequence, adjust offset and set
            # remaining length to zero so we return from the while loop.
            # Note that -1 here is for the same reason we have -1 in
            # `_num_epochs` calculations.
            if remaining_seq_length <= 0:
                doc_offset += remaining_seq_length + doc_length - add_extra_token
                remaining_seq_length = 0
            else:
                # Otherwise, start from the begining of the next document.
                if doc_idx_index == (len(doc_idx) - 1):
                    assert (
                        sample_index == num_samples
                    ), F"sample_index={sample_index} and num_samples={num_samples} should be the same"
                    doc_offset = sizes[doc_idx[doc_idx_index]] - add_extra_token
                    break
                doc_idx_index += 1
                doc_offset = 0
        # Record the sequence.
        sample_idx[sample_index][0] = doc_idx_index
        sample_idx[sample_index][1] = doc_offset
        sample_index += 1

    return sample_idx


def _build_shuffle_idx(num_samples, total_size, np_rng):
    """Build the range [0, size) and shuffle."""
    print(
        ' > building shuffle index with split [0, {}) and [{}, {}) '
        '...'.format(num_samples, num_samples, total_size),
        flush=True,
    )

    dtype_ = np.uint32
    if total_size >= (np.iinfo(np.uint32).max - 1):
        dtype_ = np.int64

    shuffle_idx_first = np.arange(start=0, stop=num_samples, step=1, dtype=dtype_)
    np_rng.shuffle(shuffle_idx_first)
    if num_samples == total_size:
        return shuffle_idx_first

    shuffle_idx_last = np.arange(start=num_samples, stop=total_size, step=1, dtype=dtype_)
    np_rng.shuffle(shuffle_idx_last)

    return np.concatenate((shuffle_idx_first, shuffle_idx_last))
