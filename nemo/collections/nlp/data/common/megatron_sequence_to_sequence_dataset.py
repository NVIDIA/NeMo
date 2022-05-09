# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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
import numpy as np
import torch

from nemo.collections.nlp.data.language_modeling.megatron.dataset_utils import get_indexed_dataset_
from nemo.collections.nlp.modules.common.megatron.utils import ApexGuardDefaults
from nemo.utils import logging

try:
    from apex.transformer import parallel_state

    HAVE_APEX = True
except (ImportError, ModuleNotFoundError):
    ModelType = ApexGuardDefaults()
    HAVE_APEX = False

__all__ = ["MegatronSequenceToSequenceDataset"]


class MegatronSequenceToSequenceDataset(torch.utils.data.Dataset):
    """Sequence to Sequence Dataset based on Megatron Dataset Utils."""

    def __init__(
        self,
        src_dataset_prefix,
        tgt_dataset_prefix,
        src_tokenizer,
        tgt_tokenizer,
        max_encoder_seq_length,
        max_decoder_seq_length,
        start_index=0,
        end_index=None,
        data_impl='mmap',
        skip_warmup=True,
        dataset_index_sampling_override=False,
    ):
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer
        self.src_dataset_prefix = src_dataset_prefix
        self.tgt_dataset_prefix = tgt_dataset_prefix
        self.data_impl = data_impl
        self.start_index = start_index
        self.end_index = end_index
        self.max_encoder_seq_length = max_encoder_seq_length
        self.max_decoder_seq_length = max_decoder_seq_length
        self.src_indexed_dataset = self._get_indexed_dataset(src_dataset_prefix, data_impl, skip_warmup)
        self.tgt_indexed_dataset = self._get_indexed_dataset(tgt_dataset_prefix, data_impl, skip_warmup)
        assert len(self.src_indexed_dataset) == len(self.tgt_indexed_dataset)
        self._dataset_length = lambda dataset: dataset.sizes.shape[0]
        if not end_index:
            self.end_index = self._dataset_length(self.src_indexed_dataset) - 1
        if dataset_index_sampling_override:
            logging.info("Disregarding index selections and making up our own.")
            # Make sure the shuffling is different per rank so that the same
            # samples aren't selected by every worker.
            self.rng = np.random.default_rng(seed=parallel_state.get_data_parallel_rank())

            dataset_len = len(self)
            if dataset_len <= np.iinfo(np.uint32).max:
                dtype = np.uint32
            else:
                dtype = np.uint64
            self.sampling = np.arange(dataset_len, dtype=dtype)
            self.rng.shuffle(self.sampling)

            self.sampling_indx = 0
        else:
            self.sampling = None

        self._print_stats('Source Datset', self.start_index, self.end_index)
        self._print_stats('Target Dataset', self.start_index, self.end_index)

    def __len__(self):
        self.end_index - self.start_index

    def _get_indexed_dataset(self, data_prefix, data_impl, skip_warmup):
        indexed_dataset = get_indexed_dataset_(data_prefix, data_impl, skip_warmup)
        return indexed_dataset

    def _print_stats(self, name, start, end):
        logging.info('    {}:'.format(name))
        logging.info('     sentence indices in [{}, {}) total of {} ' 'sentences'.format(start, end, end - start))

    def __len__(self):
        return self.end_index - self.start_index

    def __getitem__(self, idx):
        if isinstance(idx, np.int64):
            idx = idx.item()
        if self.sampling is not None:
            idx = self.sampling[self.sampling_idx]
            self.sampling_idx += 1
            if self.sampling_idx == len(self.sampling):
                self.sampling_idx = 0
                self.rng.shuffle(self.sampling)

        local_idx = idx + self.start_index
        assert local_idx < self.end_index

        # Truncate input and output sequences to the maximum length
        src_sent = self.src_indexed_dataset[local_idx]
        src_tokens = [token for token in src_sent]
        if len(src_tokens) > self.max_encoder_seq_length - 2:
            src_tokens = src_tokens[: self.max_encoder_seq_length - 2]

        tgt_sent = self.tgt_indexed_dataset[local_idx]
        tgt_tokens = [token for token in tgt_sent]
        if len(tgt_tokens) > self.max_decoder_seq_length - 2:
            tgt_tokens = tgt_tokens[: self.max_decoder_seq_length - 2]

        return self.build_tokens_types_from_ids(
            src_tokens, tgt_tokens
        )

    def round_to_nearest(self, length, modulo):
        return (length + modulo - 1) // modulo * modulo

    def build_tokens_types_from_ids(self, src_ids, tgt_ids):
        """Build token types and paddings, trim if needed, and pad if needed."""

        enc_ids = []
        tokentypes_enc = []

        # [BOS]
        enc_ids.append(self.src_tokenizer.bos_id)
        tokentypes_enc.append(0)

        # A.
        len_src = len(src_ids)
        enc_ids.extend(src_ids)
        tokentypes_enc.extend([0] * len_src)

        # Cap the size.
        if len(enc_ids) > self.max_encoder_seq_length - 1:
            enc_ids = enc_ids[0 : self.max_encoder_seq_length - 1]
            tokentypes_enc = tokentypes_enc[0 : self.max_encoder_seq_length - 1]

        # [EOS].
        enc_ids.append(self.src_tokenizer.eos_id)
        tokentypes_enc.append(0)

        # B.
        # Not enforcing sequence length checking for target sequences
        # if trg_ids is not None:
        dec_in_ids, dec_out_ids = [self.tgt_tokenizer.bos_id], []
        dec_in_ids.extend(tgt_ids)
        dec_out_ids.extend(tgt_ids)

        if len(dec_in_ids) > self.max_decoder_seq_length:
            dec_in_ids = dec_in_ids[0 : self.max_decoder_seq_length]
            dec_out_ids = dec_out_ids[0 : self.max_decoder_seq_length - 1]

        dec_out_ids.append(self.tgt_tokenizer.eos_id)

        return {
            'encoder_ids' : enc_ids,
            'token_type_ids': tokentypes_enc,
            'encoder_sequence_length': len(enc_ids),
            'decoder_input_ids': dec_in_ids,
            'decoder_output_ids': dec_out_ids,
            'decoder_sequence_length': len(dec_in_ids)
        }

    def _collate_fn(self, batch):
        """Build collate function for data loader."""
        max_enc_seq_len = self.round_to_nearest(max([x['encoder_sequence_length'] for x in batch]), 8)
        max_dec_seq_len = self.round_to_nearest(max([x['decoder_sequence_length'] for x in batch]), 8)

        batch_size = len(batch)
        enc_ids = torch.zeros((batch_size, max_enc_seq_len), dtype=torch.int64)
        dec_in_ids = torch.zeros((batch_size, max_dec_seq_len), dtype=torch.int64)
        labels = torch.zeros((batch_size, max_dec_seq_len), dtype=torch.int64)
        loss_mask = torch.zeros((batch_size, max_dec_seq_len), dtype=torch.int64)

        for i, example in enumerate(batch):
            enc_seq_len = example['encoder_sequence_length']
            enc_ids[i] = torch.tensor(example['encoder_ids'] + [0] * (max_enc_seq_len - enc_seq_len), dtype=torch.int64)

            dec_seq_len = batch[i]['decoder_sequence_length']
            dec_in_ids[i] = torch.tensor(example['decoder_input_ids'] + [0] * (max_dec_seq_len - dec_seq_len), dtype=torch.int64)
            labels[i] = torch.tensor(example['decoder_output_ids'] + [0] * (max_dec_seq_len - dec_seq_len), dtype=torch.int64)
            loss_mask[i] = torch.tensor([1] * dec_seq_len + [0] * (max_dec_seq_len - dec_seq_len), dtype=torch.int64)

        # Create attention masks
        enc_mask = (enc_ids != self.src_tokenizer.pad_id).long()
        dec_mask = (dec_in_ids != self.tgt_tokenizer.pad_id).long()

        sample = {
            'text_enc': enc_ids,
            'text_dec': dec_in_ids,
            'labels': labels,
            'loss_mask': loss_mask,
            'enc_mask': enc_mask,
            'dec_mask': dec_mask,
        }
        return sample
