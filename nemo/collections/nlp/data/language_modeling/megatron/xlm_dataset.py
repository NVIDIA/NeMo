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

"""XLM-Style datasets"""
from typing import List, Dict
import numpy as np

from nemo.collections.common.tokenizers.tokenizer_spec import TokenizerSpec
from nemo.collections.nlp.data.language_modeling.megatron.length_distribution_type import LengthDistribution
from nemo.collections.nlp.data.common.sequence_to_sequence_dataset import BinarizedMemmapSequenceToSequenceDataset
from nemo.collections.nlp.data.language_modeling.megatron.bert_dataset import (
    build_training_sample as build_training_sample_bert
)
from nemo.collections.nlp.data.language_modeling.megatron.t5_dataset import T5Dataset
from nemo.collections.nlp.data.language_modeling.megatron.ul2_dataset import UL2Dataset


class CrossLingualBERTDataset(BinarizedMemmapSequenceToSequenceDataset):
    """Cross-lingual BERT dataset similar to the translation-language modeling objective in the XLM paper (https://arxiv.org/abs/1901.07291)"""

    def __init__(
        self,
        src_dataset_prefix: str,
        tgt_dataset_prefix: str,
        src_tokenizer: TokenizerSpec,
        tgt_tokenizer: TokenizerSpec,
        max_src_seq_length: int,
        max_tgt_seq_length: int,
        seed: int = 1234,
        max_num_samples: int = None,
        masked_lm_prob: float = 0.15,
    ):
        super().__init__(
            src_dataset_prefix=src_dataset_prefix,
            tgt_dataset_prefix=tgt_dataset_prefix,
            src_tokenizer=src_tokenizer,
            tgt_tokenizer=tgt_tokenizer,
            max_src_seq_length=max_src_seq_length,
            max_tgt_seq_length=max_tgt_seq_length,
            seed=seed,
            max_num_samples=max_num_samples,
        )
        assert src_tokenizer == tgt_tokenizer
        # Vocab stuff.
        self.vocab = src_tokenizer.vocab
        self.vocab_id_list = list(range(self.src_tokenizer.original_vocab_size))
        self.vocab_id_to_token_dict = {idx: token for idx, token in enumerate(self.vocab)}
        self.cls_id = src_tokenizer.cls_id
        self.sep_id = src_tokenizer.sep_id
        self.mask_id = src_tokenizer.mask_id
        self.pad_id = src_tokenizer.pad_id
        self.bos_id = src_tokenizer.bos_id
        self.eos_id = src_tokenizer.eos_id
        self.max_seq_length = max_src_seq_length + max_tgt_seq_length
        self.masked_lm_prob = masked_lm_prob

    def __getitem__(self, idx):
        src, tgt = super()._get_sample(idx)
        if len(src) > self.max_src_seq_length:
            src = src[: self.max_src_seq_length]

        if len(tgt) > self.max_tgt_seq_length - 1: # -1 here to account for the <sep> token that gets added.
            tgt = tgt[: self.max_tgt_seq_length]

        np_rng = np.random.RandomState(seed=((self.seed + idx) % 2 ** 32))
        # Potentially swap src, tgt with a 50% chance to avoid learning associations based on position in the sequence.
        swap_src_tgt = np_rng.randint(0, 2)
        if swap_src_tgt == 0:
            sample = [np.concatenate((src, [self.sep_id], tgt))]
        elif swap_src_tgt == 1:
            sample = [np.concatenate((tgt, [self.sep_id], src))]

        return build_training_sample_bert(
            sample=sample,
            target_seq_length=sample[0].shape[0],
            max_seq_length=self.max_seq_length,  # needed for padding
            vocab_id_list=self.vocab_id_list,
            vocab_id_to_token_dict=self.vocab_id_to_token_dict,
            cls_id=self.cls_id,
            sep_id=self.sep_id,
            mask_id=self.mask_id,
            pad_id=self.pad_id,
            masked_lm_prob=self.masked_lm_prob,
            np_rng=np_rng,
            binary_head=False,
            whole_word_masking=False,
            skip_masking_id=self.sep_id
        )

    # Skip the parent collate function, since we don't need it for this dataset.
    def collate_fn(self, batch):
        return batch


class CrossLingualMakedSequenceToSequenceDataset(BinarizedMemmapSequenceToSequenceDataset):
    def __init__(
        self,
        src_dataset_prefix: str,
        tgt_dataset_prefix: str,
        src_tokenizer: TokenizerSpec,
        tgt_tokenizer: TokenizerSpec,
        max_src_seq_length: int,
        max_tgt_seq_length: int,
        max_seq_length_dec: int,
        seed: int = 1234,
        max_num_samples: int = None,
        masked_lm_prob: float = 0.15,
        extreme_masked_lm_prob: float = 0.5,
        max_ngram_size: int = 10,
        mean_ngram_size: int = None,
        min_ngram_size: int = 1,
        extreme_max_ngram_size: int = 128,
        extreme_mean_ngram_size: int = 64,
        extreme_min_ngram_size: int = 32,
        extreme_ngram_span_length_distribution: LengthDistribution = LengthDistribution.truncated_normal,
        geometric_dist: bool = True,
        permutation: bool = False,
        favor_long_ngrams: bool = False,
        masking_type: str = "t5",
    ):
        super().__init__(
            src_dataset_prefix=src_dataset_prefix,
            tgt_dataset_prefix=tgt_dataset_prefix,
            src_tokenizer=src_tokenizer,
            tgt_tokenizer=tgt_tokenizer,
            max_src_seq_length=max_src_seq_length,
            max_tgt_seq_length=max_tgt_seq_length,
            seed=seed,
            max_num_samples=max_num_samples,
        )
        self.max_seq_length_dec = max_seq_length_dec
        self.max_ngram_size = max_ngram_size
        self.mean_ngram_size = mean_ngram_size
        self.min_ngram_size = min_ngram_size
        self.geometric_dist = geometric_dist
        self.permutation = permutation
        self.favor_long_ngrams = favor_long_ngrams
        self.extreme_masked_lm_prob = extreme_masked_lm_prob
        self.extreme_max_ngram_size = extreme_max_ngram_size
        self.extreme_mean_ngram_size = extreme_mean_ngram_size
        self.extreme_min_ngram_size = extreme_min_ngram_size
        self.extreme_ngram_span_length_distribution = extreme_ngram_span_length_distribution
        self.masking_type = masking_type

        assert src_tokenizer == tgt_tokenizer
        # Vocab stuff.
        self.vocab_id_list = src_tokenizer.vocab
        self.vocab_id_to_token_dict = {idx: token for idx, token in enumerate(self.vocab_id_list)}
        self.cls_id = src_tokenizer.cls_id
        self.sep_id = src_tokenizer.sep_id
        self.mask_id = src_tokenizer.mask_id
        self.pad_id = src_tokenizer.pad_id
        self.bos_id = src_tokenizer.bos_id
        self.eos_id = src_tokenizer.eos_id
        self.max_seq_length = max_src_seq_length + max_tgt_seq_length
        self.masked_lm_prob = masked_lm_prob

        self.tokenizer_type = T5Dataset._determine_tokenizer_type(src_tokenizer, whole_word_masking=False)
        self._build()

    def _build(self):
        """
        Class-specific build method to be overridden by child classes.
        """
        self.sentinel_tokens = self.src_tokenizer.additional_special_tokens_ids
        assert len(self.sentinel_tokens) > 0
    
    @classmethod
    def get_example(
        cls,
        src: List[int],
        tgt: List[int],
        max_seq_length: int,
        max_seq_length_dec: int,
        masked_lm_prob: float,
        vocab_id_list: List[int],
        vocab_id_to_token_dict: Dict[int, str],
        cls_id: int,
        sep_id: int,
        mask_id: int,
        bos_id: int,
        eos_id: int,
        pad_id: int,
        sentinel_tokens: List[int],
        max_ngram_size: int,
        min_ngram_size: int,
        np_rng: np.random.RandomState,
        extreme_masked_lm_prob: float = None,
        extreme_max_ngram_size: int = None,
        extreme_mean_ngram_size: int = None,
        extreme_min_ngram_size: int = None,
        extreme_ngram_span_length_distribution: LengthDistribution = None,
        mean_ngram_size: int = None,
        favor_long_ngrams: bool = False,
        permutation: bool = False,
        geometric_dist: bool = True,
        tokenizer_type: str = "wordpiece",
        masking_type: str = "t5",
    ):
        """
        Arguments:
            src: A list of source token ids.
            tgt: A list of target token ids.
            max_seq_length: The maximum length of the sequence.
            max_seq_length_dec: The maximum length of the decoder sequence.
            masked_lm_prob: The probability of replacing a token with the mask token.
            vocab_id_list: A list of all token ids in the vocabulary.
            vocab_id_to_token_dict: A dictionary mapping token ids to tokens.
            cls_id: The id of the class token.
            sep_id: The id of the separator token.
            mask_id: The id of the mask token.
            pad_id: The id of the padding token.
            sentinel_tokens: A list of additional special tokens.
            max_ngram_size: The maximum ngram size to mask.
            np_rng: A numpy random number generator.
            extreme_max_ngram_size: The maximum ngram size to mask for extreme ngram masking if masking_type is "ul2".
            extreme_mean_ngram_size: The mean ngram size to mask for extreme ngram masking if masking_type is "ul2".
            extreme_min_ngram_size: The minimum ngram size to mask for extreme ngram masking if masking_type is "ul2".
            extreme_ngram_span_length_distribution: The distribution of ngram span lengths for extreme ngram masking if masking_type is "ul2".
            mean_ngram_size: The mean ngram size to mask.
            favor_long_ngrams: Whether to favor longer ngrams.
            permutation: Whether to permute the ngrams.
            geometric_dist: Whether to use a geometric distribution to sample ngram span sizes.
            tokenizer_type: The type of tokenizer being used.
            masking_type: The type of masking being used. Options ["t5", "ul2"].

        Returns:
            A dictionary containing the following items:
            text_enc: The token sequence to provide as input to the encoder.
            text_dec: The token sequence to provide as input to the decoder.
            labels: The token sequence to provide as labels to the decoder. (input sequence shifted right + <eos>)
            loss_mask: A mask indicating which elements of the decoder sequence should be ignored in loss computation.
            enc_mask: A mask indicating which elements of the encoder sequence should be ignored.
            dec_mask: A mask indicating which elements of the decoder sequence should be ignored.
        """
        # Potentially swap src, tgt with a 50% chance to avoid learning associations based on position in the sequence.
        swap_src_tgt = np_rng.randint(0, 2)
        if swap_src_tgt == 0:
            sample = [np.concatenate((src, [sep_id], tgt))]
        elif swap_src_tgt == 1:
            sample = [np.concatenate((tgt, [sep_id], src))]

        if masking_type == "t5":
            return T5Dataset.build_training_sample(
                sample=sample,
                target_seq_length=sample[0].shape[0],
                np_rng=np_rng,
                max_seq_length=max_seq_length,
                max_seq_length_dec=max_seq_length_dec,
                masked_lm_prob=masked_lm_prob,
                vocab_id_list=vocab_id_list,
                vocab_id_to_token_dict=vocab_id_to_token_dict,
                cls_id=cls_id,
                sep_id=sep_id,
                mask_id=mask_id,
                max_ngram_size=max_ngram_size,
                mean_ngram_size=mean_ngram_size,
                whole_word_masking=False,
                favor_long_ngrams=favor_long_ngrams,
                permutation=permutation,
                geometric_dist=geometric_dist,
                tokenizer_type=tokenizer_type,
                sentinel_tokens=sentinel_tokens,
                bos_id=bos_id,
                eos_id=eos_id,
                pad_id=pad_id,
                skip_masking_id=sep_id
            )
        elif masking_type == "ul2":
            assert extreme_masked_lm_prob is not None
            assert extreme_max_ngram_size is not None
            assert extreme_mean_ngram_size is not None
            assert extreme_min_ngram_size is not None
            assert extreme_ngram_span_length_distribution is not None

            return UL2Dataset.build_extreme_masking_training_sample(
                sample=sample,
                target_seq_length=sample[0].shape[0],
                np_rng=np_rng,
                max_seq_length=max_seq_length,
                max_seq_length_dec=max_seq_length_dec,
                masked_lm_prob=masked_lm_prob,
                extreme_masked_lm_prob=extreme_masked_lm_prob,
                mask_id=mask_id,
                max_ngram_size=max_ngram_size,
                mean_ngram_size=mean_ngram_size,
                min_ngram_size=min_ngram_size,
                extreme_max_ngram_size=extreme_max_ngram_size,
                extreme_mean_ngram_size=extreme_mean_ngram_size,
                extreme_min_ngram_size=extreme_min_ngram_size,
                extreme_ngram_span_length_distribution=extreme_ngram_span_length_distribution,
                sentinel_tokens=sentinel_tokens,
                bos_id=bos_id,
                eos_id=eos_id,
                pad_id=pad_id,
                skip_masking_id=sep_id
            )

    def __getitem__(self, idx):
        src, tgt = super()._get_sample(idx)
        if len(src) > self.max_src_seq_length:
            src = src[: self.max_src_seq_length]

        if len(tgt) > self.max_tgt_seq_length - 1: # -1 here to account for the <sep> token that gets added.
            tgt = tgt[: self.max_tgt_seq_length]

        np_rng = np.random.RandomState(seed=(self.seed + idx))

        return CrossLingualMakedSequenceToSequenceDataset.get_example(
            src=src,
            tgt=tgt,
            max_seq_length=self.max_seq_length,
            max_seq_length_dec=self.max_seq_length_dec,
            masked_lm_prob=self.masked_lm_prob,
            vocab_id_list=self.vocab_id_list,
            vocab_id_to_token_dict=self.vocab_id_to_token_dict,
            cls_id=self.cls_id,
            sep_id=self.sep_id,
            mask_id=self.mask_id,
            bos_id=self.bos_id,
            eos_id=self.eos_id,
            pad_id=self.pad_id,
            sentinel_tokens=self.sentinel_tokens,
            max_ngram_size=self.max_ngram_size,
            np_rng=np_rng,
            mean_ngram_size=self.mean_ngram_size,
            min_ngram_size=self.min_ngram_size,
            extreme_masked_lm_prob=self.extreme_masked_lm_prob,
            extreme_max_ngram_size=self.extreme_max_ngram_size,
            extreme_mean_ngram_size=self.extreme_mean_ngram_size,
            extreme_min_ngram_size=self.extreme_min_ngram_size,
            extreme_ngram_span_length_distribution=self.extreme_ngram_span_length_distribution,
            favor_long_ngrams=self.favor_long_ngrams,
            permutation=self.permutation,
            geometric_dist=self.geometric_dist,
            tokenizer_type=self.tokenizer_type,
            masking_type=self.masking_type
        )

    # Skip the parent collate function, since we don't need it for this dataset.
    def collate_fn(self, batch):
        return batch


class CrossLingualMLMAndTranslationDataset(BinarizedMemmapSequenceToSequenceDataset):
    def __init__(
        self,
        src_dataset_prefix: str,
        tgt_dataset_prefix: str,
        src_tokenizer: TokenizerSpec,
        tgt_tokenizer: TokenizerSpec,
        max_src_seq_length: int,
        max_tgt_seq_length: int,
        max_seq_length_dec: int,
        seed: int = 1234,
        max_num_samples: int = None,
        masked_lm_prob: float = 0.15,
        extreme_masked_lm_prob: float = 0.5,
        max_ngram_size: int = 10,
        mean_ngram_size: int = None,
        min_ngram_size: int = 1,
        extreme_max_ngram_size: int = 128,
        extreme_mean_ngram_size: int = 64,
        extreme_min_ngram_size: int = 32,
        extreme_ngram_span_length_distribution: LengthDistribution = LengthDistribution.truncated_normal,
        geometric_dist: bool = True,
        permutation: bool = False,
        favor_long_ngrams: bool = False,
        sampling_ratios: Dict[str, float] = {"t5": 0.25, "ul2": 0.25, "nmt": 0.5},
    ):
        super().__init__(
            src_dataset_prefix=src_dataset_prefix,
            tgt_dataset_prefix=tgt_dataset_prefix,
            src_tokenizer=src_tokenizer,
            tgt_tokenizer=tgt_tokenizer,
            max_src_seq_length=max_src_seq_length,
            max_tgt_seq_length=max_tgt_seq_length,
            seed=seed,
            max_num_samples=max_num_samples,
        )
        self.max_seq_length_dec = max_seq_length_dec
        self.max_ngram_size = max_ngram_size
        self.mean_ngram_size = mean_ngram_size
        self.min_ngram_size = min_ngram_size
        self.geometric_dist = geometric_dist
        self.permutation = permutation
        self.favor_long_ngrams = favor_long_ngrams
        self.extreme_masked_lm_prob = extreme_masked_lm_prob
        self.extreme_max_ngram_size = extreme_max_ngram_size
        self.extreme_mean_ngram_size = extreme_mean_ngram_size
        self.extreme_min_ngram_size = extreme_min_ngram_size
        self.extreme_ngram_span_length_distribution = extreme_ngram_span_length_distribution
        self.sampling_ratios = sampling_ratios

        assert src_tokenizer == tgt_tokenizer
        # Vocab stuff.
        self.vocab_id_list = src_tokenizer.vocab
        self.vocab_id_to_token_dict = {idx: token for idx, token in enumerate(self.vocab_id_list)}
        self.cls_id = src_tokenizer.cls_id
        self.sep_id = src_tokenizer.sep_id
        self.mask_id = src_tokenizer.mask_id
        self.pad_id = src_tokenizer.pad_id
        self.bos_id = src_tokenizer.bos_id
        self.eos_id = src_tokenizer.eos_id
        self.max_seq_length = max_src_seq_length + max_tgt_seq_length
        self.masked_lm_prob = masked_lm_prob

        self.tokenizer_type = T5Dataset._determine_tokenizer_type(src_tokenizer, whole_word_masking=False)
        self._build()

    def _build(self):
        """
        Class-specific build method to be overridden by child classes.
        """
        self.sentinel_tokens = self.src_tokenizer.additional_special_tokens_ids
        assert len(self.sentinel_tokens) > 0

    def __getitem__(self, idx):
        np_rng = np.random.RandomState(seed=(self.seed + idx))

        # Determine which task to perform - NMT/T5/UL2 based on sampling ratios.
        task = np_rng.choice(list(self.sampling_ratios.keys()), p=list(self.sampling_ratios.values()))

        # We can just use the parent's __getitem__ function for NMT.
        if task == "nmt":
            return super().__getitem__(idx)

        src, tgt = super()._get_sample(idx)
        if len(src) > self.max_src_seq_length:
            src = src[: self.max_src_seq_length]

        if len(tgt) > self.max_tgt_seq_length - 1: # -1 here to account for the <sep> token that gets added.
            tgt = tgt[: self.max_tgt_seq_length]

        return CrossLingualMakedSequenceToSequenceDataset.get_example(
            src=src,
            tgt=tgt,
            max_seq_length=self.max_seq_length,
            max_seq_length_dec=self.max_seq_length_dec,
            masked_lm_prob=self.masked_lm_prob,
            vocab_id_list=self.vocab_id_list,
            vocab_id_to_token_dict=self.vocab_id_to_token_dict,
            cls_id=self.cls_id,
            sep_id=self.sep_id,
            mask_id=self.mask_id,
            bos_id=self.bos_id,
            eos_id=self.eos_id,
            pad_id=self.pad_id,
            sentinel_tokens=self.sentinel_tokens,
            max_ngram_size=self.max_ngram_size,
            np_rng=np_rng,
            mean_ngram_size=self.mean_ngram_size,
            min_ngram_size=self.min_ngram_size,
            extreme_masked_lm_prob=self.extreme_masked_lm_prob,
            extreme_max_ngram_size=self.extreme_max_ngram_size,
            extreme_mean_ngram_size=self.extreme_mean_ngram_size,
            extreme_min_ngram_size=self.extreme_min_ngram_size,
            extreme_ngram_span_length_distribution=self.extreme_ngram_span_length_distribution,
            favor_long_ngrams=self.favor_long_ngrams,
            permutation=self.permutation,
            geometric_dist=self.geometric_dist,
            tokenizer_type=self.tokenizer_type,
            masking_type=task
        )
 