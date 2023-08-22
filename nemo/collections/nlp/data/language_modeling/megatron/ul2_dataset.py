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

"""UL2 Style dataset from https://arxiv.org/abs/2205.05131"""
import math

import numpy as np
import torch

from nemo.collections.common.tokenizers.tokenizer_spec import TokenizerSpec
from nemo.collections.nlp.data.language_modeling.megatron.dataset_utils import create_extreme_masked_lm_predictions
from nemo.collections.nlp.data.language_modeling.megatron.length_distribution_type import LengthDistribution
from nemo.collections.nlp.data.language_modeling.megatron.lm_adapted_t5_dataset import T5LMAdaptedDataset
from nemo.collections.nlp.data.language_modeling.megatron.t5_dataset import T5Dataset
import pdb


class UL2Dataset(T5Dataset):
    """ UL2 Dataset from https://arxiv.org/abs/2205.05131.
    Consists of three different objectives:
    1. Short span masking with small probabilities (ex: T5). Typically max ngram size of 5 with 0.15 mask prob.
    2. Extreme span masking with either large probabilities or large ngram sizes or both.
    3. Prefx-LM as in the T5 or LM-adapted T5 (prompt-tuning paper).
    """

    def __init__(
        self,
        cfg,
        trainer,
        tokenizer,
        name,
        indexed_dataset,
        data_prefix,
        num_epochs,
        max_num_samples,
        max_seq_length,
        max_seq_length_dec,
        seed,
        masked_lm_prob=0.15,
        extreme_masked_lm_prob=0.5,
        short_seq_prob=0.0,
        min_ngram_size=1,
        max_ngram_size=10,
        mean_ngram_size=3,
        extreme_max_ngram_size=128,
        extreme_min_ngram_size=32,
        extreme_mean_ngram_size=64,
        prefix_lm_pivot_distribution=LengthDistribution.uniform,
        prefix_lm_pivot_mean=0.25,  # This is represented as a percentage of the total length.
        ngram_span_length_distribution=LengthDistribution.geometric,
        extreme_ngram_span_length_distribution=LengthDistribution.truncated_normal,
        permutation=False,
        whole_word_masking=False,
        favor_long_ngrams=False,
        respect_document_boundaries=True,
        documents=None,
        skip_masking_id=None,
        sampling_probabilities={
            'causal': 0,
            'r-masking': 0.33,  # T5-style masking with short spans and small probabilities.
            's-masking': 0.33,  # Prefix-LM with a pivot point sampled based on prefix_lm_pivot_mean
            'x-masking-longspan-smallprob': 0.11,  # Extreme span masking with small probabilities and large ngram sizes.
            'x-masking-longspan-largeprob': 0.11,  # Extreme span masking with large probabilities and small ngram sizes.
            'x-masking-shortspan-largeprob': 0.11,  # Extreme span masking with large probabilities and small ngram sizes.
        },
        sentinel_tokens=None,
        shuffle_documents=True,
        debug=False,
        add_bos_to_enc=False,
        force_sep_tokens=False,
        use_v2_format=False
    ):
        """ Args:
        cfg: Omegaconf config object.
        trainer: Pytorch Lightning Trainer object.
        tokenizer: Tokenizer object.
        name: Name of the indexed dataset.
        indexed_dataset: Path to the indexed dataset.
        data_prefix: Prefix of the `.bin` and `.idx` files.
        num_epochs: Number of epochs to train for, otherwise uses max_num_samples.
        max_num_samples: Maximum number of training samples to use. Will upsample and shuffle the dataset as necessary.
        max_seq_length: Maximum sequence length for the encoder.
        max_seq_length_dec: Maximum sequence length for the decoder.
        seed: Random seed to use.
        masked_lm_prob: Probability of masking a token for r-masking.
        extreme_masked_lm_prob: Probability of masking a token for x-masking.
        short_seq_prob: Probability of using a short sequence. Typically used just for BERT.
        min_ngram_size: Minimum ngram size for r-masking.
        max_ngram_size: Maximum ngram size for r-masking.
        mean_ngram_size: Mean ngram size for r-masking.
        extreme_max_ngram_size: Maximum ngram size for x-masking.
        extreme_min_ngram_size: Minimum ngram size for x-masking.,
        extreme_mean_ngram_size: Mean ngram size for x-masking.,
        prefix_lm_pivot_mean: Fraction of the total input length to be used as the mean pivot point for s-masking.
        ngram_span_length_distribution: Distribution to use for sampling ngram sizes for r-masking.
        extreme_ngram_span_length_distribution: Distribution to use for sampling ngram sizes for x-masking.
        permutation: Whether to permute the order of the ngrams in the input.
        whole_word_masking: Whether to mask whole words. NOTE: Only works with the BERT wordpiece tokenizer.
        favor_long_ngrams: Whether to favor longer ngrams when sampling. (reverses the n-gram sampling probability distribution)
        respect_document_boundaries: Whether to respect document boundaries when constructing a single training example. If True will pad the document to max_seq_length.
        documents: np.arange of document indices to indicate the number of documents in the dataset.
        skip_masking_id: Token ID that will never be masked. Typically used to prevent masking the end-of-document token with respect_document_boundaries=False.
        sentinel_tokens: A list of Token IDs that correspond to mask tokens. If None, will add tokens to the tokenizer automatically.
        shuffle_documents: Whether to shuffle the documents before sampling when not respecting document boundaries.
        """
        super().__init__(
            cfg=cfg,
            trainer=trainer,
            tokenizer=tokenizer,
            name=name,
            indexed_dataset=indexed_dataset,
            data_prefix=data_prefix,
            num_epochs=num_epochs,
            max_num_samples=max_num_samples,
            max_seq_length=max_seq_length,  # do not need -1 even with the added mask type token due to causality
            max_seq_length_dec=max_seq_length_dec,
            seed=seed,
            masked_lm_prob=masked_lm_prob,
            short_seq_prob=short_seq_prob,
            max_ngram_size=max_ngram_size,
            mean_ngram_size=None,  # TODO: Determine if we want to actually pass mean ngram as an override to max here.
            geometric_dist=ngram_span_length_distribution == LengthDistribution.geometric,
            permutation=permutation,
            whole_word_masking=False,
            favor_long_ngrams=favor_long_ngrams,
            respect_document_boundaries=respect_document_boundaries,
            documents=documents,
            skip_masking_id=tokenizer.eos_id if skip_masking_id is None else skip_masking_id,
            sentinel_tokens=sentinel_tokens,
            shuffle_documents=shuffle_documents,
            force_sep_tokens=force_sep_tokens,
        )
        self.mean_ngram_size = mean_ngram_size
        self.min_ngram_size = min_ngram_size
        self.extreme_masked_lm_prob = extreme_masked_lm_prob
        self.extreme_min_ngram_size = extreme_min_ngram_size
        self.extreme_max_ngram_size = extreme_max_ngram_size
        self.extreme_mean_ngram_size = extreme_mean_ngram_size
        self.ngram_span_length_distribution = ngram_span_length_distribution
        self.extreme_ngram_span_length_distribution = extreme_ngram_span_length_distribution
        self.prefix_lm_pivot_mean = prefix_lm_pivot_mean
        self.sampling_probabilities = sampling_probabilities
        self.prefix_lm_pivot_distribution = prefix_lm_pivot_distribution

        self._normalize_sampling_probabilities()
        self.causal_prob = sampling_probabilities['causal']
        self.r_masking_prob = sampling_probabilities['r-masking']
        self.s_masking_prob = sampling_probabilities['s-masking']
        self.x_masking_prob = sum(
            [
                sampling_probabilities['x-masking-longspan-smallprob'],
                sampling_probabilities['x-masking-longspan-largeprob'],
                sampling_probabilities['x-masking-shortspan-largeprob'],
            ]
        )
        self.x_masking_longspan_smallprob = sampling_probabilities['x-masking-longspan-smallprob']
        self.x_masking_longspan_largeprob = sampling_probabilities['x-masking-longspan-largeprob']
        self.x_masking_shortspan_largeprob = sampling_probabilities['x-masking-shortspan-largeprob']

        self.debug = debug
        self.add_bos_to_enc = add_bos_to_enc
        self.use_v2_format = use_v2_format

    def _normalize_sampling_probabilities(self):
        """ Normalize the sampling probabilities to sum to 1. """
        total = sum(self.sampling_probabilities.values())
        for k in self.sampling_probabilities:
            self.sampling_probabilities[k] /= total

    @classmethod
    def get_r_masking_training_sample(
        cls,
        sample,
        tokenizer,
        np_rng,
        target_seq_length: int,
        max_seq_length: int,
        max_seq_length_dec: int,
        masked_lm_prob: float,
        vocab_id_list: list,
        vocab_id_to_token_dict: dict,
        max_ngram_size: int,
        mean_ngram_size: int,
        whole_word_masking: bool,
        favor_long_ngrams: bool,
        permutation: bool,
        geometric_dist: bool,
        tokenizer_type: str,
        sentinel_tokens: list,
        skip_masking_id: int,
        use_v2_format: bool = False,
    ):
        # Call T5's build training sample for regular short span masking.
        # Set the sequence length budget to be max_seq_length-1 to add the extra token.
        sample = T5Dataset.build_training_sample(
            sample=sample,
            target_seq_length=target_seq_length,
            np_rng=np_rng,
            max_seq_length=max_seq_length,
            max_seq_length_dec=max_seq_length_dec,
            masked_lm_prob=masked_lm_prob,
            vocab_id_list=vocab_id_list,
            vocab_id_to_token_dict=vocab_id_to_token_dict,
            cls_id=tokenizer.cls_id,
            sep_id=tokenizer.sep_id,
            mask_id=tokenizer.mask_id,
            max_ngram_size=max_ngram_size,
            mean_ngram_size=mean_ngram_size,
            whole_word_masking=whole_word_masking,
            favor_long_ngrams=favor_long_ngrams,
            permutation=permutation,
            geometric_dist=geometric_dist,
            tokenizer_type=tokenizer_type,
            sentinel_tokens=sentinel_tokens,
            bos_id=tokenizer.bos_id,
            eos_id=tokenizer.eos_id,
            pad_id=tokenizer.pad_id,
            skip_masking_id=skip_masking_id,
            use_v2_format=use_v2_format,
        )
        sample = UL2Dataset._prepend_mask_type_token(tokenizer, sample, '<extra_id_r>')
        return sample

    @classmethod
    def get_s_masking_training_sample(
        cls,
        sample,
        np_rng,
        max_seq_length_encoder: int,
        max_seq_length_decoder: int,
        tokenizer: TokenizerSpec,
        prefix_lm_pivot_mean: float,
        pivot_distribution: LengthDistribution,
        add_eos: bool = False,
        add_bos_to_enc: bool = False,
        debug: bool = False
    ):
        sample = [token for sentence in sample for token in sentence]
        # Leave space for the special tokens.
        # We only consider the decoder-only case, where we may add eos to the inputs, and prepend task token
        # if debug:
        #     pdb.set_trace()
        if max_seq_length_decoder is None:
            budget = int(add_bos_to_enc)
            sample = sample[: max_seq_length_encoder - budget]
        sample = T5LMAdaptedDataset.get_prefix_lm_sample(
            sample=sample,
            max_seq_length_encoder=max_seq_length_encoder,
            max_seq_length_decoder=max_seq_length_decoder,  # We don't use max_seq_length_decoder here since we typically want to use long decoder sequences for better LM performance and we can do +1 because we don't need to add the UL2 token here.
            np_rng=np_rng,
            tokenizer=tokenizer,
            pivot_mean=prefix_lm_pivot_mean,
            pivot_distribution=pivot_distribution,
            add_eos=add_eos,
            add_bos_to_enc=add_bos_to_enc,
            debug=debug,
        )
        sample = UL2Dataset._prepend_mask_type_token(tokenizer, sample, '<extra_id_s>')
        return sample

    @classmethod
    def get_x_masking_training_sample(
        cls,
        sample,
        tokenizer,
        np_rng,
        target_seq_length: int,
        max_seq_length: int,
        max_seq_length_dec: int,
        masked_lm_prob: float,
        extreme_masked_lm_prob: float,
        max_ngram_size: int,
        min_ngram_size: int,
        mean_ngram_size: int,
        extreme_max_ngram_size: int,
        extreme_min_ngram_size: int,
        extreme_mean_ngram_size: int,
        extreme_ngram_span_length_distribution: LengthDistribution,
        sentinel_tokens: list,
        skip_masking_id: int,
        masking_type: str,
        use_v2_format: bool = False,
    ):
        sample = UL2Dataset.build_extreme_masking_training_sample(
            sample=sample,
            target_seq_length=target_seq_length,
            np_rng=np_rng,
            max_seq_length=max_seq_length,
            max_seq_length_dec=max_seq_length_dec,
            masked_lm_prob=masked_lm_prob,
            extreme_masked_lm_prob=extreme_masked_lm_prob,
            mask_id=tokenizer.mask_id,
            max_ngram_size=max_ngram_size,
            min_ngram_size=min_ngram_size,
            extreme_max_ngram_size=extreme_max_ngram_size,
            extreme_mean_ngram_size=extreme_mean_ngram_size,
            extreme_min_ngram_size=extreme_min_ngram_size,
            extreme_ngram_span_length_distribution=extreme_ngram_span_length_distribution,
            mean_ngram_size=mean_ngram_size,
            sentinel_tokens=sentinel_tokens,
            bos_id=tokenizer.bos_id,
            eos_id=tokenizer.eos_id,
            pad_id=tokenizer.pad_id,
            skip_masking_id=skip_masking_id,
            masking_type=masking_type,
            use_v2_format=use_v2_format,
        )
        sample = UL2Dataset._prepend_mask_type_token(tokenizer, sample, '<extra_id_x>')
        return sample

    def _get_worst_case_seq_length(self, masking_prob, min_span_length, seq_length):
        """Returns the worst case sequence length for the given masking probability and minimum span length.

        Args:
        masking_prob: The masking probability.
        min_span_length: The minimum span length.
        seq_length: The sequence length.

        Returns:
        The worst case sequence length.
        """
        worst_case_added_tokens = int(math.ceil((seq_length * masking_prob) / (min_span_length)))
        worst_case_added_tokens *= (
            2  # The same sentinel token is added once in the input and once again in the completion.
        )
        return seq_length - worst_case_added_tokens

    def print_and_break(self, example):
        for tokens in [example['text_enc'], example['labels']]:
            tokens = torch.Tensor(tokens)
            seg_ids = torch.cumsum(tokens==3, 0)
            for i in range(seg_ids[-1].item() + 1):
                doc = tokens[seg_ids == i]
                text = self.tokenizer.ids_to_text(doc.long().cpu().numpy().tolist())
                print("========================================")
                print(i)
                print(text)
                print("========================================")
        import pdb; pdb.set_trace()

    def __getitem__(self, idx, decoder_only=False):
        """Returns a training sample for the given index.

        Returns:
        A dictionary with the following keys:
            text_enc: The tokenized text for the encoder.
            text_dec: The tokenized text for the decoder input.
            labels: The tokenized text for the decoder output.
            loss_mask: The loss mask for the decoder output.
            enc_mask: The padding mask corresponding to the encoder.
            dec_mask: The padding mask corresponding to the decoder.
        """
        sample, seq_length = self._get_sample(idx)
        # Note that this rng state should be numpy and not python since
        # python randint is inclusive whereas the numpy one is exclusive.
        np_rng = np.random.RandomState(seed=(self.seed + idx))
        masking_type = np_rng.choice(
            [
                'causal',
                'r-masking',
                's-masking',
                'x-masking-longspan-smallprob',
                'x-masking-longspan-largeprob',
                'x-masking-shortspan-largeprob',
            ],
            p=[
                self.causal_prob,
                self.r_masking_prob,
                self.s_masking_prob,
                self.x_masking_longspan_smallprob,
                self.x_masking_longspan_largeprob,
                self.x_masking_shortspan_largeprob,
            ],
        )  # r: short span masking, x: extreme masking, s: prefix-LM
        # We delibrately create sequences with length equal to max_length + 1
        # Later this will be consumed when we make autoregressive predictions.
        if masking_type == 'causal':
            s2s_token = self.tokenizer.text_to_ids('<extra_id_s>')
            example = {'text_enc': np.array(s2s_token, ), 'labels': sample[0]}
        elif masking_type == 'r-masking':
            # Call T5's build training sample for regular short span masking.
            # For GPT models, the insertion of sentinel tokens means that the sequence length can exceed the max_seq_length, so we need to adjust the target_seq_length for the worst case scenario accordingly.
            # target_seq_length = (
            #     # self._get_worst_case_seq_length(
            #     #     masking_prob=self.masked_lm_prob, min_span_length=1, seq_length=seq_length
            #     # )
            #     seq_length - int(self.masked_lm_prob * seq_length) - 1
            #     if decoder_only
            #     else seq_length
            # )
            target_seq_length = seq_length
            example = UL2Dataset.get_r_masking_training_sample(
                sample=sample,
                tokenizer=self.tokenizer,
                np_rng=np_rng,
                target_seq_length=target_seq_length,
                max_seq_length=self.max_seq_length,
                max_seq_length_dec=self.max_seq_length_dec,
                masked_lm_prob=self.masked_lm_prob,
                vocab_id_list=self.vocab_id_list,
                vocab_id_to_token_dict=self.vocab_id_to_token_dict,
                max_ngram_size=self.max_ngram_size,
                mean_ngram_size=self.mean_ngram_size,
                whole_word_masking=self.whole_word_masking,
                favor_long_ngrams=self.favor_long_ngrams,
                permutation=self.permutation,
                geometric_dist=self.geometric_dist,
                tokenizer_type=self.tokenizer_type,
                sentinel_tokens=self.sentinel_tokens,
                skip_masking_id=self.skip_masking_id,
                use_v2_format=self.use_v2_format,
            )
            example['masking_type'] = masking_type

            # self.print_and_break(example)
        elif masking_type == 's-masking':
            example = UL2Dataset.get_s_masking_training_sample(
                sample=sample,
                np_rng=np_rng,
                max_seq_length_encoder=self.max_seq_length,
                max_seq_length_decoder=self.max_seq_length_dec,
                tokenizer=self.tokenizer,
                prefix_lm_pivot_mean=self.prefix_lm_pivot_mean,
                pivot_distribution=self.prefix_lm_pivot_distribution,
                add_bos_to_enc=self.add_bos_to_enc,
                debug=self.debug
            )
            example['masking_type'] = masking_type

            # self.print_and_break(example)
        else:
            # Try to minimize the amount of padding based on the masking type for GPT models.
            # if masking_type == 'x-masking-longspan-smallprob':
            #     target_seq_length = (
            #         self._get_worst_case_seq_length(self.masked_lm_prob, self.extreme_min_ngram_size, seq_length)
            #         if decoder_only
            #         else seq_length
            #     )
            # elif masking_type == 'x-masking-shortspan-largeprob':
            #     target_seq_length = (
            #         self._get_worst_case_seq_length(self.extreme_masked_lm_prob, self.min_ngram_size, seq_length)
            #         if decoder_only
            #         else seq_length
            #     )
            # elif masking_type == 'x-masking-longspan-largeprob':
            #     target_seq_length = (
            #         self._get_worst_case_seq_length(
            #             self.extreme_masked_lm_prob, self.extreme_min_ngram_size, seq_length
            #         )
            #         if decoder_only
            #         else seq_length
            #     )
            target_seq_length = seq_length

            # For GPT models, the insertion of sentinel tokens means that the sequence length can exceed the max_seq_length, so we need to adjust the target_seq_length for the worst case scenario accordingly.
            example = UL2Dataset.get_x_masking_training_sample(
                sample=sample,
                tokenizer=self.tokenizer,
                np_rng=np_rng,
                target_seq_length=target_seq_length,
                max_seq_length=self.max_seq_length,
                max_seq_length_dec=self.max_seq_length_dec,
                masked_lm_prob=self.masked_lm_prob,
                extreme_masked_lm_prob=self.extreme_masked_lm_prob,
                max_ngram_size=self.max_ngram_size,
                min_ngram_size=self.min_ngram_size,
                mean_ngram_size=self.mean_ngram_size,
                extreme_max_ngram_size=self.extreme_max_ngram_size,
                extreme_min_ngram_size=self.extreme_min_ngram_size,
                extreme_mean_ngram_size=self.extreme_mean_ngram_size,
                extreme_ngram_span_length_distribution=self.extreme_ngram_span_length_distribution,
                sentinel_tokens=self.sentinel_tokens,
                skip_masking_id=self.skip_masking_id,
                masking_type=masking_type,
                use_v2_format=self.use_v2_format,
            )
            example['masking_type'] = masking_type

            # self.print_and_break(example)
        return example

    @classmethod
    def _prepend_mask_type_token(cls, tokenizer, sample, token):
        token_id = tokenizer.text_to_ids(token)
        assert len(token_id) == 1, token
        token_id = token_id[0]
        text_enc = np.concatenate([[token_id], sample['text_enc']])
        sample['text_enc'] = text_enc
        if 'enc_mask' in sample:
            sample['enc_mask'] = np.concatenate([[1], sample['enc_mask']])
        return sample

    @classmethod
    def build_extreme_masking_training_sample(
        cls,
        sample,
        target_seq_length,
        np_rng,
        max_seq_length,
        max_seq_length_dec,
        masked_lm_prob,
        extreme_masked_lm_prob,
        mask_id,
        max_ngram_size,
        min_ngram_size,
        mean_ngram_size,
        extreme_max_ngram_size,
        extreme_mean_ngram_size,
        extreme_min_ngram_size,
        extreme_ngram_span_length_distribution,
        sentinel_tokens,
        bos_id,
        eos_id,
        pad_id,
        masking_type,
        skip_masking_id=None,
        use_v2_format=False,
    ):
        """Build training sample.
        Arguments:
            sample: A list of sentences in which each sentence is a list token ids.
            target_seq_length: Desired sequence length.
            max_seq_length: Maximum length of the sequence. All values are padded to
                this length.
            vocab_id_list: List of vocabulary ids. Used to pick a random id.
            vocab_id_to_token_dict: A dictionary from vocab ids to text tokens.
            cls_id: Start of example id.
            sep_id: Separator id.
            mask_id: Mask token id.
            pad_id: Padding token id.
            masked_lm_prob: Probability to mask tokens.
            np_rng: Random number genenrator. Note that this rng state should be
                  numpy and not python since python randint is inclusive for
                  the opper bound whereas the numpy one is exclusive.
            bos_id: start of decoder example id
            eos_id: end of generation id
            sentinel_tokens: unique value to be substituted for every replaced span
            tokenizer_type: wordpiece (BERT-style) or sentencepiece tokenizer. Used for whole word masking logic.
            max_ngram_size: maximum size of ngrams to be masked.
            mean_ngram_size: mean size of ngrams to be masked (only used if geometric_dist=True).
            geometric_dist: Uses a geometric distribution to sample ngram size.
            permutation: Permutes the ngrams.
            whole_word_masking: Always masks entire words instead of individual sub-word tokens.
            favor_long_ngrams: Favor longer ngrams over shorter ones.
            masking_type: Options: ['shortspan-largeprob', 'longspan-smallprob', 'longspan-largeprob']
            skip_masking_id: id of the token to that will never be masked.
        """
        assert target_seq_length <= max_seq_length

        # flatten sentences into one list
        tokens = [token for sentence in sample for token in sentence]

        # Truncate to `target_sequence_length`.
        max_num_tokens = target_seq_length
        tokens = tokens[:max_num_tokens]

        # Determine if we have a lot of masking or little masking. There are three cases:
        # 1. Small masking prob, large spans.
        # 2. Large masking prob, small spans.
        # 3. Large masking prob, large spans.
        if masking_type == 'x-masking-longspan-smallprob':
            # Long spans, small masking prob
            max_ngram_size, mean_ngram_size, min_ngram_size, masked_lm_prob = (
                extreme_max_ngram_size,
                extreme_mean_ngram_size,
                extreme_min_ngram_size,
                masked_lm_prob,
            )
        elif masking_type == 'x-masking-shortspan-largeprob':
            # Short spans, large masking prob
            max_ngram_size, mean_ngram_size, min_ngram_size, masked_lm_prob = (
                max_ngram_size,
                mean_ngram_size,
                min_ngram_size,
                extreme_masked_lm_prob,
            )
        elif masking_type == 'x-masking-longspan-largeprob':
            # Long spans, large masking prob
            max_ngram_size, mean_ngram_size, min_ngram_size, masked_lm_prob = (
                extreme_max_ngram_size,
                extreme_mean_ngram_size,
                extreme_mean_ngram_size,
                extreme_masked_lm_prob,
            )
        else:
            raise ValueError(
                f'Invalid masking type {masking_type}. Must be one of [x-masking-longspan-smallprob, x-masking-shortspan-largeprob, x-masking-longspan-largeprob]'
            )

        # Masking.
        max_predictions_per_seq = masked_lm_prob * max_num_tokens

        lm_pred = create_extreme_masked_lm_predictions(
            tokens=tokens,
            masked_lm_prob=masked_lm_prob,
            mask_id=mask_id,
            max_predictions_per_seq=max_predictions_per_seq,
            np_rng=np_rng,
            max_ngram_size=max_ngram_size,
            min_ngram_size=min_ngram_size,
            mean_ngram_size=mean_ngram_size,
            span_length_distribution=extreme_ngram_span_length_distribution,
            skip_masking_id=skip_masking_id,
        )

        if masked_lm_prob == 0:
            (output_tokens, masked_positions, masked_labels) = lm_pred
            masked_spans = None
        else:
            (output_tokens, masked_positions, masked_labels, masked_spans) = lm_pred

        # Padding.
        if use_v2_format:
            pand_and_convert_fn = T5Dataset.pad_and_convert_to_numpy_v2
        else:
            pand_and_convert_fn = T5Dataset.pad_and_convert_to_numpy
        tokens_enc, tokens_dec_in, labels, enc_mask, dec_mask, loss_mask = pand_and_convert_fn(
            output_tokens=output_tokens,
            masked_positions=masked_positions,
            masked_labels=masked_labels,
            masked_spans=masked_spans,
            sentinel_tokens=sentinel_tokens,
            bos_id=bos_id,
            eos_id=eos_id,
            pad_id=pad_id,
            max_seq_length=max_seq_length,
            max_seq_length_dec=max_seq_length_dec,
        )

        train_sample = {
            'text_enc': tokens_enc,
            'text_dec': tokens_dec_in,
            'labels': labels,
            'loss_mask': loss_mask,
            'enc_mask': enc_mask,
            'dec_mask': dec_mask,
        }

        return train_sample


class UGPTDataset(UL2Dataset):
    """ UL2 Dataset for decoder-only models from https://arxiv.org/abs/2205.05131.
    Consists of three different objectives:
    1. Short span masking with small probabilities (ex: T5). Typically max ngram size of 5 with 0.15 mask prob.
    2. Extreme span masking with either large probabilities or large ngram sizes or both.
    3. Prefx-LM as in the T5 or LM-adapted T5 (prompt-tuning paper).
    """

    def __init__(
        self,
        cfg,
        trainer,
        tokenizer,
        name,
        indexed_dataset,
        data_prefix,
        num_epochs,
        max_num_samples,
        max_seq_length,
        max_seq_length_dec,
        seed,
        masked_lm_prob=0.15,
        extreme_masked_lm_prob=0.5,
        short_seq_prob=0.0,
        min_ngram_size=3,  # Set it to 3 here so that we don't end up padding a lot in extreme masking with short spans and large probs.
        max_ngram_size=10,
        mean_ngram_size=3,
        extreme_max_ngram_size=128,
        extreme_min_ngram_size=32,
        extreme_mean_ngram_size=64,
        prefix_lm_pivot_distribution=LengthDistribution.uniform,
        prefix_lm_pivot_mean=0.25,  # This is represented as a percentage of the total length.
        ngram_span_length_distribution=LengthDistribution.geometric,
        extreme_ngram_span_length_distribution=LengthDistribution.truncated_normal,
        permutation=False,
        whole_word_masking=False,
        favor_long_ngrams=False,
        respect_document_boundaries=False,
        documents=None,
        skip_masking_id=None,
        sampling_probabilities={
            'causal': 0,
            'r-masking': 0.33,  # T5-style masking with short spans and small probabilities.
            's-masking': 0.33,  # Prefix-LM with a pivot point sampled based on prefix_lm_pivot_mean
            'x-masking-longspan-smallprob': 0.11,  # Extreme span masking with small probabilities and large ngram sizes.
            'x-masking-longspan-largeprob': 0.11,  # Extreme span masking with large probabilities and small ngram sizes.
            'x-masking-shortspan-largeprob': 0.11,  # Extreme span masking with large probabilities and small ngram sizes.
        },
        use_prefix_noncausal_mask=True,
        sentinel_tokens=None,
        shuffle_documents=True,
        add_bos_to_enc=False,
        debug=False,
        force_sep_tokens=False,
        use_v2_format=False,
    ):
        """ Args:
        cfg: Omegaconf config object.
        trainer: Pytorch Lightning Trainer object.
        tokenizer: Tokenizer object.
        name: Name of the indexed dataset.
        indexed_dataset: Path to the indexed dataset.
        data_prefix: Prefix of the `.bin` and `.idx` files.
        num_epochs: Number of epochs to train for, otherwise uses max_num_samples.
        max_num_samples: Maximum number of training samples to use. Will upsample and shuffle the dataset as necessary.
        max_seq_length: Maximum sequence length for the encoder.
        max_seq_length_dec: Maximum sequence length for the decoder.
        seed: Random seed to use.
        masked_lm_prob: Probability of masking a token for r-masking.
        extreme_masked_lm_prob: Probability of masking a token for x-masking.
        short_seq_prob: Probability of using a short sequence. Typically used just for BERT.
        min_ngram_size: Minimum ngram size for r-masking.
        max_ngram_size: Maximum ngram size for r-masking.
        mean_ngram_size: Mean ngram size for r-masking.
        extreme_max_ngram_size: Maximum ngram size for x-masking.
        extreme_min_ngram_size: Minimum ngram size for x-masking.,
        extreme_mean_ngram_size: Mean ngram size for x-masking.,
        prefix_lm_pivot_mean: Fraction of the total input length to be used as the mean pivot point for s-masking.
        ngram_span_length_distribution: Distribution to use for sampling ngram sizes for r-masking.
        extreme_ngram_span_length_distribution: Distribution to use for sampling ngram sizes for x-masking.
        permutation: Whether to permute the order of the ngrams in the input.
        whole_word_masking: Whether to mask whole words. NOTE: Only works with the BERT wordpiece tokenizer.
        favor_long_ngrams: Whether to favor longer ngrams when sampling. (reverses the n-gram sampling probability distribution)
        respect_document_boundaries: Whether to respect document boundaries when constructing a single training example. If True will pad the document to max_seq_length.
        documents: np.arange of document indices to indicate the number of documents in the dataset.
        skip_masking_id: Token ID that will never be masked. Typically used to prevent masking the end-of-document token with respect_document_boundaries=False.
        use_prefix_noncausal_mask: Whether to use a non-causal mask over the prefix for all masking types.
        sentinel_tokens: A list of Token IDs that correspond to mask tokens. If None, will add tokens to the tokenizer automatically.
        shuffle_documents: Whether to shuffle the documents before sampling when not respecting document boundaries.
        """

        # Override MAX_SEQ_LENGTH_DELTA for GPT which doesnt need the addition of 2 tokens
        self.original_MAX_SEQ_LENGTH_DELTA = self.MAX_SEQ_LENGTH_DELTA
        # self.MAX_SEQ_LENGTH_DELTA = 0

        super().__init__(
            cfg=cfg,
            trainer=trainer,
            tokenizer=tokenizer,
            name=name,
            indexed_dataset=indexed_dataset,
            data_prefix=data_prefix,
            num_epochs=num_epochs,
            max_num_samples=max_num_samples,
            max_seq_length=max_seq_length + 1,  # +1 to account for the fact that compared to T5/UL2 we don't need to account for separate BOS/EOS.
            max_seq_length_dec=max_seq_length_dec,
            seed=seed,
            masked_lm_prob=masked_lm_prob,
            short_seq_prob=short_seq_prob,
            max_ngram_size=max_ngram_size,
            mean_ngram_size=None,  # TODO: Determine if we want to actually pass mean ngram as an override to max here.
            permutation=permutation,
            whole_word_masking=whole_word_masking,
            favor_long_ngrams=favor_long_ngrams,
            respect_document_boundaries=respect_document_boundaries,
            documents=documents,
            extreme_masked_lm_prob=extreme_masked_lm_prob,
            min_ngram_size=min_ngram_size,
            extreme_max_ngram_size=extreme_max_ngram_size,
            extreme_min_ngram_size=extreme_min_ngram_size,
            extreme_mean_ngram_size=extreme_mean_ngram_size,
            prefix_lm_pivot_mean=prefix_lm_pivot_mean,
            prefix_lm_pivot_distribution=prefix_lm_pivot_distribution,
            ngram_span_length_distribution=ngram_span_length_distribution,
            extreme_ngram_span_length_distribution=extreme_ngram_span_length_distribution,
            skip_masking_id=tokenizer.eos_id if skip_masking_id is None else skip_masking_id,
            sampling_probabilities=sampling_probabilities,
            sentinel_tokens=sentinel_tokens,
            shuffle_documents=shuffle_documents,
            add_bos_to_enc=add_bos_to_enc,
            debug=debug,
            force_sep_tokens=force_sep_tokens,
            use_v2_format=use_v2_format,
        )
        # Reset max seq length to the actual length
        self.max_seq_length = max_seq_length
        self.use_prefix_noncausal_mask = use_prefix_noncausal_mask

        # Restore MAX_SEQ_LENGTH_DELTA
        self.MAX_SEQ_LENGTH_DELTA = self.original_MAX_SEQ_LENGTH_DELTA

    def print_and_break_final(self, example, orig_example):
        keys = ['tokens', 'labels']
        inputs = self.tokenizer.ids_to_text(orig_example['text_enc'])
        targets = self.tokenizer.ids_to_text(example['labels'][example['loss_mask']==1].tolist())
        print('====================== inputs:\n', inputs)
        print('====================== targets:\n', targets)
        # for key in keys:
        #     print(f"==========={key}==========")
        #     tokens = example[key]
        #     seg_ids = torch.cumsum(tokens==3, 0)
        #     for i in range(seg_ids[-1].item() + 1):
        #         doc = tokens[seg_ids == i]
        #         text = self.tokenizer.ids_to_text(doc.long().cpu().numpy().tolist())
        #         print("========================================")
        #         print(i)
        #         print(text)
        #         print("========================================")
        import pdb; pdb.set_trace()

    def __getitem__(self, idx):
        example = super().__getitem__(idx, decoder_only=True)
        # if not self.respect_document_boundaries:
        #     assert (example['text_enc'] == self.tokenizer.pad_id).sum() == 0, 'Padding token found in encoder input.'
        #     assert (example['text_dec'] == self.tokenizer.pad_id).sum() == 0, 'Padding token found in decoder input.'
        #     assert (example['labels'] == self.tokenizer.pad_id).sum() == 0, 'Padding token found in labels.'
        # Adapt the example to the UGPT format.
        tokens = np.concatenate([example['text_enc'], example['labels']])
        # import pdb; pdb.set_trace()
        # if self.debug:
        #     import pdb; pdb.set_trace()
        inputs = tokens[:-1]
        labels = tokens[1:]
        assert (
            len(inputs) <= self.max_seq_length
        ), f'Input length {len(inputs)} exceeds max_seq_length {self.max_seq_length} for masking type {example["masking_type"]}'

        if len(inputs) < self.max_seq_length:
            inputs = np.concatenate([inputs, [self.tokenizer.pad_id] * (self.max_seq_length - len(inputs))])
            labels = np.concatenate([labels, [self.tokenizer.pad_id] * (self.max_seq_length - len(labels))])
        loss_mask = (
            [0] * (len(example['text_enc']) - 1)
            + [1] * (len(example['labels']))
            + [0] * (self.max_seq_length - len(example['text_enc']) - len(example['labels']) + 1)
        )
        # Start with a low`er triangular mask.
        attention_mask = torch.tril(torch.ones((len(inputs), len(inputs)))).unsqueeze(0)
        if self.use_prefix_noncausal_mask:
            # If using bidrectional masking, set the mask of the prefix to 1.
            attention_mask[:, :, : len(example['text_enc'])] = 1.0

        attention_mask = attention_mask < 0.5

        ret_batch = {
            'tokens': torch.LongTensor(inputs),
            'labels': torch.LongTensor(labels),
            'attention_mask': attention_mask,
            'loss_mask': torch.FloatTensor(loss_mask),
            'position_ids': np.arange(len(inputs)),
        }
        # self.print_and_break_final(ret_batch, example)
        if idx == 0:
            print("====================== Sanity check of input examples ==================")
            print(ret_batch)
            print(example['text_enc'])
        # import pdb; pdb.set_trace()
        return ret_batch
