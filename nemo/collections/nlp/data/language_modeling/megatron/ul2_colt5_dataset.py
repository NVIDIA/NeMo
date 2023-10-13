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

"""UL2 Style dataset from https://arxiv.org/abs/2205.05131"""
import numpy as np

from nemo.collections.common.tokenizers.tokenizer_spec import TokenizerSpec
from nemo.collections.nlp.data.language_modeling.megatron.dataset_utils import create_extreme_masked_lm_predictions
from nemo.collections.nlp.data.language_modeling.megatron.length_distribution_type import LengthDistribution
from nemo.collections.nlp.data.language_modeling.megatron.lm_adapted_t5_dataset import T5LMAdaptedDataset
from nemo.collections.nlp.data.language_modeling.megatron.ul2_dataset import UL2Dataset


class UL2CoLT5Dataset(UL2Dataset):
    """ 
    Modification of the UL2 dataset similar toCoLT5
    """
    def __getitem__(self, idx):
        """
        0. span corruption with noise rate 0.15 and average span 3 [R-denoiser]
        1. span corruption with noise rate 0.15 and average span 8 [R-denoiser]
        2. span corruption with noise rate 0.15 and average span 64 [R/X-denoiser, because of the span len]
        3. prefix-LM with noise rate 0.5 [S-denoiser]
        """
        sample, seq_length = self._get_sample(idx)

        # Note that this rng state should be numpy and not python since
        # python randint is inclusive whereas the numpy one is exclusive.
        np_rng = np.random.RandomState(seed=(self.seed + idx))
        masking_type = np_rng.randint(0, 4)

        if masking_type == 0:
            # Call T5's build training sample for regular short span masking.
            max_ngram_size = 7
            mean_ngram_size = 3
            return UL2Dataset.get_r_masking_training_sample(
                sample=sample,
                tokenizer=self.tokenizer,
                np_rng=np_rng,
                target_seq_length=seq_length,
                max_seq_length=self.max_seq_length,
                max_seq_length_dec=self.max_seq_length_dec,
                masked_lm_prob=self.masked_lm_prob,
                vocab_id_list=self.vocab_id_list,
                vocab_id_to_token_dict=self.vocab_id_to_token_dict,
                max_ngram_size=max_ngram_size,
                mean_ngram_size=mean_ngram_size,
                whole_word_masking=self.whole_word_masking,
                favor_long_ngrams=self.favor_long_ngrams,
                permutation=self.permutation,
                geometric_dist=self.geometric_dist,
                tokenizer_type=self.tokenizer_type,
                sentinel_tokens=self.sentinel_tokens,
                skip_masking_id=None,
            )
        elif masking_type == 1:
            # Call T5's build training sample for regular short span masking.
            max_ngram_size = 20
            mean_ngram_size = 9
            favor_long_ngrams = False
            return UL2Dataset.get_r_masking_training_sample(
                sample=sample,
                tokenizer=self.tokenizer,
                np_rng=np_rng,
                target_seq_length=seq_length,
                max_seq_length=self.max_seq_length,
                max_seq_length_dec=self.max_seq_length_dec,
                masked_lm_prob=self.masked_lm_prob,
                vocab_id_list=self.vocab_id_list,
                vocab_id_to_token_dict=self.vocab_id_to_token_dict,
                max_ngram_size=max_ngram_size,
                mean_ngram_size=mean_ngram_size,
                whole_word_masking=self.whole_word_masking,
                favor_long_ngrams=favor_long_ngrams,
                permutation=self.permutation,
                geometric_dist=self.geometric_dist,
                tokenizer_type=self.tokenizer_type,
                sentinel_tokens=self.sentinel_tokens,
                skip_masking_id=None,
            )
        elif masking_type == 2:
            return UL2Dataset.get_x_masking_training_sample(
                sample=sample,
                tokenizer=self.tokenizer,
                np_rng=np_rng,
                target_seq_length=seq_length,
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
                skip_masking_id=None,
                task_type=0,
            )
        else:
            return UL2Dataset.get_s_masking_training_sample(
                sample=sample,
                np_rng=np_rng,
                max_seq_length_encoder=self.max_seq_length,
                max_seq_length_decoder=self.max_seq_length_dec,
                tokenizer=self.tokenizer,
                prefix_lm_pivot_mean=self.prefix_lm_pivot_mean,
                pivot_distribution=self.extreme_ngram_span_length_distribution,
            )
            