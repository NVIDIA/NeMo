# =============================================================================
# Copyright 2020 NVIDIA. All Rights Reserved.
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
# =============================================================================

from nemo.collections.nlp.data.datasets.lasertagger_dataset import LaserTaggerDataset
from nemo.collections.nlp.nm.data_layers.text_datalayer import TextDataLayer
from nemo.core import ChannelType, LabelsType, MaskType, NeuralType
from nemo.utils.decorators import add_port_docs

__all__ = ['LaserTaggerDataLayer']


class LaserTaggerDataLayer(TextDataLayer):
    """
	Data layer for LaserTagger from source (src) to target (tgt) editing tasks.

	Args:
		tokenizer_src (TokenizerSpec): source language tokenizer
		tokenizer_tgt (TokenizerSpec): target language tokenizer
		dataset_src (str): path to source data
		dataset_tgt (str): path to target data
		tokens_in_batch (int): maximum allowed number of tokens in batches,
			batches will be constructed to minimize the use of <pad> tokens
		clean (bool): whether to use parallel data cleaning such as removing
			pairs with big difference in sentences length, removing pairs with
			the same tokens in src and tgt, etc; useful for training data layer
			and should not be used in evaluation data layer
		dataset_type (Dataset):
				the underlying dataset. Default: TranslationDataset
	"""

    @property
    @add_port_docs()
    def output_ports(self):
        """Returns definitions of module output ports.

		input_ids: indices of tokens which constitute batches of masked text segments
		input_mask: bool tensor with 0s in place of source tokens to be masked
		segment_ids: bool tensor with 0's and 1's to denote the text segment type
		labels_mask: bool tensor with 0s in place of label tokens to be masked
		labels: indices of tokens which should be predicted from each of the
			corresponding target tokens in tgt_ids
		sent_ids: indices of the sentences in a batch; important for
			evaluation with external metrics, such as SacreBLEU

		"""
        return {
            "input_ids": NeuralType(('B', 'T'), ChannelType()),
            "input_mask": NeuralType(('B', 'T'), ChannelType()),
            "segment_ids": NeuralType(('B', 'T'), ChannelType()),
            "tgt_ids": NeuralType(('B', 'T'), ChannelType()),
            "labels_mask": NeuralType(('B', 'T'), ChannelType()),
            "labels": NeuralType(('B', 'T'), LabelsType()),
            "loss_mask": NeuralType(('B', 'T'), MaskType()),
            "src_ids": NeuralType(('B', 'T'), ChannelType()),
            "src_first_tokens": NeuralType(('B', 'T'), ChannelType()),
        }

    def __init__(
        self,
        preprocessed_data,
        tokenizer,
        num_examples,
        batch_size,
        infer,
        shuffle=False,
        dataset_type=LaserTaggerDataset,
    ):
        dataset_params = {
            'preprocessed_data': preprocessed_data,
            'tokenizer': tokenizer,
            'num_examples': num_examples,
            'infer': infer,
        }
        super().__init__(dataset_type, dataset_params, batch_size=batch_size, shuffle=shuffle)
