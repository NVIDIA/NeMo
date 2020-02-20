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

from nemo.collections.nlp.data import BertPunctuationCapitalizationDataset
from nemo.collections.nlp.nm.data_layers.text_datalayer import TextDataLayer
from nemo.core import ChannelType, LabelsType, MaskType, NeuralType
from nemo.utils.decorators import add_port_docs

__all__ = ['PunctuationCapitalizationDataLayer']


class PunctuationCapitalizationDataLayer(TextDataLayer):
    """
    Data layer for punctuation and capitalization.

    Args:
        text_file (str): file to sequences, each line should a sentence,
            No header.
        label_file (str): file to labels, each line corresponds to
            word labels for a sentence in the text_file. No header.
        tokenizer (TokenizerSpec): text tokenizer.
        max_seq_length (int): max sequence length minus 2 for [CLS] and [SEP]
        pad_label (str): ad value use for labels.
            by default, it's the neutral label.
        punct_label_ids (dict): 
            dict to map labels to label ids.
            Starts with pad_label->0 and then increases in alphabetical order
            For dev set use label_ids generated during training to support
            cases when not all labels are present in the dev set.
            For training set label_ids should be None.
        capit_label_ids (dict):
            dict to map labels to label ids.
            Starts with pad_label->0 and then increases in alphabetical order
            For dev set use label_ids generated during training to support
            cases when not all labels are present in the dev set.
            For training set label_ids should be None.
        num_samples (int):
            number of samples you want to use for the dataset.
                If -1, use all dataset. Useful for testing.
        shuffle (bool): whether to shuffle your data.
        batch_size (int): batch size
        ignore_extra_tokens (bool): whether to ignore extra tokens in
            the loss_mask
        ignore_start_end (bool):
            whether to ignore bos and eos tokens in the loss_mask
        use_cache (bool): whether to use data cache
        dataset_type (Dataset): Default BertPunctuationCapitalizationDataset.
    """

    @property
    @add_port_docs()
    def output_ports(self):
        """Returns definitions of module output ports.
        input_ids:
            indices of tokens which constitute batches of masked text segments
        input_type_ids:
            tensor with 0's and 1's to denote the text segment type
        input_mask:
            bool tensor with 0s in place of tokens to be masked
        loss_mask:
            used to mask and ignore tokens in the loss function: indices of tokens which constitute batches of unmasked text segments
        subtokens_mask:
            used to mask all but the first subtoken of the work, could be useful during inference
        punct_labels: punctuation label ids
        capit_labels: capit_labels label ids
        """
        return {
            "input_ids": NeuralType(('B', 'T'), ChannelType()),
            "input_type_ids": NeuralType(('B', 'T'), ChannelType()),
            "input_mask": NeuralType(('B', 'T'), ChannelType()),
            "loss_mask": NeuralType(('B', 'T'), MaskType()),
            "subtokens_mask": NeuralType(('B', 'T'), ChannelType()),
            "punct_labels": NeuralType(('B', 'T'), LabelsType()),
            "capit_labels": NeuralType(('B', 'T'), LabelsType()),
        }

    def __init__(
        self,
        text_file,
        label_file,
        tokenizer,
        max_seq_length,
        pad_label='O',
        punct_label_ids=None,
        capit_label_ids=None,
        num_samples=-1,
        shuffle=False,
        batch_size=64,
        ignore_extra_tokens=False,
        ignore_start_end=False,
        use_cache=False,
        dataset_type=BertPunctuationCapitalizationDataset,
    ):
        dataset_params = {
            'text_file': text_file,
            'label_file': label_file,
            'max_seq_length': max_seq_length,
            'tokenizer': tokenizer,
            'num_samples': num_samples,
            'pad_label': pad_label,
            'punct_label_ids': punct_label_ids,
            'capit_label_ids': capit_label_ids,
            'ignore_extra_tokens': ignore_extra_tokens,
            'ignore_start_end': ignore_start_end,
            'use_cache': use_cache,
        }
        super().__init__(dataset_type, dataset_params, batch_size, shuffle=shuffle)
