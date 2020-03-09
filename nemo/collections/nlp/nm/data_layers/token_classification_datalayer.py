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

from nemo.collections.nlp.data import BertTokenClassificationDataset, BertTokenClassificationInferDataset
from nemo.collections.nlp.nm.data_layers.text_datalayer import TextDataLayer
from nemo.core import ChannelType, LabelsType, MaskType, NeuralType
from nemo.utils.decorators import add_port_docs

__all__ = ['BertTokenClassificationDataLayer', 'BertTokenClassificationInferDataLayer']


class BertTokenClassificationDataLayer(TextDataLayer):
    """
    Creates the data layer to use for the task of token classification
    with pretrained model.

    All the data processing is done BertTokenClassificationDataset.
        text_file (str):
            file to sequences, each line should a sentence,
            No header.
        label_file (str):
            file to labels, each line corresponds to word labels for a sentence in the text_file. No header.
        pad_label (int):
            d value use for labels.
            by default, it's the neutral label.
        tokenizer (TokenizerSpec): text tokenizer.
        max_seq_length (int):
            max sequence length minus 2 for [CLS] and [SEP]
        label_ids:
            dict to map labels to label ids.
            Starts with pad_label->0 and then increases in alphabetical order
            For dev set use label_ids generated during training to support
            cases when not all labels are present in the dev set.
            For training set label_ids should be None.
        num_samples (int): 
            number of samples you want to use for the dataset.
                If -1, use all dataset. Useful for testing.
        shuffle (bool): whether to shuffle data or not. Default: False.
        batch_size (int): text segments batch size
        ignore_extra_tokens (bool): whether or not to ignore extra tokens
        ignore_start_end (bool): whether or not to ignore start and end
        use_cache:
            whether to use data cache
        dataset_type (BertTokenClassificationDataset):
            the dataset that needs to be converted to DataLayerNM
    """

    @property
    @add_port_docs()
    def output_ports(self):
        """Returns definitions of module output ports.
        input_ids:
            indices of tokens which constitute batches of text segments
        input_type_ids:
            tensor with 0's and 1's to denote the text segment type
        input_mask:
            bool tensor with 0s in place of tokens to be masked
        loss_mask:
            used to mask and ignore tokens in the loss function
        subtokens_mask:
            used to mask all but the first subtoken of the work, could be useful during inference
        labels:
            token target ids
        """
        return {
            "input_ids": NeuralType(('B', 'T'), ChannelType()),
            "input_type_ids": NeuralType(('B', 'T'), ChannelType()),
            "input_mask": NeuralType(('B', 'T'), ChannelType()),
            "loss_mask": NeuralType(('B', 'T'), MaskType()),
            "subtokens_mask": NeuralType(('B', 'T'), ChannelType()),
            "labels": NeuralType(('B', 'T'), LabelsType()),
        }

    def __init__(
        self,
        text_file,
        label_file,
        tokenizer,
        max_seq_length,
        pad_label='O',
        label_ids=None,
        num_samples=-1,
        shuffle=False,
        batch_size=64,
        ignore_extra_tokens=False,
        ignore_start_end=False,
        use_cache=False,
        dataset_type=BertTokenClassificationDataset,
    ):
        dataset_params = {
            'text_file': text_file,
            'label_file': label_file,
            'max_seq_length': max_seq_length,
            'tokenizer': tokenizer,
            'num_samples': num_samples,
            'pad_label': pad_label,
            'label_ids': label_ids,
            'ignore_extra_tokens': ignore_extra_tokens,
            'ignore_start_end': ignore_start_end,
            'use_cache': use_cache,
        }
        super().__init__(dataset_type, dataset_params, batch_size, shuffle)


class BertTokenClassificationInferDataLayer(TextDataLayer):
    """
    All the data processing is done BertTokenClassificationInferDataset.
        queries:
            (list of str): quiries to run inference on
        tokenizer (TokenizerSpec): text tokenizer.
        max_seq_length (int):
            max sequence length minus 2 for [CLS] and [SEP]
        shuffle (bool): whether to shuffle data or not. Default: False.
        batch_size: text segments batch size
        dataset_type (BertTokenClassificationInferDataset):
            the dataset that needs to be converted to DataLayerNM
    """

    @property
    @add_port_docs()
    def output_ports(self):
        """Returns definitions of module output ports.
        """
        return {
            "input_ids": NeuralType(('B', 'T'), ChannelType()),
            "input_type_ids": NeuralType(('B', 'T'), ChannelType()),
            "input_mask": NeuralType(('B', 'T'), ChannelType()),
            "loss_mask": NeuralType(('B', 'T'), ChannelType()),
            "subtokens_mask": NeuralType(('B', 'T'), ChannelType()),
        }

    def __init__(
        self,
        queries,
        tokenizer,
        max_seq_length,
        batch_size=1,
        shuffle=False,
        dataset_type=BertTokenClassificationInferDataset,
    ):
        dataset_params = {'queries': queries, 'tokenizer': tokenizer, 'max_seq_length': max_seq_length}
        super().__init__(dataset_type, dataset_params, batch_size, shuffle=shuffle)
