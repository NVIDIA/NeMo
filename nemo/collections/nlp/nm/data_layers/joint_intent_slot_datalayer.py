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

from nemo.collections.nlp.data import BertJointIntentSlotDataset, BertJointIntentSlotInferDataset
from nemo.collections.nlp.nm.data_layers.text_datalayer import TextDataLayer
from nemo.core import ChannelType, LabelsType, MaskType, NeuralType
from nemo.utils.decorators import add_port_docs

__all__ = ['BertJointIntentSlotDataLayer', 'BertJointIntentSlotInferDataLayer']


class BertJointIntentSlotDataLayer(TextDataLayer):
    """
    Creates the data layer to use for the task of joint intent
    and slot classification with pretrained model.

    All the data processing is done in BertJointIntentSlotDataset.

    Args:
        input_file (str):
            data file
        slot_file (str):
            file to slot labels, each line corresponding to
            slot labels for a sentence in input_file. No header.
        pad_label (int): pad value use for slot labels
        tokenizer (TokenizerSpec): text tokenizer.
        max_seq_length (int):
            max sequence length minus 2 for [CLS] and [SEP]
        dataset_type (BertJointIntentSlotDataset):
            the dataset that needs to be converted to DataLayerNM
        shuffle (bool): whether to shuffle data or not. Default: False.
        batch_size: text segments batch size
        ignore_extra_tokens (bool): whether or not to ignore extra tokens
        ignore_start_end (bool)": whether or not to ignore start and end
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
            used to ignore the outputs of unwanted tokens in
            the inference and evaluation like the start and end tokens
        intents:
            intents labels
        slots:
            slots labels
        """
        return {
            "input_ids": NeuralType(('B', 'T'), ChannelType()),
            "input_type_ids": NeuralType(('B', 'T'), ChannelType()),
            "input_mask": NeuralType(('B', 'T'), ChannelType()),
            "loss_mask": NeuralType(('B', 'T'), MaskType()),
            "subtokens_mask": NeuralType(('B', 'T'), ChannelType()),
            "intents": NeuralType(tuple('B'), LabelsType()),
            "slots": NeuralType(('B', 'T'), LabelsType()),
        }

    def __init__(
        self,
        input_file,
        slot_file,
        pad_label,
        tokenizer,
        max_seq_length,
        num_samples=-1,
        shuffle=False,
        batch_size=64,
        ignore_extra_tokens=False,
        ignore_start_end=False,
        do_lower_case=False,
        dataset_type=BertJointIntentSlotDataset,
    ):
        dataset_params = {
            'input_file': input_file,
            'slot_file': slot_file,
            'pad_label': pad_label,
            'tokenizer': tokenizer,
            'max_seq_length': max_seq_length,
            'num_samples': num_samples,
            'ignore_extra_tokens': ignore_extra_tokens,
            'ignore_start_end': ignore_start_end,
            'do_lower_case': do_lower_case,
        }
        super().__init__(dataset_type, dataset_params, batch_size, shuffle=shuffle)


class BertJointIntentSlotInferDataLayer(TextDataLayer):
    """
    Creates the data layer to use for the task of joint intent
    and slot classification with pretrained model. This is for

    All the data processing is done in BertJointIntentSlotInferDataset.

    Args:
        queries (list): list of queries for inference
        tokenizer (TokenizerSpec): text tokenizer.
        max_seq_length (int):
            max sequence length minus 2 for [CLS] and [SEP]
        dataset_type (BertJointIntentSlotDataset):
            the dataset that needs to be converted to DataLayerNM
        shuffle (bool): whether to shuffle data or not. Default: False.
        do_lower_case (bool): whether to make the sentence all lower case
        batch_size: text segments batch size
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
            used to ignore the outputs of unwanted tokens in
            the inference and evaluation like the start and end tokens
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
        do_lower_case=False,
        dataset_type=BertJointIntentSlotInferDataset,
    ):
        dataset_params = {
            'queries': queries,
            'tokenizer': tokenizer,
            'max_seq_length': max_seq_length,
            'do_lower_case': do_lower_case,
        }
        super().__init__(dataset_type, dataset_params, batch_size, shuffle=shuffle)
