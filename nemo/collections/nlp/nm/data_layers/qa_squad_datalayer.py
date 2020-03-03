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

from nemo.collections.nlp.data import SquadDataset
from nemo.collections.nlp.nm.data_layers.text_datalayer import TextDataLayer
from nemo.core import ChannelType, LabelsType, NeuralType
from nemo.utils.decorators import add_port_docs

__all__ = ['BertQuestionAnsweringDataLayer']


class BertQuestionAnsweringDataLayer(TextDataLayer):
    """
    Creates the data layer to use for Question Answering classification task.

    Args:
        data_file (str): data_file in *.json.
        tokenizer (obj): Tokenizer object, e.g. NemoBertTokenizer.
        version_2_with_negative (bool): True if training should allow
            unanswerable questions.
        doc_stride (int): When splitting up a long document into chunks,
            how much stride to take between chunks.
        max_query_length (int): All training files which have a duration less
            than min_duration are dropped. Can't be used if the `utt2dur` file
            does not exist. Defaults to None.
        max_seq_length (int): All training files which have a duration more
            than max_duration are dropped. Can't be used if the `utt2dur` file
            does not exist. Defaults to None.
        mode (str): Use "train", "eval", or "test" to define between
            training and evaluation and inference.
        batch_size (int): Batch size. Defaults to 64.
        dataset_type (Dataset): Question Answering class.
            Defaults to SquadDataset.
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
        start_positions: indices of tokens which constitute start position of answer
        end_positions: indices of tokens which constitute end position of answer
        unique_ids: id of the Question answer example this instance belongs to
        """
        return {
            "input_ids": NeuralType(('B', 'T'), ChannelType()),
            "input_type_ids": NeuralType(('B', 'T'), ChannelType()),
            "input_mask": NeuralType(('B', 'T'), ChannelType()),
            "unique_ids": NeuralType(tuple('B'), ChannelType()),
            "start_positions": NeuralType(tuple('B'), ChannelType(), optional=True),
            "end_positions": NeuralType(tuple('B'), ChannelType(), optional=True),
        }

    def __init__(
        self,
        data_file,
        tokenizer,
        version_2_with_negative,
        doc_stride,
        max_query_length,
        max_seq_length,
        mode,
        batch_size=64,
        use_cache=True,
        shuffle=False,
        dataset_type=SquadDataset,
    ):
        dataset_params = {
            'data_file': data_file,
            'mode': mode,
            'tokenizer': tokenizer,
            'version_2_with_negative': version_2_with_negative,
            'max_query_length': max_query_length,
            'max_seq_length': max_seq_length,
            'use_cache': use_cache,
            'doc_stride': doc_stride,
        }

        super().__init__(dataset_type, dataset_params, batch_size, shuffle=shuffle)
