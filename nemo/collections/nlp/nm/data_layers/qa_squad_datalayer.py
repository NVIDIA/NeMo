# =============================================================================
# Copyright 2019 AI Applications Design Team at NVIDIA. All Rights Reserved.
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
from nemo.core import AxisType, BatchTag, NeuralType, TimeTag

__all__ = ['BertQuestionAnsweringDataLayer']


class BertQuestionAnsweringDataLayer(TextDataLayer):
    """
    Creates the data layer to use for Question Answering classification task.

    Args:
        data_file (str): data file.
        tokenizer (obj): Tokenizer object, e.g. NemoBertTokenizer.
        version_2_with_negative (bool): True if training should allow
            unanswerable questions.
        doc_stride (int): When splitting up a long document into chunks,
            how much stride to take between chunks.
        max_query_length (iny): All training files which have a duration less
            than min_duration are dropped. Can't be used if the `utt2dur` file
            does not exist. Defaults to None.
        max_seq_length (int): All training files which have a duration more
            than max_duration are dropped. Can't be used if the `utt2dur` file
            does not exist. Defaults to None.
        mode (str): Use "train" or "dev" to define between
            training and evaluation.
        batch_size (int): Batch size. Defaults to 64.
        dataset_type (class): Question Answering class.
            Defaults to SquadDataset.
    """

    @property
    def output_ports(self):
        """Returns definitions of module output ports.

            input_ids:
                0: AxisType(BatchTag)

                1: AxisType(TimeTag)

            input_type_ids:
                0: AxisType(BatchTag)

                1: AxisType(TimeTag)

            input_mask:
                0: AxisType(BatchTag)

                1: AxisType(TimeTag)

            start_positions:
                0: AxisType(BatchTag)

            end_positions:
                0: AxisType(BatchTag)

            unique_ids:
                0: AxisType(BatchTag)

        """
        return {
            "input_ids": NeuralType({0: AxisType(BatchTag), 1: AxisType(TimeTag)}),
            "input_type_ids": NeuralType({0: AxisType(BatchTag), 1: AxisType(TimeTag)}),
            "input_mask": NeuralType({0: AxisType(BatchTag), 1: AxisType(TimeTag)}),
            "start_positions": NeuralType({0: AxisType(BatchTag)}),
            "end_positions": NeuralType({0: AxisType(BatchTag)}),
            "unique_ids": NeuralType({0: AxisType(BatchTag)}),
        }

    def __init__(
        self,
        data_file,
        tokenizer,
        version_2_with_negative,
        doc_stride,
        max_query_length,
        max_seq_length,
        mode="train",
        batch_size=64,
        dataset_type=SquadDataset,
    ):
        dataset_params = {
            'data_file': data_file,
            'mode': mode,
            'tokenizer': tokenizer,
            'version_2_with_negative': version_2_with_negative,
            'max_query_length': max_query_length,
            'max_seq_length': max_seq_length,
            'doc_stride': doc_stride,
        }

        super().__init__(dataset_type, dataset_params, batch_size, shuffle=False)
