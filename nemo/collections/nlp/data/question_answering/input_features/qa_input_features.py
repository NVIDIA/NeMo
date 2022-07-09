# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

from typing import Dict, List, Optional


class BERTQAInputFeatures(object):
    """ A single set of features of a QA example for BERT-like model """

    def __init__(
        self,
        unique_id: int,
        input_ids: List[int],
        input_mask: List[int],
        segment_ids: List[int],
        example_index: int = None,
        doc_span_index: int = None,
        tokens: List[str] = None,
        token_to_orig_map: Dict[int, int] = None,
        token_is_max_context: Dict[int, bool] = None,
        start_position: Optional[int] = None,
        end_position: Optional[int] = None,
        is_impossible: Optional[int] = None,
    ):
        self.unique_id = unique_id
        self.example_index = example_index
        self.doc_span_index = doc_span_index
        self.tokens = tokens
        self.token_to_orig_map = token_to_orig_map
        self.token_is_max_context = token_is_max_context
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.start_position = start_position
        self.end_position = end_position
        self.is_impossible = is_impossible


class S2SQAInputFeatures(object):
    """ A single set of features of a QA example for T5/BART-like model """

    def __init__(
        self,
        unique_id: int,
        input_ids: List[int],
        input_attn_mask: List[int],
        labels: List[int] = None,
        example_index: int = None,
        context_span_index: int = None,
        is_impossible: Optional[bool] = False,
    ):
        self.unique_id = unique_id
        self.input_ids = input_ids
        self.input_attn_mask = input_attn_mask
        self.labels = labels
        self.example_index = example_index
        self.context_span_index = context_span_index
        self.is_impossible = is_impossible


class GPTQAInputFeatures(object):
    """ A single set of features of a QA example for GPT-like model """

    def __init__(
        self,
        unique_id: int,
        input_ids: List[int],
        input_attn_mask: List[int],
        training_mask_end: int = None,
        labels: List[int] = None,
        example_index: int = None,
        context_span_index: int = None,
        is_impossible: Optional[bool] = False,
    ):
        self.unique_id = unique_id
        self.input_ids = input_ids
        self.input_attn_mask = input_attn_mask
        self.training_mask_end = training_mask_end
        self.labels = labels
        self.example_index = example_index
        self.context_span_index = context_span_index
        self.is_impossible = is_impossible
