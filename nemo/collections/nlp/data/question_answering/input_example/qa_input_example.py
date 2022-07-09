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

from typing import List


class QAExample(object):
    """
    A single training/test example for a QA dataset, as loaded from disk.

    Args:
        qas_id: The example's unique identifier
        question_text: The question string
        context_text: The context string
        context_id: id representing context string
        answer_text: The answer string
        start_position_character: The character position of the start of
            the answer, 0 indexed
        title: The title of the example
        answers: None by default, this is used during evaluation.
            Holds answers as well as their start positions.
        is_impossible: False by default, set to True if the example has
            no possible answer.
    """

    def __init__(
        self,
        qas_id: str,
        question_text: str,
        context_text: str,
        context_id: int,
        answer_text: str,
        start_position_character: int,
        title: str,
        answers: List[str] = [],
        is_impossible: bool = False,
    ):
        self.qas_id = qas_id
        self.question_text = question_text
        self.context_id = context_id
        self.context_text = context_text
        self.answer_text = answer_text
        self.title = title
        self.is_impossible = is_impossible
        self.answers = answers
        self.start_position_character = start_position_character
