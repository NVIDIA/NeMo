# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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


from nemo.collections.nlp.modules.common import VirtualPromptSource
from nemo.core import Dataset
from nemo.utils import logging

__all__ = ['BasePromptLearningDataset']


class BasePromptLearningDataset(Dataset):
    """
    The base dataset class for prompt-tuning or p-tuning.
    """

    def __init__(
        self,
        datasets,
        tokenizer,
        virtual_prompt_source: VirtualPromptSource,
        task_templates: dict,
        pseudo_tokens,
        pad_token_id: str,
        max_seq_length: int,
        min_seq_length: int = 1,
        add_bos: bool = False,
        add_eos: bool = True,
        for_train: bool = True,
    ):
        self.tokenizer = tokenizer
        self.virtual_prompt_source = virtual_prompt_source
        self.task_templates = task_templates
        self.pseudo_tokens = pseudo_tokens
        self.pseudo_token_ids = set(self.tokenizer.tokens_to_ids(self.pseudo_tokens))
        self.pad_token_id = pad_token_id
        self.max_seq_length = max_seq_length
        self.min_seq_length = min_seq_length
        self.add_bos = add_bos
        self.add_eos = add_eos
        self.for_train = for_train
        self.examples = []

        assert self.min_seq_length <= max_seq_length, "Min sequence length should be less than or equal to max"
        assert self.max_seq_length > 0, "Max sequence length should be greater than 0"

        logging.info("Loading and tokenizing dataset ... ")

        # Datasets is just a list of json dicts
        if isinstance(datasets[0], dict):
            self.load_data(datasets)

        # Datasets are a list of file path strings to .json or .jsonl files
        elif isinstance(datasets[0], str):
            for path in datasets:
                dataset = open(path, 'r', encoding='utf-8')
                self.load_data(dataset)
        else:
            raise ValueError("Datasets must be a list of dicts or a list of filepath strings")

    def _insert_virtual_token_placeholders(self, input_example, virtual_token_splits):
        """ Insert the correct number of pseudo tokens at the <|VIRTUAL_PROMPT_n|> markers """
        total_inserted_tokens = 0

        for idx in range(len(virtual_token_splits)):
            split_start = total_inserted_tokens
            split_end = total_inserted_tokens + virtual_token_splits[idx]
            pseudo_tokens_for_split = "".join(self.pseudo_tokens[split_start:split_end])
            input_example = input_example.replace(f'<|VIRTUAL_PROMPT_{idx}|>', pseudo_tokens_for_split)
            total_inserted_tokens = split_end

        return input_example

    def _truncate_input(self, truncation_field, input_ids, taskname, doc):
        """ Try to truncate input text to fit into the max sequence length """
        logging.info(
            f"Input greater than max sequence length. Attempting to truncate: '{truncation_field}' in task: '{taskname}'"
        )

        # Truncate the text ids in this part of input to try and fit max sequence length
        if truncation_field is not None and truncation_field in doc.keys():
            truncation_length = len(input_ids) - self.max_seq_length
            field_text = doc[truncation_field]
            field_text = self._add_leading_space(taskname, truncation_field, field_text)

            # Truncate field text
            field_text_ids = self.tokenizer.text_to_ids(field_text)
            truncated_text_ids = field_text_ids[: -min(truncation_length, len(field_text_ids))]

            # Replace original text ids with truncated text ids
            field_start, field_end = find_subsequence_location(input_ids, field_text_ids)
            input_ids = input_ids[:field_start] + truncated_text_ids + input_ids[field_end + 1 :]

        return input_ids

    def _add_leading_space(self, taskname, field_name, field_text):
        """ Add leading space to text if there is a space before it in the template """
        prompt_template = self.task_templates[taskname]["prompt_template"]
        field_text_start = prompt_template.find("{" + field_name + "}")
        if field_text_start != 0 and prompt_template[field_text_start - 1] == " ":
            field_text = " " + field_text

        return field_text

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


def find_subsequence_location(sequence, subsequence):
    """ Finds the start and end index of the first occurance 
        of a given subsequence within a larger list. Returns 
        the two indices corresponding to the postition of 
        the first and last token of the subseqeunce.
        Assumes subsequence is known to be in sequence. 
    """
    assert len(sequence) >= len(subsequence), "subsequence too long"

    start_idx = None
    next_subseq_token = subsequence[0]
    next_subsequence_idx = 1

    for seq_idx, token in enumerate(sequence):
        if token == next_subseq_token:
            if start_idx is None:
                start_idx = seq_idx

            if next_subsequence_idx == len(subsequence):
                end_idx = seq_idx
                return start_idx, end_idx
            else:
                next_subseq_token = subsequence[next_subsequence_idx]
                next_subsequence_idx += 1
        else:
            start_idx = None
            next_subseq_token = subsequence[0]
            next_subsequence_idx = 1

    raise ValueError("Subsequence not found in sequence")
