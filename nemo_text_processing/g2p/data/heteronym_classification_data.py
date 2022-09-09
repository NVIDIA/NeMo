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


import json
import os
import re
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from nemo.collections.common.tokenizers.tokenizer_spec import TokenizerSpec
from nemo.core.classes import Dataset
from nemo.utils import logging

__all__ = ['HeteronymClassificationDataset']


class HeteronymClassificationDataset(Dataset):
    def __init__(
        self,
        manifest: str,
        tokenizer: TokenizerSpec,
        wiki_homograph_dict: Dict[str, Dict[str, str]],
        wordid_to_idx: Dict[str, int],
        max_seq_len: int = 512,
        grapheme_field: str = "text_graphemes",
        with_labels: bool = True,
    ):
        """
        Creates dataset to use to run training and inference on G2PClassificationModel.
        Processes WikiHomograph raw data files:
        https://github.com/google-research-datasets/WikipediaHomographData/tree/master/data

        Args:
            manifest: path to manifest with "homograph_span", "start_end", "text_graphemes"
                and (optional) "word_id" fields. "word_id" is required for model training.
            tokenizer: pretrained tokenizer
            wiki_homograph_dict: a dictionary where each grapheme contains word_id to ipa_form mappings, e.g.,
                {'use': {'use_nou': "'juːs", 'use_vrb': "'juːz"}}
            wordid_to_idx: mapping from word id to index
            max_seq_len: maximum input sequence length
            grapheme_field: name of the field in the .json manifest with grapheme input
            with_labels: indicates whether labels are provided in the manifest. False for inference, True for training
        """
        super().__init__()

        if not os.path.exists(manifest):
            raise ValueError(f"{manifest} not found")

        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.data = []
        self.pad_token = 0
        self.with_labels = with_labels
        self.wiki_homograph_dict = wiki_homograph_dict
        self.wordid_to_idx = wordid_to_idx
        self.LOSS_PAD_TOKEN = -100
        self.PAD_TOKEN = 0

        num_skipped = 0
        with open(manifest, "r") as f:
            for line in f:
                line = json.loads(line)
                cur_start_end, cur_homographs = (line["start_end"], line["homograph_span"])

                # during inference word_id is not present in the manifest
                if "word_id" in line:
                    cur_word_ids = line["word_id"]
                else:
                    if isinstance(cur_homographs, str):
                        cur_word_ids = None
                    else:
                        cur_word_ids = [None] * len(cur_homographs)

                if isinstance(cur_homographs, str):
                    cur_start_end, cur_homographs, cur_word_ids = [cur_start_end], [cur_homographs], [cur_word_ids]

                example = self._prepare_sample(line[grapheme_field], cur_start_end, cur_homographs, cur_word_ids)
                if example is None:
                    num_skipped += 1
                    continue

                example_dict = {
                    "input_ids": example[0],
                    "target_and_negatives": example[1],
                    "subtokens_mask": example[2],
                }
                if with_labels:
                    example_dict["target"] = example[-1]

                self.data.append(example_dict)
        logging.info(f"Number of samples in {manifest}: {len(self.data)}, remove {num_skipped} lines")

    def _prepare_sample(
        self, sentence: str, start_end: List[Tuple[int, int]], homograph: str, word_ids: Optional[List[int]] = None
    ):
        """
        Prepares a single training sample

        Args:
            sentence: input sentence in grapheme form
            start_end: start and end indices of the homograph spans
            homograph: homograph
            word_ids: [Optional] target word_ids, use None for inference
        """
        # drop example where sequence length exceeds max sequence length, +2 for special tokens
        length = len(self.tokenizer.text_to_tokens(sentence)) + 2
        if length > self.max_seq_len:
            return None

        # add bos token
        input_ids = [self.tokenizer.bos_id]
        subtokens_mask = [self.PAD_TOKEN]  # the first tokens of heteronym spans are 1s, the rest of the tokens are 0s
        target_and_negatives = []
        with_labels = True
        if with_labels:
            target_word_ids = [self.LOSS_PAD_TOKEN]  # -100 to pad plain tokens

        heteronym_span_idx = 0
        # split sentence by space and keep track of word boundaries
        # we assume a heteronym is a starndalone word
        matches = [(m.group(0), (m.start(), m.end() - 1)) for m in re.finditer(r'\S+', sentence)]
        # import pdb; pdb.set_trace()
        for match in matches:
            word, word_start_end = match

            # check if the start of the next heteronym span is within the word indices
            if (
                heteronym_span_idx < len(start_end)
                and word_start_end[0] <= start_end[heteronym_span_idx][0] < word_start_end[1]
            ):
                heteronym_start_end = start_end[heteronym_span_idx]
                prefix = ""
                prefix_ids = []
                # for cases when word also includes punctuation marks at the beginning or a prefix,
                # e.g. "diffuse" vs. diffuse vs. pre-diffuse for heteronym {diffuse}
                if word_start_end[0] < heteronym_start_end[0]:
                    prefix = sentence[word_start_end[0] : heteronym_start_end[0]]
                    prefix_ids = self.tokenizer.text_to_ids(prefix)
                    subtokens_mask.extend([self.PAD_TOKEN] * len(prefix_ids))
                    # if "-Nestle" in word:
                    #     import pdb;
                    #     pdb.set_trace()

                word = word[word.index(prefix) + len(prefix_ids) :]
                word_input_ids = self.tokenizer.text_to_ids(word)
                input_ids.extend(prefix_ids + word_input_ids)

                subtokens_mask.extend([1] + [self.PAD_TOKEN] * (len(word_input_ids) - 1))
                if with_labels:
                    try:
                        cur_target_word_id = self.wordid_to_idx[word_ids[heteronym_span_idx]]
                    except:
                        import pdb

                        pdb.set_trace()
                        print()
                    target_word_ids.extend(
                        [self.LOSS_PAD_TOKEN] * len(prefix_ids)
                        + [cur_target_word_id]
                        + [self.LOSS_PAD_TOKEN] * (len(word_input_ids) - 1)
                    )

                heteronym = sentence.lower()[heteronym_start_end[0] : heteronym_start_end[1]]
                if heteronym not in self.wiki_homograph_dict:
                    logging.debug(f"{heteronym} is not supported. Skipping example.")
                    return None

                grapheme_ipa_forms = self.wiki_homograph_dict[heteronym]
                target_and_negatives.extend([self.wordid_to_idx[wordid_] for wordid_ in grapheme_ipa_forms])
                heteronym_span_idx += 1
            else:
                ids = self.tokenizer.text_to_ids(word)
                input_ids.extend(ids)
                subtokens_mask.extend([self.PAD_TOKEN] * len(ids))
                if with_labels:
                    target_word_ids.extend([self.LOSS_PAD_TOKEN] * len(ids))

        if heteronym_span_idx < len(start_end):
            logging.info("Not all heteronym spans were processed. Skipping example.")
            import pdb

            pdb.set_trace()
            return None

        # add eos token
        input_ids.append(self.tokenizer.eos_id)
        subtokens_mask.append(self.PAD_TOKEN)
        if with_labels:
            target_word_ids.append(self.LOSS_PAD_TOKEN)

        output = [input_ids, target_and_negatives, subtokens_mask]
        if with_labels:
            output.append(target_word_ids)

        if len(input_ids) != len(subtokens_mask):
            import pdb

            pdb.set_trace()
            print()
        # import pdb;
        # pdb.set_trace()
        return output  # [input_ids, target_and_negatives, subtokens_mask, [Optional] target]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    def map(self, text: str) -> List[int]:
        """ Creates a mapping from target labels to ids."""
        tokens = []
        for word_id, word in enumerate(text.split()):
            tokens.append(self.labels_tkn2id[word])
        return tokens

    def _collate_fn(self, batch):
        """
        Args:
            batch:  A list of tuples of (input_ids, target_and_negatives, subtokens_mask, [Optional] target_word_ids).
        """
        max_length = max([len(entry["input_ids"]) for entry in batch])

        padded_input_ids = []
        padded_subtokens_mask = []
        padded_attention_mask = []

        if self.with_labels:
            padded_targets = []

        for item in batch:
            input_ids = item["input_ids"]
            if len(input_ids) < max_length:
                pad_width = max_length - len(input_ids)
                padded_attention_mask.append([1] * len(input_ids) + [0] * pad_width)
                padded_input_ids.append(np.pad(input_ids, pad_width=[0, pad_width], constant_values=self.PAD_TOKEN))
                padded_subtokens_mask.append(
                    np.pad(item["subtokens_mask"], pad_width=[0, pad_width], constant_values=self.PAD_TOKEN)
                )

                if self.with_labels:
                    padded_targets.append(
                        np.pad(item["target"], pad_width=[0, pad_width], constant_values=self.LOSS_PAD_TOKEN)
                    )
            else:
                padded_attention_mask.append([1] * len(input_ids))
                padded_input_ids.append(input_ids)
                padded_subtokens_mask.append(item["subtokens_mask"])
                if self.with_labels:
                    padded_targets.append(item["target"])

        batch_size = len(batch)
        num_classes = len(self.wordid_to_idx)
        target_and_negatives = [entry["target_and_negatives"] for entry in batch]
        target_and_negatives_mask = torch.zeros(batch_size, num_classes)
        for i, cur_target_and_negatives in enumerate(target_and_negatives):
            for value in cur_target_and_negatives:
                target_and_negatives_mask[i][value] = 1

        # import pdb;
        # pdb.set_trace()
        # # Encode inputs (graphemes)
        # input_batch = [entry["input_ids"] for entry in batch]
        # input_encoding = self.tokenizer.tokenizer(
        #     input_batch, padding='longest', max_length=self.max_seq_len, truncation=True, return_tensors='pt',
        # )
        #
        # input_ids, attention_mask = input_encoding.input_ids, input_encoding.attention_mask
        # # create a mask 1 for target_and_negatives and 0 for the rest of the entries since they're not relevant to the target word
        # target_and_negatives = [entry["target_and_negatives"] for entry in batch]
        # batch_size = input_ids.shape[0]
        # num_classes = len(self.wordid_to_idx)
        # target_and_negatives_mask = torch.zeros(batch_size, num_classes)
        # for i, values in enumerate(target_and_negatives):
        #     for v in values:
        #         target_and_negatives_mask[i][v] = 1
        #
        # # pad prediction mask (masks out irrelevant subwords)
        # subword_mask = [entry["subword_mask"] for entry in batch]
        # pred_mask_len = [len(entry) for entry in subword_mask]
        # max_pred_mask_len = max(pred_mask_len)
        # subword_mask = [
        #     entry + [0] * (max_pred_mask_len - entry_len) for entry, entry_len in zip(subword_mask, pred_mask_len)
        # ]
        # subword_mask = torch.tensor(subword_mask)
        try:
            output = [
                torch.LongTensor(padded_input_ids),
                torch.LongTensor(padded_attention_mask),
                target_and_negatives_mask,
                torch.LongTensor(padded_subtokens_mask),
            ]

            if self.with_labels:
                output.append(torch.LongTensor(padded_targets))
        except:
            import pdb

            pdb.set_trace()
            print()
        return output


"""
    def _collate_fn(self, batch):
        max_length = 0
        for input_ids, segment_ids, input_mask, label in batch:
            if len(input_ids) > max_length:
                max_length = len(input_ids)

        padded_input_ids = []
        padded_segment_ids = []
        padded_input_mask = []
        labels = []
        for input_ids, segment_ids, input_mask, label in batch:
            if len(input_ids) < max_length:
                pad_width = max_length - len(input_ids)
                padded_input_ids.append(np.pad(input_ids, pad_width=[0, pad_width], constant_values=self.pad_id))
                padded_segment_ids.append(np.pad(segment_ids, pad_width=[0, pad_width], constant_values=self.pad_id))
                padded_input_mask.append(np.pad(input_mask, pad_width=[0, pad_width], constant_values=self.pad_id))
            else:
                padded_input_ids.append(input_ids)
                padded_segment_ids.append(segment_ids)
                padded_input_mask.append(input_mask)
            labels.append(label)

        return (
            torch.LongTensor(padded_input_ids),
            torch.LongTensor(padded_segment_ids),
            torch.LongTensor(padded_input_mask),
            torch.LongTensor(labels),
        )


"""
