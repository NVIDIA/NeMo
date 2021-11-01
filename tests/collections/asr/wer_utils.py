# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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

from abc import ABC, abstractmethod
from copy import deepcopy
from itertools import chain
from typing import List, Tuple

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence

from nemo.collections.asr.metrics.wer import word_error_rate
from nemo.collections.common.tokenizers.tokenizer_spec import TokenizerSpec


class AbstractWEREncoderDecoder(ABC):
    @property
    @abstractmethod
    def blank_id(self):
        return self._blank_id

    def ctc_tensor_to_text(self, tensor: torch.Tensor, sequence_lengths: torch.Tensor = None) -> List[str]:
        """
        Decodes a sequence of labels to words

        Args:
            tensor: A torch.Tensor of shape [Batch, Time] of integer indices that correspond
                to the index of some character in the vocabulary of the tokenizer.
            sequence_lengths: Optional tensor of length `Batch` which contains the integer lengths
                of the sequence in the padded `predictions` tensor.

        Returns:
            A list of str which represent the CTC decoded strings per sample
        """
        hypotheses = []
        # Drop predictions to CPU
        cpu_tensor = tensor.long().cpu()
        if tensor.ndim > 2:
            raise ValueError(f"Input tensor has {tensor.ndim} dimensions.")
        # iterate over batch
        for ind in range(cpu_tensor.shape[0]):
            ids = cpu_tensor[ind].detach().numpy().tolist()
            if sequence_lengths is not None:
                ids = ids[: sequence_lengths[ind]]
            # CTC decoding procedure
            decoded_prediction = []
            previous = self.blank_id
            for p in ids:
                if (p != previous or previous == self.blank_id) and p != self.blank_id:
                    decoded_prediction.append(p)
                previous = p

            text = self.decode_tokens_to_str(decoded_prediction)
            hypotheses.append(text)
        return hypotheses

    def string_to_ctc_tensor(self, txt: str) -> torch.Tensor:
        # This function emulates how CTC output could like for txt
        string_in_id_form = self.text_to_ids(txt)
        ctc_list = []
        prev_id = -1
        for c in string_in_id_form:
            # when character is repeated we need to insert CTC blank symbol
            if c != prev_id:
                ctc_list.append(c)
            else:
                ctc_list.append(self.blank_id)
                ctc_list.append(c)
            prev_id = c
        return torch.Tensor(ctc_list)

    def text_batch_to_tensor(self, batch: List[str], ctc: bool) -> Tuple[torch.Tensor, torch.Tensor]:
        batch = [self.string_to_ctc_tensor(s) if ctc else self.string_to_tensor(s) for s in batch]
        lengths = torch.tensor([len(s) for s in batch])
        return pad_sequence(batch, batch_first=True, padding_value=0), lengths

    def batch_of_text_batches_to_tensor(
        self, batch_of_text_batches: List[List[str]], ctc: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        is_batch_size_equal_to_size_of_first_batch = [
            len(batch) == len(batch_of_text_batches[0]) for batch in batch_of_text_batches
        ]
        if not all(is_batch_size_equal_to_size_of_first_batch):
            raise ValueError(
                f"All batches have to have equal sizes. Batches with indices "
                f"{np.nonzero(np.array(is_batch_size_equal_to_size_of_first_batch) == False)[:3]} have sizes which "
                f"are not equal to the size of the first batch."
            )
        huge_batch, lengths = self.text_batch_to_tensor(list(chain(*batch_of_text_batches)), ctc)
        first_dims = [len(batch_of_text_batches), len(batch_of_text_batches[0])]
        return huge_batch.reshape(first_dims + [-1]), lengths.reshape(first_dims)

    def string_to_tensor(self, txt: str) -> torch.Tensor:
        return torch.Tensor(self.text_to_ids(txt))

    @abstractmethod
    def decode_tokens_to_str(self, tokens: List[str]) -> str:
        pass

    @abstractmethod
    def decode_ids_to_tokens(self, tokens: List[int]) -> List[str]:
        pass

    @abstractmethod
    def text_to_ids(self, txt: str) -> List[int]:
        pass


class WEREncoderDecoderVocabulary(AbstractWEREncoderDecoder):
    def __init__(self, vocabulary: List[str]):
        self.labels_map = vocabulary.copy()
        self._blank_id = len(self.labels_map)
        self.inv_vocabulary = {v: i for i, v in enumerate(self.labels_map)}

    @property
    def blank_id(self) -> int:
        return self._blank_id

    def decode_tokens_to_str(self, tokens: List[int]) -> str:
        hypothesis = ''.join(self.decode_ids_to_tokens(tokens))
        return hypothesis

    def decode_ids_to_tokens(self, tokens: List[int]) -> List[str]:
        token_list = [self.labels_map[c] for c in tokens if c != self.blank_id]
        return token_list

    def text_to_ids(self, txt: str) -> List[int]:
        return [self.inv_vocabulary[c] for c in txt]


class WEREncoderDecoderBPE(AbstractWEREncoderDecoder):
    def __init__(self, tokenizer: TokenizerSpec):
        self.tokenizer = deepcopy(tokenizer)
        self._blank_id = self.tokenizer.tokenizer.vocab_size

    @property
    def blank_id(self) -> int:
        return self._blank_id

    def decode_tokens_to_str(self, tokens: List[int]) -> str:
        hypothesis = self.tokenizer.ids_to_text(tokens)
        return hypothesis

    def decode_ids_to_tokens(self, tokens: List[int]) -> List[str]:
        token_list = self.tokenizer.ids_to_tokens(tokens)
        return token_list

    def text_to_ids(self, txt: str) -> List[int]:
        return self.tokenizer.text_to_ids(txt)


def reference_wer_func(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    target_lengths: torch.Tensor,
    predictions_lengths: torch.Tensor,
    decoder: AbstractWEREncoderDecoder,
):
    predictions = decoder.ctc_tensor_to_text(predictions, predictions_lengths)
    targets_cpu = targets.long().cpu()
    references = []
    for tgt_len, target in zip(target_lengths, targets_cpu):
        references.append(decoder.decode_tokens_to_str(target[: tgt_len.item()].numpy().tolist()))
    return word_error_rate(predictions, references)
