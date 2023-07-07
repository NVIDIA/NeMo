# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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

from pathlib import Path
from typing import Callable, Dict, List, Optional, Union

import numpy as np
import torch

from nemo.collections.common.tokenizers.tokenizer_spec import TokenizerSpec
from nemo.collections.nlp.data.language_modeling.megatron.dataset_utils import get_samples_mapping
from nemo.collections.nlp.data.language_modeling.text_memmap_dataset import JSONLMemMapDataset
from nemo.core.classes import Dataset
from nemo.collections.tts.parts.utils.tts_dataset_utils import (
    get_base_dir,
    general_padding
)

__all__ = ['GPTSFTDataset']


class GPTSFTDataset(Dataset):
    def __init__(
        self,
        file_path: str,
        tokenizer: TokenizerSpec,
        max_seq_length: int = 1024,
        min_seq_length: int = 1,
        add_bos: bool = False,
        add_eos: bool = True,
        add_sep: bool = False,
        sep_id: int = None,
        max_num_samples: int = None,
        seed: int = 1234,
        context_key: str = "text",
        label_key: str = "answer",
        separate_prompt_and_response_with_newline: bool = False,
        answer_only_loss: bool = True,
        truncation_field: str = "answer",
        pad_to_max_length: bool = False,  # (@adithyare) allows for much faster training especially in PEFT settings.
        index_mapping_dir: str = None,
        prompt_template: str = None,
        virtual_tokens: int = 0,
        tokens_to_generate: int = 0,
        sup_data_path: Optional[Union[Path, str]] = None,
    ):
        """
        DOES NOT SUPPORT ON THE FLY GENERATION OF SPEECH CODES
        Speech codes are load from pytorch file on each call of __getitem__, could be inefficient

        file_path: Path to a JSONL GPT supervised fine-tuning dataset. Data is formatted as multiple JSON lines with each line formatted as follows. {'input': 'John von Neumann\nVon Neumann made fundamental contributions .... Q: What did the math of artificial viscosity do?', 'output': 'smoothed the shock transition without sacrificing basic physics'}
        tokenizer: Tokenizer for the dataset. Instance of a class that inherits TokenizerSpec (ex: YTTM, SentencePiece).
        max_seq_length (int): maximum sequence length for each dataset examples. Examples will either be truncated to fit this length or dropped if they cannot be truncated.
        min_seq_length (int): min length of each data example in the dataset. Data examples will be dropped if they do not meet the min length requirements.
        add_bos (bool): Whether to add a beginning of sentence token to each data example
        add_eos (bool): Whether to add an end of sentence token to each data example
        add_sep (bool): Whether to add a separation token to each data example (goes between prompt and answer)
        tokens_to_generate (int): (inference only) Number of tokens to generate during inference
        seed: Random seed for data shuffling.
        max_num_samples: Maximum number of samples to load. This can be > dataset length if you want to oversample data. If None, all samples will be loaded.
        seed: int = 1234,
        context_key: Key to use for the context in your JSONL file
        label_key: Key to use for the label in your JSONL file
        separate_prompt_and_response_with_newline: Adds a newline between prompt and response.
        answer_only_loss: If True, will compute the loss only on the answer part of the input. If False, will compute the loss on the entire input.
        truncation_field: Field to use for truncation. (Options: "answer", "context"). Field to be used for truncation if the combined length exceeds the max sequence length.
        pad_to_max_length: Whether to pad the input to the max sequence length. If False, will pad to the max length of the current batch.
        index_mapping_dir: Directory to save the index mapping to. If None, will write to the same folder as the dataset.
        prompt_template: Prompt template to inject via an fstring. Formatted like Q: {input}\n\nA: {output}
        """
        self.tokenizer = tokenizer
        self.file_path = file_path
        self.max_seq_length = max_seq_length
        self.min_seq_length = min_seq_length
        self.add_bos = add_bos
        self.add_eos = add_eos
        self.add_sep = add_sep
        self.sep_id = sep_id
        self.max_num_samples = max_num_samples
        self.seed = seed
        self.context_key = context_key
        self.label_key = label_key
        self.separate_prompt_and_response_with_newline = separate_prompt_and_response_with_newline
        self.answer_only_loss = answer_only_loss
        self.truncation_field = truncation_field
        self.pad_to_max_length = pad_to_max_length
        self.index_mapping_dir = index_mapping_dir
        self.prompt_template = prompt_template
        self.virtual_tokens = virtual_tokens
        self.tokens_to_generate = tokens_to_generate
        if self.prompt_template is not None:  #jasoli: Should be None at the first stage
            # When providing things like newlines in the prompt template via the CLI, they are escaped. This line unescapes them.
            self.prompt_template = self.prompt_template.encode('utf-8').decode('unicode_escape')
        assert self.truncation_field in ["answer", "context"]

        self.speech_offset = 256000 #TODO: Fix hardcode
        # Initialize sup_data_path, sup_data_types and run preprocessing methods for every supplementary data type
        if sup_data_path is not None:
            Path(sup_data_path).mkdir(parents=True, exist_ok=True)
            self.sup_data_path = sup_data_path

        self.codec_folder = Path(self.sup_data_path) / "codec"
        self.codec_folder.mkdir(exist_ok=True, parents=True)

        self.indexed_dataset = JSONLMemMapDataset(
            dataset_paths=[file_path], tokenizer=None, header_lines=0, index_mapping_dir=index_mapping_dir
        )

        # Will be None after this call if `max_num_samples` is None
        # Should be None, disabling
        # self._build_samples_mapping()

    # def _build_samples_mapping(self):
    #     if self.max_num_samples is not None:
    #         self.samples_mapping = get_samples_mapping(
    #             indexed_dataset=self.indexed_dataset,
    #             data_prefix=self.file_path,
    #             num_epochs=None,
    #             max_num_samples=self.max_num_samples,
    #             max_seq_length=self.max_seq_length - 2,
    #             short_seq_prob=0,
    #             seed=self.seed,
    #             name=self.file_path.split('/')[-1],
    #             binary_head=False,
    #             index_mapping_dir=self.index_mapping_dir,
    #         )
    #     else:
    #         self.samples_mapping = None

    def __len__(self):
        return len(self.indexed_dataset)
        # if self.max_num_samples is None:
        #     return len(self.indexed_dataset)
        # else:
        #     return len(self.samples_mapping)

    def __getitem__(self, idx):
        if isinstance(idx, np.int64):
            idx = idx.item()

        if self.samples_mapping is not None:
            assert idx < len(self.samples_mapping)
            idx, _, _ = self.samples_mapping[idx]
            if isinstance(idx, np.uint32):
                idx = idx.item()

        assert idx < len(self.indexed_dataset)
        example = self.indexed_dataset[idx]
        return self._process_example(example)

    def _process_example(self, example):
        """
        TODO: Need to concatenate question as well if going to use SQUAD
        Create an example by concatenating text and answer.
        Truncation is carried out when needed, but it is performed only on the prompt side.
        BOS, EOS, and SEP, are added if specified.
        """
        is_speech = example.get("answer_type", False)  # TODO: answer_type hardcode needs to be updated
        if is_speech:
            # TODO: Enable context in speech?

            dur = example.get("answer_duration", -1)
            audio_filepath = example["answer"]
            # Let's keep audio name and all internal directories in rel_audio_path_as_text_id to avoid any collisions
            # TODO: fix hardcode
            rel_audio_path = Path(audio_filepath).relative_to("/data/speech/LibriTTS2/LibriTTS/train-clean-100/").with_suffix("")
            rel_audio_path_as_text_id = str(rel_audio_path).replace("/", "_")

            # Convert to codes
            codec_codes, codec_codes_length = None, None # Codes
            codec_path = self.codec_folder / f"{rel_audio_path_as_text_id}.pt"

            if codec_path.exists():
                codec_codes = torch.load(codec_path).long()
            else:
                raise NotImplementedError("No audio loading option yet")
                codec_codes = self.get_codec(audio).long()
                torch.save(codec_codes, codec_path)

            for i in range(codec_codes.shape[0]):
                codec_codes[i] = (codec_codes[i] + self.speech_offset+i*1024).long()

            context_ids = []
            answer_ids = codec_codes  # 8 x speech_length
        else:
            context = example[self.context_key]
            output = example[self.label_key]

            if self.separate_prompt_and_response_with_newline and self.prompt_template is None:
                text = context + '\n' + output
            elif not self.separate_prompt_and_response_with_newline and self.prompt_template is None:
                text = context + ' ' + output

            # TODO: Need to add virtual tokens for different speech tasks
            pre_pad = []
            tokenized_text = pre_pad + self.tokenizer.text_to_ids(text)
            context_ids = pre_pad + self.tokenizer.text_to_ids(context)
            answer_ids = tokenized_text[len(context_ids) :]

        # for the long context cases, collate_fn includes self.tokens_to_generate for padding
        answer_length = len(answer_ids[0]) if is_speech else len(answer_ids)
        total_ids = len(context_ids) + max(answer_length, self.tokens_to_generate)

        # TODO: Not supported for speech codes
        if self.add_bos:
            raise NotImplementedError("BOS not implemented speech codes")
            total_ids += 1
        if self.add_sep:
            raise NotImplementedError("SEP not implemented speech codes")
            total_ids += 1
        if self.add_eos:
            raise NotImplementedError("EOS not implemented speech codes")
            total_ids += 1

        # If the total number of token is greater than the max, we will try to truncate the answer
        if total_ids > self.max_seq_length:
            truncation_length = total_ids - self.max_seq_length
            if self.truncation_field == "answer":
                if is_speech:
                    for i in range(codec_codes.shape[0]):
                        answer_ids[i] = answer_ids[i, : -min(truncation_length, len(answer_ids))]
                else:
                    answer_ids = answer_ids[: -min(truncation_length, len(answer_ids))]
            elif self.truncation_field == "context":
                # TODO: Not implemented for speech codes
                context_ids = context_ids[: -min(truncation_length, len(context_ids))]

        if len(context_ids) > self.max_seq_length:
            context_ids = context_ids[: self.max_seq_length]

        assert len(context_ids) <= self.max_seq_length
        # input_ids = context_ids

        answer_start_idx = 0
        # # Adds sep token between text/prompt and answer
        # if self.add_sep:
        #     input_ids = input_ids + [self.sep_id]
        #     answer_start_idx += 1

        input_ids = answer_ids

        if self.add_bos:
            if is_speech:
                pass
            else:
                input_ids = [self.tokenizer.bos_id] + input_ids
                answer_start_idx += 1
        if self.add_eos:
            if is_speech:
                pass
            else:
                input_ids = input_ids + [self.tokenizer.eos_id]

        if len(input_ids) < self.min_seq_length or len(input_ids) > self.max_seq_length:
            raise NotImplementedError("Does not work for speech :(")
            input_ids = input_ids[: self.max_seq_length]

        context_ids = torch.LongTensor(context_ids)
        if not is_speech:
            input_ids = torch.LongTensor(input_ids)

        processed_example = {
            'input_ids': input_ids,
            'answer_start_idx': answer_start_idx,
            'context_ids': context_ids,
            'context_length': len(context_ids),
        }

        return processed_example

    def _maybe_cast_to_list(self, x):
        if isinstance(x, np.ndarray):
            return [item.tolist() for item in x]
        return x

    def _ceil_to_nearest(self, n, m):
        return (n + m - 1) // m * m

    def _collate_item(self, item, max_length, pad_id):
        item = self._maybe_cast_to_list(item)
        # max_length = max([len(x) for x in item]) if item else 0
        # here [0] should be tokenizer.pad_id
        item = [x + [pad_id] * (max_length - len(x)) for x in item]
        return item

    def _build_loss_mask(self, processed_example):
        """ Pad input_ids in batch to max batch length while building loss mask """
        input_ids = processed_example['input_ids']
        answer_start_idx = processed_example['answer_start_idx']
        if self.answer_only_loss:
            loss_mask = [float(idx >= answer_start_idx) for idx in range(len(input_ids))]
        else:
            loss_mask = [1.0] * len(input_ids)

        return loss_mask

    def _build_loss_mask_2(self, input_id_length, answer_start_idx):
        """ Pad input_ids in batch to max batch length while building loss mask """
        if self.answer_only_loss:
            loss_mask = [float(idx >= answer_start_idx) for idx in range(input_id_length)]
        else:
            loss_mask = [1.0] * input_id_length

        return loss_mask

    @torch.no_grad()
    def _create_attention_mask(self, max_length):
        """Create `attention_mask`.
        Args:
            input_ids: A 1D tensor that holds the indices of tokens.
        """
        # seq_length = len(input_ids)
        # `attention_mask` has the shape of [1, seq_length, seq_length]
        attention_mask = torch.tril(torch.ones((max_length, max_length))).unsqueeze(0)
        attention_mask = attention_mask < 0.5
        return attention_mask

    def collate_fn(self, batch):
        # TODO: input_ids could be 2D array if speech
        (input_ids, answer_start_idx, context_ids, context_length) = zip(*batch)
        input_lengths = []
        max_length = -1
        for input_id in input_ids:
            length_i = input_id.shape[0] if input_id.dim() < 2 else input_id.shape[1]
            input_lengths.append(length_i)
        max_length = max(input_lengths) - 1

        # increase max length to nearest multiple of 4 or 8
        if self.pad_to_max_length:
            max_length = self.max_seq_length
        else:
            max_length = min(self.max_seq_length, self._ceil_to_nearest(max_length, 8))
        assert max_length <= self.max_seq_length

        (tokens, labels, loss_mask, contexts, context_lengths, speech_mask_list) = ([],[],[],[],[],[])

        def pad_text_to_speech_dims(text_tensor, pad_id):
            token_len = text_tensor.shape[0]
            empty_padding = torch.ones((7, token_len), dtype=text_tensor.dtype, device=text_tensor.device) * pad_id
            return torch.cat((text_tensor.unsqueeze(0), empty_padding), dim=0)

        for i in range(len(batch)):
            # Should pad_id be eos_id?
            input_id_padded = general_padding(input_ids[i], input_lengths[i], max_length, pad_value=self.tokenizer.pad_id)
            if len(input_id_padded.shape) < 2:
                input_id_padded = pad_text_to_speech_dims(pad_text_to_speech_dims, self.tokenizer.pad_id)
            tokens.append(input_id_padded[:, :-1])
            labels.append(input_id_padded[:, 1:])

            context_id_padded = general_padding(context_ids[i], context_length[i], max_length, pad_value=self.tokenizer.pad_id)
            # if len(context_id_padded.shape) < 2:
            #     context_id_padded = torch.LongTensor(pad_text_to_speech_dims(pad_text_to_speech_dims, self.tokenizer.pad_id))
            contexts.append(context_id_padded)

            context_lengths.append(torch.LongTensor(context_length[i]))

            loss_mask_i = torch.LongTensor(self._build_loss_mask_2(input_lengths[i]-1, answer_start_idx[i]))
            loss_mask_i_padded = general_padding(loss_mask_i, input_lengths[i]-1, max_length, pad_value=0)
            loss_mask.append(loss_mask_i_padded)

            speech_mask = loss_mask_i_padded if len(input_id_padded.shape) >= 2 else torch.zeros(loss_mask_i_padded.shape)
            speech_mask_list.append(speech_mask)

        attention_mask = self._create_attention_mask(max_length)
        attention_mask = attention_mask.repeat(len(batch))

        position_ids = torch.LongTensor(list(range(max_length)))
        position_ids = position_ids.repeat(len(batch))

        processed_batch = {
            'tokens': torch.stack(tokens),
            'labels': torch.stack(labels),
            'attention_mask': attention_mask,
            'loss_mask': torch.stack(loss_mask),
            'position_ids': position_ids,
            'contexts': torch.stack(contexts),  #@jasoli: Seems to only be used in predict_step
            'context_lengths': torch.stack(context_lengths),
            "speech_mask": torch.stack(speech_mask_list),
        }

        return processed_batch
