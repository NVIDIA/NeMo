import copy
import random
from typing import Callable, Optional, Sequence

import numpy as np
import torch.utils.data
from lhotse import CutSet
from lhotse.dataset.collation import collate_vectors as collate_vectors_lhotse

from nemo.collections.asr.data.audio_to_text_lhotse import TokenizerWrapper
from nemo.collections.multimodal.speech_llm.parts.utils.data_utils import build_loss_mask, ceil_to_nearest
from nemo.utils import logging


def collate_vectors(items, max_length: int, padding_value):
    vectors = collate_vectors_lhotse(items, padding_value=padding_value)
    if max_length > vectors.size(1):
        vectors = torch.cat(
            [vectors, padding_value * torch.ones(vectors.size(0), max_length - vectors.size(1), dtype=vectors.dtype)],
            dim=1,
        )
    if items[0].shape[0] < 1:
        vectors = vectors.long()
    return vectors


class TextProcessing:
    """
    Text processing pipeline for lhotse based speech_llm data loader.
    This class is adapted from the one used in nemo/collections/nlp/data/language_modeling/megatron/gpt_sft_dataset.py
    The class follows the interface of _process_example which takes in a context and an output
      and processes them into a formatted training example.

    Args:
        tokenizer: text tokenizer object
        max_seq_length (int): maximum sequence length for each dataset examples. Examples will either be truncated to fit this length or dropped if they cannot be truncated.
        min_seq_length (int): min length of each data example in the dataset. Data examples will be dropped if they do not meet the min length requirements.
        add_bos (bool): Whether to add a beginning of sentence token to each data example
        add_eos (bool): Whether to add an end of sentence token to each data example
        add_sep (bool): Whether to add a separation token to each data example (goes between prompt and answer)
        sep_id (int): The id of the separation token
        separate_prompt_and_response_with_newline (bool): Whether to separate the prompt and response with a newline character
        answer_only_loss (bool): Whether to compute the loss only on the answer part of the input
        truncation_field (str): Field to use for truncation. (Options: "answer", "context"). Field to be used for truncation if the combined length exceeds the max sequence length.
        pad_to_max_length (bool): Whether to pad the input to the max sequence length. If False, will pad to the max length of the current batch.
        prompt_template (str): Prompt template to inject via an fstring. Formatted like Q: {input}\n\nA: {output}
        virtual_tokens (int): Number of virtual tokens to add to the beginning of the input
        tokens_to_generate (int): Number of tokens to generate during inference
        input_key (str): Key to use for the input in your JSONL file
        output_key (str): Key to use for the output in your JSONL file
        end_string (Optional[str]): If not None, add this string to the end of the answer.
        sample_alpha (Optional[float]): For SPE subword sampling
        input_text_mask_ratio (Optional[float]): If not None, will mask the input text at this ratio.
    """

    def __init__(
        self,
        tokenizer: 'nemo.collections.common.tokenizers.TokenizerSpec',
        max_seq_length: int = 1024,
        min_seq_length: int = 1,
        add_bos: bool = False,
        add_eos: bool = True,
        add_sep: bool = False,
        sep_id: Optional[int] = None,
        seed: int = 1234,
        separate_prompt_and_response_with_newline: bool = False,
        answer_only_loss: bool = True,
        truncation_field: str = "answer",
        pad_to_max_length: bool = False,  # (@adithyare) allows for much faster training especially in PEFT settings.
        prompt_template: str = None,
        virtual_tokens: int = 0,
        tokens_to_generate: int = 0,
        input_key: str = 'input',
        output_key: str = 'output',
        end_string: Optional[str] = None,
        sample_alpha: Optional[float] = None,
        input_text_mask_ratio: Optional[float] = None,
    ):
        self.input_key = input_key
        self.output_key = output_key
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.min_seq_length = min_seq_length
        self.seed = seed
        self.separate_prompt_and_response_with_newline = separate_prompt_and_response_with_newline
        self.answer_only_loss = answer_only_loss
        self.truncation_field = truncation_field
        self.pad_to_max_length = pad_to_max_length
        self.prompt_template = prompt_template
        self.virtual_tokens = virtual_tokens
        self.tokens_to_generate = tokens_to_generate
        self.add_bos = add_bos
        self.add_eos = add_eos
        self.add_sep = add_sep
        self.end_string = end_string
        self.sample_alpha = sample_alpha
        self.input_text_mask_ratio = input_text_mask_ratio

        if add_bos and hasattr(tokenizer, "bos_id") and tokenizer.bos_id > 0:
            self.bos_id = tokenizer.bos_id
        else:
            self.bos_id = None

        if add_eos and hasattr(tokenizer, "eos_id") and tokenizer.eos_id > 0:
            self.eos_id = tokenizer.eos_id
        else:
            self.eos_id = None
        self.pad_id = self.eos_id if self.eos_id is not None else 0
        self.pad_id = tokenizer.pad_id if tokenizer.pad_id >= 0 else self.pad_id

        self.sep_id = sep_id if add_sep else None

        if self.prompt_template is not None:
            # When providing things like newlines in the prompt template via the CLI, they are escaped. This line unescapes them.
            self.prompt_template = self.prompt_template.encode('utf-8').decode('unicode_escape')
        assert self.truncation_field in ["answer", "context"]

    def _random_mask_tokens(self, input_tokens, mask_ratio, mask_token, sample_tokens=None):
        output_tokens = input_tokens[:]
        mask = []
        for i, token in enumerate(input_tokens):
            prob = random.random()
            mask_or_not = prob < mask_ratio
            if mask_or_not:
                output_tokens[i] = mask_token if sample_tokens is None or i >= len(sample_tokens) else sample_tokens[i]
            mask.append(mask_or_not)
        return output_tokens, mask

    def _process_example(self, context: str, output: str, lang: str):
        """
        Create an example by concatenating text and answer.
        Truncation is carried out when needed, but it is performed only on the prompt side.
        BOS, EOS, and SEP, are added if specified.

        function copied from nemo/collections/nlp/data/language_modelling/megatron/gpt_sft_dataset.py
        """

        def _text_to_ids(text, alpha=None, lang=None):
            from nemo.collections.common.tokenizers.aggregate_tokenizer import AggregateTokenizer

            if isinstance(self.tokenizer, AggregateTokenizer):
                return self.tokenizer.text_to_ids(text, lang)
            else:
                return self.tokenizer.text_to_ids(text, alpha)

        if self.prompt_template is not None:
            assert f'{{{self.input_key}}}' in self.prompt_template
            assert f'{{{self.output_key}}}' in self.prompt_template
            # Make sure that '{output}' always occurs at the end of the prompt template string
            assert self.prompt_template.index(f'{{{self.output_key}}}') == len(self.prompt_template) - len(
                f'{{{self.output_key}}}'
            )
            # Get the context by replacing only the input
            original_context = context
            context = (
                self.prompt_template.replace(f'{{{self.input_key}}}', context)
                .replace(f'{{{self.output_key}}}', '')
                .strip(' ')
            )
            # Replace the input and output placeholders with the actual input and output
            text = self.prompt_template.replace(f'{{{self.input_key}}}', original_context).replace(
                f'{{{self.output_key}}}', output
            )

        elif self.separate_prompt_and_response_with_newline:
            text = context + '\n' + output
        else:
            text = context + ' ' + output

        if self.virtual_tokens:
            # (@adithyare) we are going to insert "pad/eos" tokens in the beginning of the text and context
            # these pad/eos tokens are placeholders for virtual tokens
            pre_pad = [self.tokenizer.eos_id] * self.virtual_tokens
        else:
            pre_pad = []
        answer_text = text[len(context) :]
        # if input_text_mask_ratio, only do it on the input but not label
        answer_ids = pre_pad + _text_to_ids(
            answer_text, self.sample_alpha if self.input_text_mask_ratio is None else None, lang=lang
        )
        if self.end_string:
            answer_ids += _text_to_ids(self.end_string, lang=lang)
        context_ids = pre_pad + _text_to_ids(context, lang=lang)

        # for the long context cases, collate_fn includes self.tokens_to_generate for padding
        total_ids = len(context_ids) + max(len(answer_ids), self.tokens_to_generate)
        if self.add_bos:
            total_ids += 1
        if self.add_sep:
            total_ids += 1
        # Only training need to consider eos token
        if self.add_eos:
            total_ids += 1

        # If the total number of token is greater than the max, we will try to truncate the answer
        if total_ids > self.max_seq_length:
            truncation_length = total_ids - self.max_seq_length
            answer_ids = answer_ids[: -min(truncation_length, len(answer_ids))]
            context_ids = context_ids[: -min(truncation_length, len(context_ids))]

        input_ids = context_ids
        answer_start_idx = len(input_ids)

        # Adds bos token in the start
        if self.add_bos:
            context_ids = [self.bos_id] + context_ids
            input_ids = [self.bos_id] + input_ids
            answer_start_idx += 1

        # Adds sep token between text/prompt and answer
        if self.add_sep:
            context_ids = context_ids + [self.sep_id]
            input_ids = input_ids + [self.sep_id]
            answer_start_idx += 1

        # create a copy of answer_ids and mask on it
        if self.input_text_mask_ratio is not None and self.input_text_mask_ratio > 0:
            if self.sample_alpha is None:
                masked_answer_ids, _ = self._random_mask_tokens(
                    answer_ids, self.input_text_mask_ratio, self.tokenizer.unk_id
                )
            else:
                sample_answer_ids = pre_pad + _text_to_ids(answer_text, self.sample_alpha, lang=lang)
                # does not consider different length for now
                masked_answer_ids, _ = self._random_mask_tokens(
                    answer_ids,
                    self.input_text_mask_ratio,
                    self.tokenizer.unk_id,
                    sample_tokens=sample_answer_ids,
                )
            masked_input_ids = input_ids + masked_answer_ids
        input_ids = input_ids + answer_ids

        # Only training need to consider eos token
        if self.add_eos:
            input_ids = input_ids + [self.tokenizer.eos_id]
            answer_ids = answer_ids + [self.tokenizer.eos_id]
            if self.input_text_mask_ratio is not None and self.input_text_mask_ratio > 0:
                masked_input_ids = masked_input_ids + [self.tokenizer.eos_id]

        if len(input_ids) > self.max_seq_length:
            logging.warning(f'Input ids length {len(input_ids)} exceed max sequence length {self.max_seq_length}')
            input_ids = input_ids[: self.max_seq_length]
            if self.input_text_mask_ratio is not None and self.input_text_mask_ratio > 0:
                masked_input_ids = masked_input_ids[: self.max_seq_length]

        processed_example = {
            'input_ids': torch.as_tensor(input_ids),
            'answer_start_idx': torch.as_tensor(answer_start_idx),
            'context_ids': torch.as_tensor(context_ids),
            'context_length': len(context_ids),
            'answer_ids': torch.as_tensor(answer_ids),
        }

        if self.input_text_mask_ratio is not None and self.input_text_mask_ratio > 0:
            processed_example['masked_input_ids'] = torch.as_tensor(masked_input_ids)

        return processed_example


# The following function tries to reuse the canary data by following the special
# tokens in nemo/collections/asr/data/audio_to_text_lhotse_prompted.py
# TODO: try to move away from canary special token conversion and design a configurable prompting setup instead
def convert_canary_prompt_to_text(prompt):
    ps = prompt.replace("<pad>", "").split('>')

    def get_lang(text):
        if text == "<|fr|":
            lang = 'French'
        elif text == "<|es|":
            lang = 'Spanish'
        elif text == "<|de|":
            lang = 'German'
        elif text == "<|en|":
            lang = 'English'
        else:
            assert False, 'Unknown language {}'.format(prompt)
        return lang

    def get_task_template(text):
        if text == "<|transcribe|":
            template = 'Transcribe the spoken content to written <|SLANG|> text, <|PNC|>.'
        elif text == "<|translate|":
            template = 'Translate the spoken <|SLANG|> content to written <|TLANG|> text, <|PNC|>'
        else:
            assert False, 'Unknown task {}'.format(prompt)
        return template

    def get_pnc(text):
        if text == "<|nopnc|":
            pnc = 'ignoring punctuations and capitalization'
        elif text == "<|pnc|":
            pnc = 'with punctuations and capitalizations'
        else:
            assert False, 'Unknown pnc {}'.format(prompt)
        return pnc

    if len(ps) == 6:
        source_lang = get_lang(ps[1])
        target_lang = get_lang(ps[3])
        pnc = get_pnc(ps[4])
        task = get_task_template(ps[2])
        task = task.replace('<|SLANG|>', source_lang)
        task = task.replace('<|TLANG|>', target_lang)
        task = task.replace('<|PNC|>', pnc)
    else:
        task = 'Above is not speech.'
    return task


class LhotseAudioQuestionAnswerDataset(torch.utils.data.Dataset):
    """
    This dataset is based on Lhotse ASR dataset from ``audio_to_text_lhotse.py``
    and ``TarredAudioQuestionAnswerDataset`` from ``audio_text_qa_dataset.py``.

    Unlike native NeMo datasets, Lhotse dataset defines only the mapping from
    a CutSet (meta-data) to a mini-batch with PyTorch tensors.
    Specifically, it performs tokenization, I/O, augmentation, and feature extraction (if any).
    Managing data, sampling, de-duplication across workers/nodes etc. is all handled
    by Lhotse samplers instead.

    Args:
        text_processor: TextProcessing object
        default_question: Default question to use if no question is provided
        tokens_to_generate: Number of tokens to generate during inference
        pad_to_max_length: Whether to pad the input to the max sequence length. If False, will pad to the max length of the current batch.
        max_seq_length: Maximum sequence length for each dataset examples. Examples will either be truncated to fit this length or dropped if they cannot be truncated.
        prompt_format_fn: Optional function to format the prompt
        prompt_tokenizer: Optional tokenizer to use for the prompt
        convert_canary_prompt_to_text: Whether to convert canary prompt to text
        prepend_to_exist_question: Optional string to prepend to existing question
    """

    def __init__(
        self,
        text_processor: TextProcessing,
        default_question: str,
        tokens_to_generate: int,
        pad_to_max_length: bool,
        max_seq_length: int,
        prompt_format_fn: Optional[Callable[[CutSet, TokenizerWrapper, bool], Sequence[Sequence[int]]]] = None,
        prompt_tokenizer: TokenizerWrapper = None,
        convert_canary_prompt_to_text: bool = False,
        prepend_to_exist_question: Optional = None,
    ):
        from lhotse.dataset import AudioSamples, CutMix

        super().__init__()
        self.text_processor = text_processor
        self.load_audio = AudioSamples(fault_tolerant=True)
        self.tokens_to_generate = tokens_to_generate
        self.pad_to_max_length = pad_to_max_length
        self.max_seq_length = max_seq_length

        self.question = default_question
        self.prompt_format_fn = prompt_format_fn
        self.prompt_tokenizer = prompt_tokenizer
        self.convert_canary_prompt_to_text = convert_canary_prompt_to_text
        self.prepend_to_exist_question = prepend_to_exist_question

    def __getitem__(self, cuts) -> dict[str, torch.Tensor | list[str] | dict]:
        cuts = cuts.sort_by_duration()

        audio, audio_lens, cuts = self.load_audio(cuts)

        return_batch = {}
        audio_ratio = []
        for id, cut in enumerate(cuts):
            if hasattr(cut, "is_text_only") and cut.is_text_only:
                audio_ratio.append(0.0)
            else:
                audio_ratio.append(1.0)

        if self.prompt_format_fn != None:
            _, prompt_tokens = self.prompt_format_fn(cuts, self.prompt_tokenizer, inference=True)

            for id, cut in enumerate(cuts):
                prompt_text = self.prompt_tokenizer._tokenizer.ids_to_text(prompt_tokens[id])
                if audio_ratio[id] == 0.0:  # text only data should include question
                    assert hasattr(cut, "question")
                elif self.prepend_to_exist_question and hasattr(cut, "question"):
                    cut.question = self.prepend_to_exist_question + cut.question
                elif self.convert_canary_prompt_to_text:
                    cut.question = convert_canary_prompt_to_text(prompt_text)
                elif hasattr(cut, "question"):
                    # if the manifest has question field, use it
                    pass
                else:
                    # use the canary special token as it is
                    cut.question = self.question + ' ' + prompt_text
        else:  # the default format for speech_llm
            if hasattr(cut, "question"):
                pass
            else:
                cut.question = self.question

        metadata = []
        for id, cut in enumerate(cuts):
            metadata.append({'audio_filepath': cut.id + '.wav'})

        collated_text_data = collate_text_data(
            cuts=cuts,
            default_question=self.question,
            text_processor=self.text_processor,
            tokens_to_generate=self.tokens_to_generate,
            pad_to_max_length=self.pad_to_max_length,
            max_seq_length=self.max_seq_length,
        )
        return_batch.update(
            {
                "sample_ids": list(cuts.ids),
                "audio_signal": audio,
                "audio_signal_length": audio_lens,
                "audio_ratio": torch.FloatTensor(audio_ratio),
                "metadata": metadata,
                **collated_text_data,
            }
        )

        return return_batch


def collate_text_data(
    cuts,
    default_question: str,
    text_processor: TextProcessing,
    tokens_to_generate: int,
    pad_to_max_length: bool,
    max_seq_length: int,
) -> dict:
    """Perform text collation equivalent to nemo/collections/multimodal/data/audio_text_qa_dataset.py:121"""
    batch_size = len(cuts)
    pad_id = text_processor.pad_id
    examples = [
        adjust_input_ids(
            text_processor._process_example(
                context=cut.question if hasattr(cut, "question") else default_question,
                output=cut.supervisions[0].text,
                lang='en' if cut.supervisions[0].language is None else cut.supervisions[0].language,
            )
        )
        for cut in cuts
    ]
    fields = as_dict(examples)

    def get_max_len(input_list):
        return max([len(x) for x in input_list])

    max_length = tokens_to_generate + max(
        get_max_len(fields["input_ids"]), get_max_len(fields["context_ids"]), get_max_len(fields["answer_ids"])
    )
    # increase max length to nearest multiple of 4 or 8
    if pad_to_max_length:
        max_length = max_seq_length
    else:
        max_length = min(max_seq_length, ceil_to_nearest(max_length, 8))

    all_tokens = collate_vectors(fields["input_ids"], max_length=max_length, padding_value=pad_id)
    full_lengths = torch.LongTensor([len(item) for item in fields["input_ids"]])

    assert max_length <= max_seq_length, f"{max_length=} <= {max_seq_length=}"

    return {
        "tokens": all_tokens[:, :-1],
        "tokens_length": full_lengths - 1,
        "labels": all_tokens[:, 1:],
        "loss_mask": collate_vectors(
            [torch.as_tensor(build_loss_mask(item)) for item in examples], max_length=max_length, padding_value=0
        )[:, 1:],
        "position_ids": torch.arange(max_length, dtype=torch.long).repeat(batch_size, 1),
        "contexts": collate_vectors(fields["context_ids"], max_length=max_length, padding_value=pad_id),
        "context_lengths": torch.LongTensor([len(seq) for seq in fields["context_ids"]]),
        "answers": collate_vectors(fields["answer_ids"], max_length=max_length, padding_value=pad_id),
        "max_length": torch.LongTensor([max_length] * batch_size),
    }


def adjust_input_ids(item: dict) -> dict:
    """masked_input_ids is used for masking llm input text. Use when exists."""
    item["input_ids"] = item.get("masked_input_ids", item["input_ids"])
    return item


def as_dict(arg: list[dict]) -> dict[str, list]:
    return {k: [item[k] for item in arg] for k in arg[0].keys()}
