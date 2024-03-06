import copy
import random
from typing import Optional

import numpy as np
import torch.utils.data
from lhotse.dataset.collation import collate_vectors as collate_vectors_lhotse


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


def ceil_to_nearest(n, m):
    return (n + m - 1) // m * m


class TextProcessing:
    """
    Text processing pipeline for AudioQuestionAnswerDataset and TarredAudioQuestionAnswerDataset.
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
        if self.add_eos and self.tokens_to_generate == 0:
            total_ids += 1

        # If the total number of token is greater than the max, we will try to truncate the answer
        if total_ids > self.max_seq_length:
            truncation_length = total_ids - self.max_seq_length
            # TODO(zhehuai)
            answer_ids = answer_ids[: -min(truncation_length, len(answer_ids))]
            context_ids = context_ids[: -min(truncation_length, len(context_ids))]

        input_ids = context_ids
        answer_start_idx = len(input_ids)

        # Adds bos token in the start
        if self.add_bos:
            context_ids = [self.tokenizer.bos_id] + context_ids
            input_ids = [self.tokenizer.bos_id] + input_ids
            answer_start_idx += 1

        # Adds sep token between text/prompt and answer
        if self.add_sep:
            context_ids = context_ids + [self.sep_id]
            input_ids = input_ids + [self.sep_id]
            answer_start_idx += 1

        # TODO: create a copy of answer_ids and mask on it
        if self.input_text_mask_ratio is not None and self.input_text_mask_ratio > 0:
            if self.sample_alpha is None:
                masked_answer_ids, _ = self._random_mask_tokens(
                    answer_ids, self.input_text_mask_ratio, self.tokenizer.unk_id
                )
            else:
                sample_answer_ids = pre_pad + _text_to_ids(answer_text, self.sample_alpha, lang=lang)
                # does not consider different length for now
                masked_answer_ids, _ = self._random_mask_tokens(
                    answer_ids, self.input_text_mask_ratio, self.tokenizer.unk_id, sample_tokens=sample_answer_ids,
                )
            masked_input_ids = input_ids + masked_answer_ids
        input_ids = input_ids + answer_ids

        # Only training need to consider eos token
        if self.add_eos and self.tokens_to_generate == 0:
            input_ids = input_ids + [self.tokenizer.eos_id]
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


# TODO(zhehuai)
def update_to_asr_task(canary_tokens):
    if canary_tokens.shape[1] > 5:
        canary_tokens = copy.deepcopy(canary_tokens)
        canary_tokens[:, 3] = canary_tokens[:, 1]
        canary_tokens[:, 2] = 8
    return canary_tokens


def convert_canary_prompt_to_text(prompt, is_canary_tokens_augment):
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

    def get_task_template(text, is_canary_tokens_augment, simple_augment=True):
        if text == "<|transcribe|":
            template = 'Transcribe the spoken content to written <|SLANG|> text, <|PNC|>.'
        elif text == "<|translate|":
            if is_canary_tokens_augment:
                if simple_augment:
                    template = 'Transcribe the spoken content to written <|SLANG|> text, then translate this to <|TLANG|> text, <|PNC|>'
                else:
                    template = 'Transcribe the spoken content to written <|SLANG|> text, then translate this to English text, then translate this to <|TLANG|> text, <|PNC|>'
            else:
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
        task = get_task_template(ps[2], is_canary_tokens_augment)
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
    """

    def __init__(
        self,
        text_processor: TextProcessing,
        default_question: str,
        tokens_to_generate: int,
        pad_to_max_length: bool,
        max_seq_length: int,
        noise_cuts: Optional = None,
        canary_processor: Optional = None,
        context_len_for_AR_decoding: Optional = 5,
        convert_canary_prompt_to_text: bool = False,
        prepend_to_exist_question: Optional = None,
        canary_tokens_augment_ratio: float = 0.0,
        random_context_prob: float = 0.0,
    ):
        from lhotse.dataset import AudioSamples, CutMix

        super().__init__()
        self.text_processor = text_processor
        self.load_audio = AudioSamples(fault_tolerant=True)
        self.maybe_mix_noise = (
            _identity if noise_cuts is None else CutMix(noise_cuts, pad_to_longest=False, random_mix_offset=True)
        )
        self.tokens_to_generate = tokens_to_generate
        self.pad_to_max_length = pad_to_max_length
        self.max_seq_length = max_seq_length

        self.question = default_question
        self.canary_processor = canary_processor
        self.context_len_for_AR_decoding = context_len_for_AR_decoding
        self.convert_canary_prompt_to_text = convert_canary_prompt_to_text
        self.prepend_to_exist_question = prepend_to_exist_question
        self.canary_tokens_augment_ratio = canary_tokens_augment_ratio
        self.random_context_prob = random_context_prob

    def _inject_random_context_into_question(self, cut, random_context_num=32, random_context_positive_percent=0.1):
        if self.random_context_prob is not None and self.random_context_prob > 0:
            current_words = cut.supervisions[0].text.split()
            if len(current_words) == 0:
                return
            if np.random.random() < self.random_context_prob and hasattr(self, 'random_context') and len(self.random_context) > 0:
                positive_num = int(random_context_num * random_context_positive_percent)
                positives = np.random.choice(current_words, positive_num)
                negatives = np.random.choice(self.random_context, random_context_num - positive_num)
                candidate_words = np.concatenate((positives, negatives))
                np.random.shuffle(candidate_words)
                context = f"Following words may occur in audio: {candidate_words} ".replace('\n', '')
                cut.question = context + cut.question
            self.random_context = current_words

    def __getitem__(self, cuts) -> dict[str, torch.Tensor | list[str] | dict]:
        cuts = cuts.sort_by_duration()
        cuts = self.maybe_mix_noise(cuts)

        audio, audio_lens, cuts = self.load_audio(cuts)

        return_batch = {}
        audio_ratio = []
        for id, cut in enumerate(cuts):
            if hasattr(cut, "is_text_only") and cut.is_text_only:
                audio_ratio.append(0.0)
            else:
                audio_ratio.append(1.0)

        if self.canary_processor != None:
            is_canary_tokens_augment = torch.rand(1) < self.canary_tokens_augment_ratio
            _, _, canary_tokens, canary_token_lens = self.canary_processor.__getitem__(cuts)
            if is_canary_tokens_augment:
                return_batch['canary_tokens'] = update_to_asr_task(canary_tokens)
            else:
                return_batch['canary_tokens'] = canary_tokens
            return_batch['canary_token_lengths'] = canary_token_lens
            for id, cut in enumerate(cuts):
                canary_text = self.canary_processor.tokenizer._tokenizer.ids_to_text(
                    canary_tokens[id][: self.context_len_for_AR_decoding].tolist()
                )
                if audio_ratio[id] == 0.0:
                    assert hasattr(cut, "question")
                elif self.prepend_to_exist_question and hasattr(cut, "question"):
                    cut.question = self.prepend_to_exist_question + cut.question
                elif self.convert_canary_prompt_to_text:
                    cut.question = convert_canary_prompt_to_text(canary_text, is_canary_tokens_augment)
                elif hasattr(cut, "question"):
                    pass
                else:
                    cut.question = self.question + ' ' + canary_text
        for id, cut in enumerate(cuts):
            self._inject_random_context_into_question(cut)

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
                context=cut.question if hasattr(cut, "question") else default_question, output=cut.supervisions[0].text,
                lang='en' if cut.supervisions[0].language is None else cut.supervisions[0].language
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
            [torch.as_tensor(_build_loss_mask(item)) for item in examples], max_length=max_length, padding_value=0
        )[:, 1:],
        "position_ids": torch.arange(max_length, dtype=torch.long).repeat(batch_size, 1),
        "contexts": collate_vectors(fields["context_ids"], max_length=max_length, padding_value=pad_id),
        "context_lengths": torch.LongTensor([len(seq) for seq in fields["context_ids"]]),
        "answers": collate_vectors(fields["answer_ids"], max_length=max_length, padding_value=pad_id),
        "max_length": torch.LongTensor([max_length]*batch_size),
    }


def adjust_input_ids(item: dict) -> dict:
    """Mimics the logic from nemo/collections/multimodal/data/audio_text_qa_dataset.py:131"""
    item["input_ids"] = item.get("masked_input_ids", item["input_ids"])
    return item


def as_dict(arg: list[dict]) -> dict[str, list]:
    return {k: [item[k] for item in arg] for k in arg[0].keys()}


def _identity(x):
    return x


def _build_loss_mask(processed_example: dict, answer_only_loss: bool = True):
    """ Pad input_ids in batch to max batch length while building loss mask """
    # function copied from nemo/collections/nlp/data/language_modelling/megatron/gpt_sft_dataset.py
    input_ids = processed_example['input_ids']
    answer_start_idx = processed_example['answer_start_idx']
    if answer_only_loss:
        loss_mask = [float(idx >= answer_start_idx) for idx in range(len(input_ids))]
    else:
        loss_mask = [1.0] * len(input_ids)

    return loss_mask
