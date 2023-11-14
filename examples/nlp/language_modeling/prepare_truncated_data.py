import json
import os
from statistics import quantiles

import numpy as np
import pandas as pd
from constants import TASKS
from joblib import Parallel, delayed
from tqdm import tqdm

from nemo.collections.common.tokenizers import SentencePieceTokenizer, TokenizerSpec

"""
This scpript processes the zero-scrolls datasets and truncates the input to a given max seq len
based on the format described in https://arxiv.org/pdf/2305.14196.pdf

For CHAT models:
If computing log-prob for a given sequence
f"<extra_id_0>System\n\n<extra_id_1>User\n{context}\n<extra_id_1>Assistant\n{response}"
If doing free-form generation
f"<extra_id_0>System\n\n<extra_id_1>User\n{context}\n<extra_id_1>Assistant\n"

BOS=False for all

For BASE models:
no prompt is needed
"""


def _process_line(
    line, task, tokenizer, max_seq_length, tokens_to_generate_extra, prompt, truncation_pos, remove_newline_tab=False
):
    """
        Read line, tokenize input and truncate to max_seq_len

        Returns:
            truncated_text: truncated input text
    """
    # For some of the tasks, we add an additional instructions for the chat models.
    if prompt is not None and TASKS[task]['chat_instruction'] != '':
        document_header_idx = line['input'][: line['document_start_index']].rfind(f'\n\n{TASKS[task]["context"]}\n')
        line['input'] = (
            line['input'][:document_header_idx]
            + f' {TASKS[task]["chat_instruction"]}'
            + line['input'][document_header_idx:]
        )
        line['query_start_index'] += len(TASKS[task]['chat_instruction']) + 1
        line['query_end_index'] += len(TASKS[task]['chat_instruction']) + 1
        line['document_start_index'] += len(TASKS[task]['chat_instruction']) + 1
        line['document_end_index'] += len(TASKS[task]['chat_instruction']) + 1

    question = line['input'][line['query_start_index'] :]
    question_len = len(tokenizer.text_to_tokens(question))

    context = line['input'][: line['query_start_index']]
    context_tokens = tokenizer.text_to_tokens(context)
    context_len = len(context_tokens)

    truncation_seperator = line['truncation_seperator']
    truncation_seperator_tokens_len = len(tokenizer.text_to_tokens(truncation_seperator))

    if remove_newline_tab:
        question = question.replace('\n', ' ').replace('\t', ' ').strip().replace('  ', ' ')
        question_len = len(tokenizer.text_to_tokens(question))

        context = context.replace('\n', ' ').replace('\t', ' ').strip().replace('  ', ' ')
        context_tokens = tokenizer.text_to_tokens(context)
        context_len = len(context_tokens)

        input_text = line['input']
        input_text = input_text.replace('\n', ' ').replace('\t', ' ').strip().replace('  ', ' ')

    total_len = context_len + tokens_to_generate_extra + question_len
    if truncation_pos == 'right':
        if total_len > max_seq_length:
            truncated_text = (
                tokenizer.tokens_to_text(
                    context_tokens[
                        : (max_seq_length - tokens_to_generate_extra - truncation_seperator_tokens_len - question_len)
                    ]
                )
                + truncation_seperator
                + question
            )
        else:
            truncated_text = input_text
    elif truncation_pos == 'left':
        if total_len > max_seq_length:
            truncation_seperator = "[The beginning of the transcript is omitted] ..."
            truncation_seperator_tokens_len = len(tokenizer.text_to_tokens(truncation_seperator))
            # Split the context into two parts at the last occurrence of the context string
            context_prefix, context_suffix = context.split(TASKS[task]['context'], maxsplit=1)
            # Truncate the suffix from the right side to ensure that its length is equal to the maximum allowed length for the truncated text
            suffix_tokens = tokenizer.text_to_tokens(context_suffix)
            suffix_tokens = suffix_tokens[
                -(
                    max_seq_length
                    - tokens_to_generate_extra
                    - truncation_seperator_tokens_len
                    - question_len
                    - len(tokenizer.text_to_tokens(TASKS[task]['context']))
                    - len(tokenizer.text_to_tokens(context_prefix))
                ) :
            ]
            # Convert the truncated suffix back to text and concatenate it with the prefix, truncation separator, and question to create the final truncated text
            truncated_suffix = tokenizer.tokens_to_text(suffix_tokens)
            truncated_text = (
                context_prefix + TASKS[task]['context'] + truncation_seperator + truncated_suffix + question
            )
            print(len(tokenizer.text_to_tokens(truncated_text)))
        else:
            truncated_text = input_text
    elif truncation_pos == 'middle':
        if total_len > max_seq_length:
            truncation_seperator = "... [The middle of the transcript is omitted] ..."
            truncation_seperator_tokens_len = len(tokenizer.text_to_tokens(truncation_seperator))
            # Calculate the number of tokens to truncate from each side
            num_tokens_to_truncate = total_len - (
                max_seq_length + tokens_to_generate_extra + truncation_seperator_tokens_len + question_len
            )
            num_tokens_to_truncate_half = num_tokens_to_truncate // 2

            # Find the middle of context_tokens and truncate num_tokens_to_truncate tokens from the middle
            middle = len(context_tokens) // 2
            prefix_tokens = context_tokens[: middle - num_tokens_to_truncate_half]
            suffix_tokens = context_tokens[middle + num_tokens_to_truncate_half :]
            num_tokens_to_remove_from_prefix = len(prefix_tokens) - (
                max_seq_length - tokens_to_generate_extra - truncation_seperator_tokens_len - question_len
            )
            num_tokens_to_remove_from_suffix = len(suffix_tokens) - (
                max_seq_length - tokens_to_generate_extra - truncation_seperator_tokens_len - question_len
            )
            if num_tokens_to_remove_from_prefix > 0 and num_tokens_to_remove_from_suffix > 0:
                prefix_tokens = prefix_tokens[num_tokens_to_remove_from_prefix // 2 :]
                suffix_tokens = suffix_tokens[: len(suffix_tokens) - (num_tokens_to_remove_from_suffix // 2)]
            elif num_tokens_to_remove_from_prefix > 0:
                prefix_tokens = prefix_tokens[num_tokens_to_remove_from_prefix:]
            elif num_tokens_to_remove_from_suffix > 0:
                suffix_tokens = suffix_tokens[: len(suffix_tokens) - num_tokens_to_remove_from_suffix]

            # Convert the truncated parts back to text and concatenate them with the truncation separator and the question to create the final truncated text
            truncated_prefix = tokenizer.tokens_to_text(prefix_tokens)
            truncated_suffix = tokenizer.tokens_to_text(suffix_tokens)
            truncated_text = truncated_prefix + truncation_seperator + truncated_suffix + question
        else:
            truncated_text = input_text

    if prompt is not None:
        end = f'\n\n{TASKS[task]["response"]}\n'
        assert truncated_text.endswith(end)
        truncated_text = truncated_text.replace(end, '')

    if prompt is not None:
        truncated_text = prompt.replace('{context}', truncated_text)
    assert len(tokenizer.text_to_tokens(truncated_text)) <= max_seq_length
    return truncated_text


def _process_line_longbench(
    line, task, tokenizer, max_seq_length, tokens_to_generate_extra, prompt, truncation_pos, remove_newline_tab=False
):
    """
        Read line, tokenize input and truncate to max_seq_len

        Returns:
            truncated_text: truncated input text
    """
    # For some of the tasks, we add an additional instructions for the chat models.
    if prompt is not None and TASKS[task]['chat_instruction'] != '':
        document_header_idx = line['input'][: line['document_start_index']].rfind(f'\n\n{TASKS[task]["context"]}\n')
        line['input'] = (
            line['input'][:document_header_idx]
            + f' {TASKS[task]["chat_instruction"]}'
            + line['input'][document_header_idx:]
        )
        line['query_start_index'] += len(TASKS[task]['chat_instruction']) + 1
        line['query_end_index'] += len(TASKS[task]['chat_instruction']) + 1
        line['document_start_index'] += len(TASKS[task]['chat_instruction']) + 1
        line['document_end_index'] += len(TASKS[task]['chat_instruction']) + 1

    input_text = line['input']
    context_text = line['context']
    prompt_text = TASKS[task]['template']
    if remove_newline_tab:
        input_text = input_text.replace('\n', ' ').replace('\t', ' ').strip().replace('  ', ' ')
        context_text = context_text.replace('\n', ' ').replace('\t', ' ').strip().replace('  ', ' ')

    tokenized_inputs = tokenizer.text_to_tokens(input_text)
    tokenized_contexts = tokenizer.text_to_tokens(context_text)
    tokenized_prompt = tokenizer.text_to_tokens(prompt_text)

    total_len = (
        len(tokenized_prompt) + len(tokenized_inputs) + len(tokenized_contexts) + TASKS[task]['tokens_to_generate']
    )
    if truncation_pos == 'right':
        if total_len > max_seq_length:
            truncation_seperator = "... [The end of the transcript is omitted]"
            truncations_tokens = tokenizer.text_to_tokens(truncation_seperator)
            truncated_context = tokenizer.tokens_to_text(
                tokenized_contexts[
                    : (
                        max_seq_length
                        - tokens_to_generate_extra
                        - len(tokenized_inputs)
                        - len(tokenized_prompt)
                        - len(truncations_tokens)
                    )
                ]
            )
            truncated_text = TASKS[task]['template'].format(
                context=truncated_context + truncation_seperator, input=input_text
            )
        else:
            truncated_text = TASKS[task]['template'].format(context=context_text, input=input_text)
    elif truncation_pos == 'left':
        if total_len > max_seq_length:
            truncation_seperator = "[The beginning of the transcript is omitted] ..."

            # Split the context into two parts at the last occurrence of the context string
            # Truncate the suffix from the right side to ensure that its length is equal to the maximum allowed length for the truncated text
            context = context_text
            truncations_tokens = tokenizer.text_to_tokens(truncation_seperator)
            truncated_context = context[
                -(
                    max_seq_length
                    - tokens_to_generate_extra
                    - len(tokenized_inputs)
                    - len(tokenized_prompt)
                    - len(truncations_tokens)
                ) :
            ]
            truncated_text = TASKS[task]['template'].format(
                context=truncation_seperator + truncated_context, input=input_text
            )
        else:
            truncated_text = TASKS[task]['template'].format(context=context_text, input=input_text)
    elif truncation_pos == 'middle':
        if total_len > max_seq_length:
            truncation_seperator = "... [The middle of the transcript is omitted] ..."
            # Calculate the number of tokens to truncate from each side
            truncations_tokens = tokenizer.text_to_tokens(truncation_seperator)
            num_tokens_to_truncate = (
                total_len
                - max_seq_length
                + tokens_to_generate_extra
                + len(tokenized_inputs)
                + len(tokenized_prompt)
                + len(truncations_tokens)
            )
            num_tokens_to_truncate_half = num_tokens_to_truncate // 2

            # Find the middle of context_tokens and truncate num_tokens_to_truncate tokens from the middle
            middle = len(tokenized_contexts) // 2
            prefix_tokens = tokenized_contexts[: middle - num_tokens_to_truncate_half]
            suffix_tokens = tokenized_contexts[middle + num_tokens_to_truncate_half :]

            # Convert the truncated parts back to text and concatenate them with the truncation separator and the question to create the final truncated text
            truncated_prefix = tokenizer.tokens_to_text(prefix_tokens)
            truncated_suffix = tokenizer.tokens_to_text(suffix_tokens)
            truncated_context = truncated_prefix + truncation_seperator + truncated_suffix
            truncated_text = TASKS[task]['template'].format(context=truncated_context, input=input_text)
            # print(len(tokenizer.text_to_tokens(line['context'])),len(tokenizer.text_to_tokens(truncated_context)))
        else:
            truncated_text = TASKS[task]['template'].format(context=context_text, input=input_text)
            # print(len(tokenizer.text_to_tokens(truncated_text)))

    if prompt is not None:
        end = f'\n\n{TASKS[task]["response"]}\n'
        assert truncated_text.endswith(end)
        truncated_text = truncated_text.replace(end, '')

    if prompt is not None:
        truncated_text = prompt.replace('{context}', truncated_text)

    assert len(tokenizer.text_to_tokens(truncated_text)) <= max_seq_length
    return truncated_text


def _long_bench_metadata(
    line, task, tokenizer: TokenizerSpec,
):

    tokenized_inputs = tokenizer.text_to_tokens(line['input'])
    context = line['context'].replace('\n', ' ').replace('\t', ' ').strip().replace('  ', ' ')
    tokenized_contexts = tokenizer.text_to_tokens(context)
    tokenized_answers = [tokenizer.text_to_tokens(answer) for answer in line['answers']]

    total_tokens_in_answers = sum(len(answer) for answer in tokenized_answers)

    return len(tokenized_inputs), len(tokenized_contexts), total_tokens_in_answers


def process_data(
    tokenizer: TokenizerSpec,
    prompt: str,
    task: str,
    max_seq_length: int,
    data_dir: str,
    truncation_pos: str,
    n_jobs: int = -1,
    remove_newline_tab: bool = False,
):
    task = task.lower()
    if task not in TASKS:
        raise NotImplementedError(f'{task} not implemented')

    task_file = os.path.join(data_dir, TASKS[task]['subset'])
    if not os.path.exists(data_dir) or not os.path.exists(task_file):
        raise ValueError(f'{data_dir} or {task_file} not found')

    tokens_to_generate = TASKS[task]['tokens_to_generate']
    if prompt not in ['None', None, '']:
        # when passing via command line, '\n' is changed to '\n\n'
        prompt = prompt.replace('\\n', '\n')
        assert '{context}' in prompt, 'prompt must contain {context} for replacement'
        tokens_to_generate += len(tokenizer.text_to_tokens(prompt.replace('{context}', '')))

    with open(task_file, 'r') as f_in:
        lines = [json.loads(line) for line in f_in]

    if task[-3:] == '_lb':
        truncated_texts = Parallel(n_jobs)(
            delayed(_process_line_longbench)(
                line,
                task,
                tokenizer,
                max_seq_length,
                tokens_to_generate,
                prompt,
                remove_newline_tab=remove_newline_tab,
                truncation_pos=truncation_pos,
            )
            for line in tqdm(lines)
        )
    else:
        truncated_texts = Parallel(n_jobs)(
            delayed(_process_line)(
                line,
                task,
                tokenizer,
                max_seq_length,
                tokens_to_generate,
                prompt,
                remove_newline_tab=remove_newline_tab,
                truncation_pos=truncation_pos,
            )
            for line in tqdm(lines)
        )

    return lines, truncated_texts


if __name__ == '__main__':
    # For testing
    process_data(
        task='trivia_qa_lb',
        data_dir='/mnt/ssd8/llm/data/zero-scrolls/Scrolls-zero/LongBench/',
        tokenizer=SentencePieceTokenizer(
            model_path='/mnt/ssd8/llm/checkpoints/cpts/mt_nlg_plus_multilingual_ja_zh_the_stack_frac_015_256k.model'
        ),
        max_seq_lens=8192,
        tokens_to_generate=128,
        prompt=None,
    )
