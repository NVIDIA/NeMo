import json
import os

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


def _process_line(line, task, tokenizer, max_seq_length, tokens_to_generate_extra, prompt, remove_newline_tab=False):
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

    total_len = context_len + tokens_to_generate_extra + question_len
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
        truncated_text = line['input']

    if prompt is not None:
        end = f'\n\n{TASKS[task]["response"]}\n'
        assert truncated_text.endswith(end)
        truncated_text = truncated_text.replace(end, '')

    if remove_newline_tab:
        truncated_text = truncated_text.replace('\n', ' ').replace('\t', ' ').strip().replace('  ', ' ')

    if prompt is not None:
        truncated_text = prompt.replace('{context}', truncated_text)

    assert len(tokenizer.text_to_tokens(truncated_text)) <= max_seq_length
    return truncated_text


def process_data(
    tokenizer: TokenizerSpec,
    prompt: str,
    task: str,
    max_seq_length: int,
    data_dir: str,
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

    truncated_texts = Parallel(n_jobs)(
        delayed(_process_line)(
            line, task, tokenizer, max_seq_length, tokens_to_generate, prompt, remove_newline_tab=remove_newline_tab
        )
        for line in tqdm(lines)
    )

    return lines, truncated_texts


if __name__ == '__main__':
    # For testing
    process_data(
        task='qasper',
        data_dir='/mnt/ssd8/llm/data/zero-scrolls/Scrolls-zero',
        tokenizer=SentencePieceTokenizer(
            model_path='/mnt/ssd8/llm/checkpoints/cpts/mt_nlg_plus_multilingual_ja_zh_the_stack_frac_015_256k.model'
        ),
        max_seq_lens=8192,
        tokens_to_generate=128,
        prompt=None,
    )
