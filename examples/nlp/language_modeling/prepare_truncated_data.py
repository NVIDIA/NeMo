import os
from tqdm import tqdm
from joblib import Parallel, delayed
import json

from nemo.collections.common.tokenizers import TokenizerSpec
from nemo.collections.common.tokenizers import SentencePieceTokenizer
import argparse
from constants import TASKS

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

def _process_line(line, task, tokenizer, max_seq_length, tokens_to_generate_extra, prompt):
    """
        Read line, tokenize input and truncate to max_seq_len
        
        Returns:
            truncated_text: truncated input text
    """
    question = line['input'][line['query_start_index']:]
    question_len = len(tokenizer.text_to_tokens(question))
    context = line['input'][:line['query_start_index']]
    context_tokens = tokenizer.text_to_tokens(context)
    context_len = len(context_tokens)
    truncation_seperator = line['truncation_seperator']
    truncation_seperator_tokens_len = len(tokenizer.text_to_tokens(truncation_seperator))
    
    total_len = context_len + tokens_to_generate_extra + question_len
    if total_len > max_seq_length:
        truncated_text = tokenizer.tokens_to_text(context_tokens[:(max_seq_length - tokens_to_generate_extra - truncation_seperator_tokens_len - question_len)]) + truncation_seperator + question
    else:
        truncated_text = line['input']
    if prompt is not None:
        end = f"\n\n{TASKS[task]['response']}\n"
        assert truncated_text.endswith(end)
        truncated_text = truncated_text.replace(end, "")
        truncated_text = prompt.replace("{context}", truncated_text)
    return truncated_text

def process_data(tokenizer: TokenizerSpec, prompt:str, task: str, max_seq_length: int, data_dir: str, n_jobs: int=-1, remove_newline_tab: bool=False):
    task = task.lower()
    if task not in TASKS:
        raise NotImplementedError(f"{task} not implemented")
    
    task_file = f"{data_dir}/{TASKS[task]['subset']}"
    if not os.path.exists(data_dir) or not os.path.exists(task_file):
        raise ValueError(f"{data_dir} or {task_file} not found")
    
    tokens_to_generate = TASKS[task]["tokens_to_generate"]
    prompt = None if prompt in ["None", None, ""] else prompt
    if prompt is not None:
        # when passing via command line, "\n" is changed to "\n\n"
        prompt = prompt.replace("\\n", "\n")
        assert "{prompt}" in prompt, "prompt must contain {prompt} for replacement"
        tokens_to_generate += len(tokenizer.text_to_tokens(prompt.replace("{prompt}", "")))
       
    with open(task_file, 'r') as f_in:
        lines = [json.loads(line) for line in f_in]

    truncated_texts = Parallel(n_jobs)(delayed(_process_line)(line, task, tokenizer, max_seq_length, tokens_to_generate, prompt) for line in tqdm(lines))

    if remove_newline_tab:
        truncated_texts = [s.replace("\n"," ").replace("\t"," ").strip().replace("  ", " ") for s in truncated_texts]
    return lines, truncated_texts
            

if __name__ == "__main__":
    process_data(task="qasper",
                 data_dir="/mnt/ssd8/llm/data/zero-scrolls/Scrolls-zero",
                 tokenizer=SentencePieceTokenizer(model_path="/mnt/ssd8/llm/checkpoints/cpts/mt_nlg_plus_multilingual_ja_zh_the_stack_frac_015_256k.model"),
                 max_seq_lens=8192,
                 tokens_to_generate=128,
                 prompt=None)
