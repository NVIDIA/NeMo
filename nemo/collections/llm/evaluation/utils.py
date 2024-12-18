# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

from typing import List, Union

import numpy as np
from pytriton.client import ModelClient


def str_list2numpy(str_list: List[str]) -> np.ndarray:
    """"
    Convert a list of strings to a numpy array of strings.
    """
    str_ndarray = np.array(str_list)[..., np.newaxis]
    return np.char.encode(str_ndarray, "utf-8")


def query_llm(
    url: str,
    model: str,
    prompt: Union[str, List[str]],
    output_generation_logits: bool = True,
    stop_words_list: List[str] = None,
    bad_words_list: List[str] = None,
    no_repeat_ngram_size: int = None,
    max_tokens: int = 128,
    top_k: int = 1,
    top_p: float = 0.0,
    temperature: float = 1.0,
    random_seed: int = None,
    task_id: str = None,
    lora_uids: str = None,
    init_timeout: float = 60.0,
):
    """
    A method that sends post request to the model on PyTriton server and returns either generated text or
    logits.

    Args:
        url (str): The URL for the Triton server. Required.
        model_name (str): The name of the Triton model. Required.
        prompt (str, optional): The prompt to be used. Required if `prompt_file` is not provided.
        prompt_file (str, optional): The file path to read the prompt from. Required if `prompt` is not provided.
        stop_words_list (str, optional): A list of stop words.
        bad_words_list (str, optional): A list of bad words.
        no_repeat_ngram_size (int, optional): The size of the n-grams to disallow repeating.
        max_output_len (int): The maximum length of the output tokens. Defaults to 128.
        top_k (int): The top-k sampling parameter. Defaults to 1.
        top_p (float): The top-p sampling parameter. Defaults to 0.0.
        temperature (float): The temperature for sampling. Defaults to 1.0.
        task_id (str, optional): The task ID for the prompt embedding tables.
    """
    prompts = str_list2numpy([prompt] if isinstance(prompt, str) else prompt)
    inputs = {"prompts": prompts}

    if output_generation_logits:
        inputs["output_generation_logits"] = np.full(prompts.shape, output_generation_logits, dtype=np.bool_)

    if max_tokens is not None:
        inputs["max_output_len"] = np.full(prompts.shape, max_tokens, dtype=np.int_)

    if top_k is not None:
        inputs["top_k"] = np.full(prompts.shape, top_k, dtype=np.int_)

    if top_p is not None:
        inputs["top_p"] = np.full(prompts.shape, top_p, dtype=np.single)

    if temperature is not None:
        inputs["temperature"] = np.full(prompts.shape, temperature, dtype=np.single)

    if random_seed is not None:
        inputs["random_seed"] = np.full(prompts.shape, random_seed, dtype=np.single)

    if stop_words_list is not None:
        stop_words_list = np.char.encode(stop_words_list, "utf-8")
        inputs["stop_words_list"] = np.full((prompts.shape[0], len(stop_words_list)), stop_words_list)

    if bad_words_list is not None:
        bad_words_list = np.char.encode(bad_words_list, "utf-8")
        inputs["bad_words_list"] = np.full((prompts.shape[0], len(bad_words_list)), bad_words_list)

    if no_repeat_ngram_size is not None:
        inputs["no_repeat_ngram_size"] = np.full(prompts.shape, no_repeat_ngram_size, dtype=np.single)

    if task_id is not None:
        task_id = np.char.encode(task_id, "utf-8")
        inputs["task_id"] = np.full((prompts.shape[0], len([task_id])), task_id)

    if lora_uids is not None:
        lora_uids = np.char.encode(lora_uids, "utf-8")
        inputs["lora_uids"] = np.full((prompts.shape[0], len(lora_uids)), lora_uids)

    with ModelClient(url, model, init_timeout_s=init_timeout) as client:
        result_dict = client.infer_batch(**inputs)

    return result_dict
