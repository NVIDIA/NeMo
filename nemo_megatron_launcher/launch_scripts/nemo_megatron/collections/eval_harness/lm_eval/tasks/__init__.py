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

from pprint import pprint

from . import hellaswag, lambada, piqa, prompt, race, superglue, wikitext, winogrande
from .common import HFTask

########################################
# Translation tasks
########################################

# 6 total
gpt3_translation_benchmarks = {
    "wmt14": ["en-fr", "fr-en"],  # French
    "wmt16": ["en-ro", "ro-en", "de-en", "en-de"],  # German, Romanian
}

########################################
# All tasks
########################################


TASK_REGISTRY = {
    "lambada": lambada.LAMBADA,
    "boolq": superglue.BoolQ,
    "piqa": piqa.PiQA,
    "hellaswag": hellaswag.HellaSwag,
    "race": race.RACE,
    "wikitext2": wikitext.WikiText,
    "wikitext103": wikitext.WikiText103,
    "winogrande": winogrande.Winogrande,
}

PROMPT_TASK_REGISTRY = {"prompt": prompt.Prompt}

ALL_TASKS = sorted(list(TASK_REGISTRY))


def get_task(task_name):
    if task_name in TASK_REGISTRY:
        return TASK_REGISTRY[task_name]

    print("Available tasks:")
    pprint(TASK_REGISTRY)
    raise KeyError(f"Missing task {task_name}")


def get_task_dict(task_name_list, cache_dir):
    return {task_name: get_task(task_name)(cache_dir) for task_name in task_name_list}


def get_prompt_task(task_name):
    if task_name in PROMPT_TASK_REGISTRY:
        return PROMPT_TASK_REGISTRY[task_name]

    print("Available tasks:")
    pprint(PROMPT_TASK_REGISTRY)
    raise KeyError(f"Missing task {task_name}")


def get_prompt_task_dict(task_name_list, **kwargs):
    return {task_name: get_prompt_task(task_name)(**kwargs) for task_name in task_name_list}
