# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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
import logging

import submitit

H2MIN = 60
MIN2S = 60
H2S = 60 * 60


def config_logger(verbose: bool = False):
    log_level = logging.INFO if not verbose else logging.DEBUG
    log_format = "%(asctime)s %(levelname)s %(name)s %(message)s"
    logging.basicConfig(level=log_level, format=log_format)
    logging.getLogger("sh.command").setLevel(logging.WARNING)
    logging.getLogger("sh.stream_bufferer").setLevel(logging.WARNING)
    logging.getLogger("sh.streamreader").setLevel(logging.WARNING)


def monkeypatch_submitit():
    from .slurm import PyxisInfoWatcher

    submitit.Job._results_timeout_s = 2 * H2S
    submitit.SlurmJob.watcher = PyxisInfoWatcher(delay_s=5)


def get_YN_input(prompt, default):
    yes = {"yes", "ye", "y"}
    no = {"no", "n"}
    return get_choice_string_input(prompt, default, yes, no)


def get_choice_string_input(prompt, default, first_choice, second_choice):
    choice = input(prompt).lower()
    if choice in first_choice:
        return True
    elif choice in second_choice:
        return False
    elif choice is "":  # Just enter pressed
        return default
    else:
        print("input not recognized, please try again: ")
        return get_choice_string_input(prompt, default, first_choice, second_choice)
