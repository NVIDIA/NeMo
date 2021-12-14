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
import pathlib
import time

H2MIN = 60
MIN2S = 60
H2S = 60 * 60

CLUSTER_DIR_NAME = "cluster_workspace"
FS_SYNC_TIMEOUT_S = 30


def config_logger(workspace_path: pathlib.Path, verbose: bool = False):
    log_level = logging.INFO if not verbose else logging.DEBUG
    log_format = "%(asctime)s %(levelname)s %(name)s %(message)s"
    logging.basicConfig(level=log_level, format=log_format)

    # set requested level to each handler and DEBUG to root logger
    root_logger = logging.getLogger()
    for handler in root_logger.handlers:
        handler.setLevel(log_level)
    root_logger.setLevel(logging.DEBUG)

    # add file handler with DEBUG level to root logger
    log_path = workspace_path / "script.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter(log_format)
    file_handler = logging.FileHandler(log_path, mode="a")
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.DEBUG)
    root_logger.addHandler(file_handler)

    logging.getLogger("sh.command").setLevel(logging.WARNING)
    logging.getLogger("sh.stream_bufferer").setLevel(logging.WARNING)
    logging.getLogger("sh.streamreader").setLevel(logging.WARNING)


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
    elif choice == "":  # Just enter pressed
        return default
    else:
        print("input not recognized, please try again: ")
        return get_choice_string_input(prompt, default, first_choice, second_choice)


def wait_for(name, predicate_fn, timeout_s: int):
    step_s = 1
    while not predicate_fn():
        time.sleep(step_s)
        timeout_s -= step_s
        if timeout_s <= 0:
            raise TimeoutError(f"Wait for {name} failed (timeout_s={timeout_s})")
