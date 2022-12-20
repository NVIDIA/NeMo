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

import argparse

from lm_eval import tasks

try:
    from nemo.utils.get_rank import is_global_rank_zero
except ModuleNotFoundError:
    print("Importing NeMo module failed, checkout the NeMo submodule")


def parse_args(parser_main):
    # parser = argparse.ArgumentParser()
    parser = parser_main.add_argument_group(title="download-tasks")
    parser.add_argument("--tasks", default="all_tasks")
    parser.add_argument("--cache_dir", default="")
    # return parser.parse_args()
    return parser_main


def main():
    parser = argparse.ArgumentParser()
    args, unknown_args = parse_args(parser).parse_known_args()
    if args.tasks == "all_tasks":
        task_names = tasks.ALL_TASKS
    else:
        task_names = args.tasks.split(",")
    if is_global_rank_zero():
        print("***** Downloading tasks data...")
        tasks.get_task_dict(task_names, args.cache_dir)
        print("***** Tasks data downloaded.")


if __name__ == "__main__":
    main()
