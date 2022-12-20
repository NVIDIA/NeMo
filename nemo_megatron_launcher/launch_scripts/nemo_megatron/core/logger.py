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

import logging
import os
from logging import config
from typing import Union

# provide a way to change level through NEMO_MEGATRON_LOG_LEVEL environment variable:
# ...
LOG_VARNAME = "NEMO_MEGATRON_LOG_LEVEL"
level_str = os.environ.get(LOG_VARNAME, "INFO").upper()
level: Union[int, str] = level_str if not level_str.isdigit() else int(level_str)

CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {"nemo_megatron_basic": {"format": "%(name)s %(levelname)s (%(asctime)s) - %(message)s"}},
    "handlers": {
        "nemo_megatron_out": {
            "class": "logging.StreamHandler",
            "level": "DEBUG",
            "formatter": "nemo_megatron_basic",
            "stream": "ext://sys.stdout",
        },
        "nemo_megatron_err": {
            "class": "logging.StreamHandler",
            "level": "WARNING",
            "formatter": "nemo_megatron_basic",
            "stream": "ext://sys.stderr",
        },
    },
    "loggers": {"nemo_megatron": {"handlers": ["nemo_megatron_err", "nemo_megatron_out"], "level": level}},
}


if level != "NOCONFIG":
    logging.config.dictConfig(CONFIG)


def get_logger() -> logging.Logger:
    return logging.getLogger("NEMO_MEGATRON")


def exception(*args: str) -> None:
    get_logger().exception(*args)


def warning(*args: str) -> None:
    get_logger().warning(*args)


logger = get_logger()
