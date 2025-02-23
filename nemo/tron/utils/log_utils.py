# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
import sys
from logging import LogRecord, StreamHandler

from nemo.tron.config import ConfigContainer

BLACKLISTED_MODULES = ["torch.distributed"]
logger = logging.getLogger(__name__)


class CustomHandler(StreamHandler):
    """
    Custom handler to filter out logging from code outside of
    Megatron Core, and dump to stdout.
    """

    def __init__(self):
        super().__init__(stream=sys.stdout)

    def filter(self, record: LogRecord) -> bool:
        # Prevent log entries that come from the blacklisted modules
        # through (e.g., PyTorch Distributed).
        for blacklisted_module in BLACKLISTED_MODULES:
            if record.name.startswith(blacklisted_module):
                return False
        return True


def setup_logging(cfg: ConfigContainer) -> None:
    """Sets the default logging level based on cmdline args and env vars.

    Precedence:
    1. Command line argument `--logging-level`
    2. Env var `MEGATRON_LOGGING_LEVEL`
    3. Default logging level (INFO)

    Returns: None
    """
    logging_level = None
    env_logging_level = os.getenv("MEGATRON_LOGGING_LEVEL", None)
    if env_logging_level is not None:
        logging_level = int(env_logging_level)
    if cfg.logger_config.logging_level is not None:
        logging_level = cfg.logger_config.logging_level

    if logging_level is not None:
        logger.info(f"Setting logging level to {logging_level}")
        logging.getLogger().setLevel(logging_level)

        # Make default logging level INFO, but filter out all log messages not from MCore.
        logging.basicConfig(handlers=[CustomHandler()], level=logging.INFO)
