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
from functools import partial
from logging import LogRecord
from typing import Optional

logger = logging.getLogger(__name__)


def warning_filter(record: LogRecord) -> bool:
    if record.levelno == logging.WARNING:
        return False

    return True


def module_filter(record: LogRecord, modules_to_filter: list[str]) -> bool:
    for module in modules_to_filter:
        if record.name.startswith(module):
            return False
    return True


def add_filter_to_all_loggers(filter):
    """
    Adds the specified filter to all existing loggers.

    Args:
        handler: A logging handler instance to add to all loggers
    """
    # Get the root logger
    root = logging.getLogger()
    root.addFilter(filter)

    # Add handler to all existing loggers
    for logger_name in logging.root.manager.loggerDict:
        logger = logging.getLogger(logger_name)
        logger.addFilter(filter)


def setup_logging(
    logging_level: int = logging.INFO,
    filter_warning: bool = True,
    modules_to_filter: Optional[list[str]] = None,
    set_level_for_all_loggers: bool = False,
) -> None:
    """Sets the logging level and filters.

    Precedence:
    1. Config argument `logger_config.logging_level`
    2. Env var `TRON_LOGGING_LEVEL`
    3. Default logging level (INFO)

    Returns: None
    """
    env_logging_level = os.getenv("TRON_LOGGING_LEVEL", None)
    if env_logging_level is not None:
        logging_level = int(env_logging_level)

    logger.info(f"Setting logging level to {logging_level}")
    logging.getLogger().setLevel(logging_level)

    for _logger_name in logging.root.manager.loggerDict:
        if _logger_name.startswith("nemo") or set_level_for_all_loggers:
            _logger = logging.getLogger(_logger_name)
            _logger.setLevel(logging_level)

    if filter_warning:
        add_filter_to_all_loggers(warning_filter)
    if modules_to_filter:
        add_filter_to_all_loggers(partial(module_filter, modules_to_filter=modules_to_filter))
