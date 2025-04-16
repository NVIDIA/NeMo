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
from logging import Filter, LogRecord
from typing import Callable, List, Optional, Union

logger = logging.getLogger(__name__)


def warning_filter(record: LogRecord) -> bool:
    """Logging filter to exclude WARNING level messages.

    Args:
        record: The logging record to check.

    Returns:
        False if the record level is WARNING, True otherwise.
    """
    if record.levelno == logging.WARNING:
        return False

    return True


def module_filter(record: LogRecord, modules_to_filter: List[str]) -> bool:
    """Logging filter to exclude messages from specific modules.

    Args:
        record: The logging record to check.
        modules_to_filter: A list of module name prefixes to filter out.

    Returns:
        False if the record's logger name starts with any of the specified
        module prefixes, True otherwise.
    """
    for module in modules_to_filter:
        if record.name.startswith(module):
            return False
    return True


def add_filter_to_all_loggers(filter: Union[Filter, Callable[[LogRecord], bool]]) -> None:
    """Add a filter to the root logger and all existing loggers.

    Args:
        filter: A logging filter instance or callable to add.
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
    modules_to_filter: Optional[List[str]] = None,
    set_level_for_all_loggers: bool = False,
) -> None:
    """Set up logging level and filters for the application.

    Configures the logging level based on arguments, environment variables,
    or defaults. Optionally adds filters to suppress warnings or messages
    from specific modules.

    Logging Level Precedence:
    1. Env var `TRON_LOGGING_LEVEL`
    2. `logging_level` argument
    3. Default: `logging.INFO`

    Args:
        logging_level: The desired logging level (e.g., logging.INFO, logging.DEBUG).
        filter_warning: If True, adds a filter to suppress WARNING level messages.
        modules_to_filter: An optional list of module name prefixes to filter out.
        set_level_for_all_loggers: If True, sets the logging level for all existing
                                   loggers. If False (default), only sets the level
                                   for the root logger and loggers starting with 'nemo'.
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
