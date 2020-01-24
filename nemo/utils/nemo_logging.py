#! /usr/bin/python
# -*- coding: utf-8 -*-

# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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

import os
import sys

import inspect
import warnings

from contextlib import contextmanager

import threading
import logging as _logging

from nemo.constants import DLLOGGER_ENV_VARNAME_SAVE_LOGS_TO_DIR
from nemo.constants import DLLOGGER_ENV_VARNAME_MLPERF_COMPLIANT
from nemo.constants import DLLOGGER_ENV_VARNAME_REDIRECT_LOGS_TO_STDERR

from nemo.utils.formatters import StdOutFormatter
from nemo.utils.formatters import MLPerfFormatter

from nemo.utils.formatters import StdFileFormatter
from nemo.utils.formatters import MLPerfFileFormatter

from nemo.utils.metaclasses import SingletonMetaClass

# from nemo.utils import MPI_rank_and_size

from nemo.utils import get_envbool
from nemo.utils import get_env

__all__ = [
    'Logger',
]


class Logger(metaclass=SingletonMetaClass):

    # Level 0
    NOTSET = _logging.NOTSET

    # Level 10
    DEBUG = _logging.DEBUG

    # Level 20
    INFO = _logging.INFO

    # Level 30
    WARNING = _logging.WARNING

    # Level 40
    ERROR = _logging.ERROR

    # Level 50
    CRITICAL = _logging.CRITICAL

    _level_names = {
        0: 'NOTSET',
        10: 'DEBUG',
        20: 'INFO',
        30: 'WARNING',
        40: 'ERROR',
        50: 'CRITICAL',
    }

    def __init__(self):

        self._logger = None
        self._logger_lock = threading.Lock()

        self._handlers = dict()

        self._log_dir = get_env(DLLOGGER_ENV_VARNAME_SAVE_LOGS_TO_DIR, "")

        self.old_warnings_showwarning = None

        if MPI_rank_and_size()[0] == 0:
            self._define_logger()

    def _define_logger(self):

        # Use double-checked locking to avoid taking lock unnecessarily.
        if self._logger is not None:
            return self._logger

        with self._logger_lock:

            try:
                # Scope the TensorFlow logger to not conflict with users' loggers.
                self._logger = _logging.getLogger('nemo_logger')
                self.reset_stream_handler()

            finally:
                self.set_verbosity(verbosity_level=Logger.INFO)

            self._logger.propagate = False

    def reset_stream_handler(self):

        if self._logger is None:
            raise RuntimeError("Impossible to set handlers if the Logger is not predefined")

        # ======== Remove Handler if already existing ========

        try:
            self._logger.removeHandler(self._handlers["stream_stdout"])
        except KeyError:
            pass

        try:
            self._logger.removeHandler(self._handlers["stream_stderr"])
        except KeyError:
            pass

        # ================= Streaming Handler =================

        # Add the output handler.
        if get_envbool(DLLOGGER_ENV_VARNAME_REDIRECT_LOGS_TO_STDERR, False):
            self._handlers["stream_stdout"] = _logging.StreamHandler(sys.stderr)

        else:
            self._handlers["stream_stdout"] = _logging.StreamHandler(sys.stdout)
            self._handlers["stream_stdout"].addFilter(
                lambda record: record.levelno <= _logging.INFO
            )

            self._handlers["stream_stderr"] = _logging.StreamHandler(sys.stderr)
            self._handlers["stream_stderr"].addFilter(
                lambda record: record.levelno > _logging.INFO
            )

        if get_envbool(DLLOGGER_ENV_VARNAME_MLPERF_COMPLIANT, False):
            Formatter = MLPerfFormatter
        else:
            Formatter = StdOutFormatter

        self._handlers["stream_stdout"].setFormatter(Formatter())
        self._logger.addHandler(self._handlers["stream_stdout"])

        try:
            self._handlers["stream_stderr"].setFormatter(Formatter())
            self._logger.addHandler(self._handlers["stream_stderr"])
        except KeyError:
            pass

    def get_verbosity(self):
        """Return how much logging output will be produced."""
        if self._logger is not None:
            return self._logger.getEffectiveLevel()

    def set_verbosity(self, verbosity_level):
        """Sets the threshold for what messages will be logged."""
        if self._logger is not None:
            self._logger.setLevel(verbosity_level)

            for handler in self._logger.handlers:
                handler.setLevel(verbosity_level)

    @contextmanager
    def temp_verbosity(self, verbosity_level):
        """Sets the a temporary threshold for what messages will be logged."""

        if self._logger is not None:

            old_verbosity = self.get_verbosity()

            try:
                self.set_verbosity(verbosity_level)
                yield

            finally:
                self.set_verbosity(old_verbosity)

        else:
            try:
                yield

            finally:
                pass

    def captureWarnings(self, capture):
        """
        If capture is true, redirect all warnings to the logging package.
        If capture is False, ensure that warnings are not redirected to logging
        but to their original destinations.
        """

        if self._logger is not None:

            if capture and self.old_warnings_showwarning is None:
                # Backup Method
                self.old_warnings_showwarning = warnings.showwarning
                warnings.showwarning = self._showwarning

            elif not capture and self.old_warnings_showwarning is not None:
                # Restore Method
                warnings.showwarning = self.old_warnings_showwarning
                self.old_warnings_showwarning = None

    def _showwarning(self, message, category, filename, lineno, file=None, line=None):
        """
        Implementation of showwarnings which redirects to logging.
        It will call warnings.formatwarning and will log the resulting string
        with level logging.WARNING.
        """
        s = warnings.formatwarning(message, category, filename, lineno, line)
        self.warning("%s", s)

    def debug(self, msg, *args, **kwargs):
        """
        Log 'msg % args' with severity 'DEBUG'.

        To pass exception information, use the keyword argument exc_info with
        a true value, e.g.

        logger.debug("Houston, we have a %s", "thorny problem", exc_info=1)
        """
        if self._logger is not None and self._logger.isEnabledFor(Logger.DEBUG):
            self._logger._log(Logger.DEBUG, msg, args, **kwargs)

    def info(self, msg, *args, **kwargs):
        """
        Log 'msg % args' with severity 'INFO'.

        To pass exception information, use the keyword argument exc_info with
        a true value, e.g.

        logger.info("Houston, we have a %s", "interesting problem", exc_info=1)
        """
        if self._logger is not None and self._logger.isEnabledFor(Logger.INFO):
            self._logger._log(Logger.INFO, msg, args, **kwargs)

    def warning(self, msg, *args, **kwargs):
        """
        Log 'msg % args' with severity 'WARNING'.

        To pass exception information, use the keyword argument exc_info with
        a true value, e.g.

        logger.warning("Houston, we have a %s", "bit of a problem", exc_info=1)
        """
        if self._logger is not None and self._logger.isEnabledFor(Logger.WARNING):
            self._logger._log(Logger.WARNING, msg, args, **kwargs)

    def error(self, msg, *args, **kwargs):
        """
        Log 'msg % args' with severity 'ERROR'.

        To pass exception information, use the keyword argument exc_info with
        a true value, e.g.

        logger.error("Houston, we have a %s", "major problem", exc_info=1)
        """
        if self._logger is not None and self._logger.isEnabledFor(Logger.ERROR):
            self._logger._log(Logger.ERROR, msg, args, **kwargs)

    def critical(self, msg, *args, **kwargs):
        """
        Log 'msg % args' with severity 'CRITICAL'.

        To pass exception information, use the keyword argument exc_info with
        a true value, e.g.

        logger.critical("Houston, we have a %s", "major disaster", exc_info=1)
        """
        if self._logger is not None and self._logger.isEnabledFor(Logger.CRITICAL):
            self._logger._log(Logger.CRITICAL, msg, args, **kwargs)


# Necessary to catch the correct caller
_logging._srcfile = os.path.normcase(inspect.getfile(Logger.__class__))
