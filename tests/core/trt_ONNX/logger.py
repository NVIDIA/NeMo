# ! /usr/bin/python
# -*- coding: utf-8 -*-

# =============================================================================
# Copyright 2020 NVIDIA. All Rights Reserved.
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
# =============================================================================

import enum
import inspect
import os
import sys
import time


class LogMode(enum.IntEnum):
    EACH = 0  # Log the message each time
    ONCE = 1  # Log the message only once. The same message will not be logged again.


class Logger(object):
    ULTRA_VERBOSE = -10
    VERBOSE = 0
    DEBUG = 10
    INFO = 20
    WARNING = 30
    ERROR = 40
    CRITICAL = 50

    def __init__(self, severity=INFO, colors=True):
        """
        Logger.

        Optional Args:
            severity (Logger.Severity): Messages below this severity are ignored.
        """
        self.severity = severity
        self.root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
        self.once_logged = set()
        self.colors = colors

    @staticmethod
    def severity_color_prefix(sev):
        prefix = "\033[1;"
        color = {
            Logger.ULTRA_VERBOSE: "90m",  # Gray
            Logger.VERBOSE: "90m",
            Logger.DEBUG: "90m",
            Logger.INFO: "92m",  # Green
            Logger.WARNING: "33m",  # Yellow
            Logger.ERROR: "31m",  # Red
            Logger.CRITICAL: "31m",
        }[sev]
        return prefix + color

    def assemble_message(self, message, stack_depth, prefix):
        module = inspect.getmodule(sys._getframe(stack_depth))
        # Handle logging from the top-level of a module.
        if not module:
            module = inspect.getmodule(sys._getframe(stack_depth - 1))
        filename = module.__file__
        filename = os.path.relpath(filename, self.root_dir)
        # If the file is not located in trt_smeagol, use its basename instead.
        if os.pardir in filename:
            filename = os.path.basename(filename)
        return "{:} ({:}) [{:}:{:}] {:}".format(
            prefix, time.strftime("%H:%M:%S"), filename, sys._getframe(stack_depth).f_lineno, message
        )

    # If once is True, the logger will only log this message a single time. Useful in loops.
    def log(self, message, severity, mode=False):
        PREFIX_LEN = 12
        if mode == LogMode.ONCE:
            if message[PREFIX_LEN:] in self.once_logged:
                return
            self.once_logged.add(message[PREFIX_LEN:])

        if severity >= self.severity:
            if self.colors:
                message = "{:}{:}\033[0m".format(Logger.severity_color_prefix(severity), message)
            print(message)

    def ultra_verbose(self, message, mode=LogMode.EACH):
        self.log(self.assemble_message(message, stack_depth=2, prefix="V"), Logger.VERBOSE, mode=mode)

    def verbose(self, message, mode=LogMode.EACH):
        self.log(self.assemble_message(message, stack_depth=2, prefix="V"), Logger.VERBOSE, mode=mode)

    def debug(self, message, mode=LogMode.EACH):
        self.log(self.assemble_message(message, stack_depth=2, prefix="D"), Logger.DEBUG, mode=mode)

    def info(self, message, mode=LogMode.EACH):
        self.log(self.assemble_message(message, stack_depth=2, prefix="I"), Logger.INFO, mode=mode)

    def warning(self, message, mode=LogMode.EACH):
        self.log(self.assemble_message(message, stack_depth=2, prefix="W"), Logger.WARNING, mode=mode)

    def error(self, message, mode=LogMode.EACH):
        self.log(self.assemble_message(message, stack_depth=2, prefix="E"), Logger.ERROR, mode=mode)

    # Like error, but immediately exits.
    def critical(self, message):
        full_msg = self.assemble_message(message, stack_depth=2, prefix="C")
        self.log(full_msg, Logger.CRITICAL)
        raise Exception(full_msg)


global G_LOGGER
G_LOGGER = Logger()
