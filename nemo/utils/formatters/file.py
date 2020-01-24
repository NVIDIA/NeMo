#! /usr/bin/python
# -*- coding: utf-8 -*-

from dllogger.constants import MLPERF_TOKEN
from dllogger.constants import MLPERF_VERSION

from dllogger.formatters.base import BaseFormatter

__all__ = ["StdFileFormatter", "MLPerfFileFormatter"]


class StdFileFormatter(BaseFormatter):
    DEFAULT_FORMAT = '%(levelname)8s: %(message)s'.format(header="DLLogger")


class MLPerfFileFormatter(BaseFormatter):
    DEFAULT_FORMAT = '[{token}v{ver} %(asctime)s.%(msecs)011.7f %(module)7s:%(lineno)04d] '\
                     '%(levelname)8s: %(message)s'.format(token=MLPERF_TOKEN, ver=MLPERF_VERSION)
