#! /usr/bin/python
# -*- coding: utf-8 -*-

from dllogger.constants import MLPERF_TOKEN
from dllogger.constants import MLPERF_VERSION

from dllogger.formatters.base import BaseFormatter

__all__ = ["StdOutFormatter", "MLPerfFormatter"]


class StdOutFormatter(BaseFormatter):
    DEFAULT_FORMAT = '%(color)s[DLLogger] %(levelname)-8s: %(end_color)s%(message)s'


class MLPerfFormatter(BaseFormatter):
    DEFAULT_FORMAT = '%(color)s[{token}v{ver} %(asctime)s.%(msecs)011.7f %(module)7s:%(lineno)04d] '\
                     '%(levelname)8s: %(end_color)s%(message)s'.format(token=MLPERF_TOKEN, ver=MLPERF_VERSION)
