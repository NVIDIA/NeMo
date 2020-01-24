#! /usr/bin/python
# -*- coding: utf-8 -*-

from nemo.utils.formatters.base import BaseFormatter

__all__ = ["StdOutFormatter"]


class StdOutFormatter(BaseFormatter):
    DEFAULT_FORMAT = '%(color)s[NeMo] %(levelname)-8s: ' \
                     '%(end_color)s%(message)s'
