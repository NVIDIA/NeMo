#! /usr/bin/python
# -*- coding: utf-8 -*-

from nemo.utils.formatters.base import BaseFormatter

__all__ = ["StdFileFormatter"]


class StdFileFormatter(BaseFormatter):
    DEFAULT_FORMAT = '%(levelname)8s: %(message)s'.format(header="DLLogger")
