# ! /usr/bin/python
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

import abc
import argparse
import collections
import glob
import json
import os
import pprint
from collections import namedtuple
from typing import Mapping

import numpy as np

import nemo
from nemo.core import callbacks as nemo_callbacks

logging = nemo.logging


class Metric(abc.ABC):
    @abc.abstractmethod
    def clear(self) -> None:
        pass

    @abc.abstractmethod
    def batch(self, tensors) -> None:
        pass

    @abc.abstractmethod
    def final(self) -> Mapping[str, float]:
        pass

    def __call__(self, tensors):
        self.clear()
        self.batch(tensors)
        output = self.final()
        self.clear()
        return output


class Loss(Metric):
    KEY = 'loss'

    def __init__(self):
        super().__init__()

        self._values = None

    def clear(self) -> None:
        self._values = []

    def batch(self, tensors) -> None:
        self._values.append(tensors.loss.item())

    def final(self) -> Mapping[str, float]:
        return {self.KEY: np.array(self._values).mean()}


def metric_by(name) -> Metric:
    for cls in Metric.__subclasses__():
        if hasattr(cls, 'KEY') and cls.KEY == name:
            return cls()

    raise ValueError("No such metric key.")


class TrainLogger(nemo_callbacks.SimpleLossLoggerCallback):
    def __init__(self, tensors, metrics, freq, tb_writer, mu=0.99, prefix='train'):
        self._cache = collections.defaultdict(float)

        metrics = [metric_by(metric) if isinstance(metric, str) else metric for metric in metrics]

        def print_func(pt_tensors):
            kv_tensors = argparse.Namespace(**dict(zip(tensors.keys(), pt_tensors)))

            for metric in metrics:
                # We feed pytorch tensors rather then numpy ones because metrics could be calculated on GPU.
                for k, v in metric(kv_tensors).items():
                    self._cache[k] = (1 - mu) * self._cache[k] + mu * v

        # noinspection PyUnusedLocal
        def get_tb_values(*args, **kwargs):
            output = {f'{prefix}/{k}': v for k, v in self._cache.items()}
            logging.info(json.dumps(output, indent=4))
            return list((k, np.array(v)) for k, v in output.items())

        super().__init__(
            tensors=list(tensors.values()),
            print_func=print_func,
            get_tb_values=get_tb_values,
            step_freq=freq,
            tb_writer=tb_writer,
        )


class EvalLogger(nemo_callbacks.EvaluatorCallback):
    def __init__(self, tensors, metrics, freq, tb_writer, prefix='eval'):
        metrics = [metric_by(metric) if isinstance(metric, str) else metric for metric in metrics]
        for metric in metrics:
            metric.clear()

        # noinspection PyUnusedLocal
        def user_iter_callback(pt_tensors, global_var_dict):
            del global_var_dict

            del pt_tensors['IS_FROM_DIST_EVAL']
            pt_tensors = [t[0] for t in pt_tensors.values()]
            kv_tensors = argparse.Namespace(**dict(zip(tensors.keys(), pt_tensors)))

            for metric in metrics:
                metric.batch(kv_tensors)

        # noinspection PyUnusedLocal
        def user_epochs_done_callback(global_var_dict):
            del global_var_dict

            output = {}
            for metric in metrics:
                output.update(metric.final())
                metric.clear()

            output = {f'{prefix}/{k}': v for k, v in output.items()}
            logging.info(json.dumps(output, indent=4))
            return output

        super().__init__(
            eval_tensors=list(tensors.values()),
            user_iter_callback=user_iter_callback,
            user_epochs_done_callback=user_epochs_done_callback,
            tb_writer=tb_writer,
            eval_step=freq,
        )
