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
from typing import Any, Dict, Mapping, Optional

import numpy as np
import torch
import wandb
from torch.utils import tensorboard as pt_tb

import nemo
from nemo.core import callbacks as nemo_callbacks

logging = nemo.logging


class Metric(abc.ABC):
    def clear(self) -> None:
        """Clears cache."""

        pass

    def batch(self, tensors) -> None:
        """Processes single batch.

        We feed pytorch tensors rather then numpy ones because metrics could be calculated on GPU.

        """

        pass

    def final(self) -> Any:
        """Finalizes calculation with optional dict of values (or anything else)."""

        return None

    def log(self, prefix, step, final_output=None) -> None:
        # Default "str -> float" dict log
        for key, val in (final_output or {}).items():
            # tb_writer.add_scalar(f'{prefix}/{key}', val, step)
            wandb.log({f'{prefix}/{key}': val}, step=step)

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


class MaskUsage(Metric):
    KEY = 'mask-usage'

    def __init__(self):
        super().__init__()

        self._values = None

    def clear(self) -> None:
        self._values = []

    def batch(self, tensors) -> None:
        mask = tensors.mask
        self._values.append(mask.sum().item() / mask.numel())

    def final(self) -> Mapping[str, float]:
        return {self.KEY: np.array(self._values).mean()}


def metric_by(name) -> Metric:
    for cls in Metric.__subclasses__():
        if hasattr(cls, 'KEY') and cls.KEY == name:
            return cls()

    raise ValueError("No such metric key.")


class TrainLogger(nemo_callbacks.SimpleLossLoggerCallback):
    def __init__(self, tensors, metrics, freq, mu=1.0, prefix='train', warmup=None):
        self._cache = collections.defaultdict(float)

        metrics = [metric_by(metric) if isinstance(metric, str) else metric for metric in metrics]
        for metric in metrics:
            metric.clear()

        def print_func(pt_tensors):
            kv_tensors = argparse.Namespace(**dict(zip(tensors.keys(), pt_tensors)))

            for metric in metrics:
                # Processes single batch for single GPU (current NeMo limitation).
                metric.batch(kv_tensors)

        # noinspection PyUnusedLocal
        def log_to_tb_func(unused1, unused2, step):
            del unused1
            del unused2

            output = {}
            for metric in metrics:
                final_output = metric.final()
                # No `mu` discounting for TB, it's already there.
                if warmup is None or step >= warmup:
                    metric.log(prefix, step, final_output)
                metric.clear()

                if isinstance(final_output, dict):
                    for k, v in final_output.items():
                        self._cache[k] = (1 - mu) * self._cache[k] + mu * v
                        output[k] = self._cache[k]

            output = {f'{prefix}/{k}': v for k, v in output.items()}
            logging.info(json.dumps(output, indent=4))

        super().__init__(
            tensors=list(tensors.values()),
            print_func=print_func,
            log_to_tb_func=log_to_tb_func,
            step_freq=freq,
            tb_writer=pt_tb.SummaryWriter(),  # Fake TB
        )


class EvalLogger(nemo_callbacks.EvaluatorCallback):
    def __init__(self, tensors, metrics, freq, prefix='eval', warmup=None):
        metrics = [metric_by(metric) if isinstance(metric, str) else metric for metric in metrics]
        for metric in metrics:
            metric.clear()

        # noinspection PyUnusedLocal
        def user_iter_callback(pt_tensors, global_var_dict):
            del global_var_dict

            del pt_tensors['IS_FROM_DIST_EVAL']
            kv_tensors = [argparse.Namespace(**dict(zip(tensors.keys(), ts))) for ts in zip(*pt_tensors.values())]

            for metric in metrics:
                for kv_tensors1 in kv_tensors:
                    metric.batch(kv_tensors1)

        # noinspection PyUnusedLocal
        def tb_writer_func(unused1, unused2, step):
            del unused1
            del unused2

            output = {}
            for metric in metrics:
                final_output = metric.final()
                if warmup is None or step >= warmup:
                    metric.log(prefix, step, final_output)
                metric.clear()

                if isinstance(final_output, dict):
                    output.update(final_output)

            output = {f'{prefix}/{k}': v for k, v in output.items()}
            logging.info(json.dumps(output, indent=4))

        super().__init__(
            eval_tensors=list(tensors.values()),
            user_iter_callback=user_iter_callback,
            user_epochs_done_callback=lambda _: dict(),  # Not None.
            tb_writer=pt_tb.SummaryWriter(),  # Fake TB
            tb_writer_func=tb_writer_func,
            eval_step=freq,
        )
