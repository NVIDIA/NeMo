# ! /usr/bin/python
# -*- coding: utf-8 -*-

# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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

__all__ = [
    "ActionCallback",
    "ModuleSaverCallback",
    "SimpleLossLoggerCallback",
    "EvaluatorCallback",
    "ValueSetterCallback",
    "UnfreezeCallback",
    "WandbCallback",
]

import datetime
import glob
import os
import sys
import time
from abc import ABC, abstractmethod
from collections import namedtuple

from nemo.utils import logging
from nemo.utils.decorators import deprecated

try:
    import wandb

    _WANDB_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    _WANDB_AVAILABLE = False


class ActionCallback(ABC):
    """Abstract interface for callbacks.
    """

    def __init__(self):
        self._registered_tensors = {}
        self._action = None

    @property
    def step(self):
        return self.action.step

    @property
    def epoch_num(self):
        return self.action.epoch_num

    @property
    def registered_tensors(self):
        return self._registered_tensors

    @property
    def local_rank(self):
        return self.action.local_rank

    @property
    def global_rank(self):
        return self.action.global_rank

    @property
    def action(self):
        return self._action

    @action.setter
    def action(self, action_obj):
        self._action = action_obj

    @property
    def logger(self):
        return logging

    def on_action_start(self):
        pass

    def on_action_end(self):
        pass

    def on_epoch_start(self):
        pass

    def on_epoch_end(self):
        pass

    def on_iteration_start(self):
        pass

    def on_iteration_end(self):
        pass


class ModuleSaverCallback(ActionCallback):
    """
    For callback documentation: please see
    https://nvidia.github.io/NeMo/tutorials/callbacks.html
    """

    @deprecated(version="0.12", explanation="The callback section of NeMo has been updated.")
    def __init__(
        self, save_modules_list, step_freq=1000, folder=None, checkpoints_to_keep=4,
    ):
        super().__init__()
        self._save_modules_list = save_modules_list
        self._folder = folder
        self._step_freq = step_freq
        self._ckpt2keep = checkpoints_to_keep
        self._saved_ckpts = []

    def on_iteration_end(self):
        step = self.step
        if (
            self._step_freq > 0
            and step % self._step_freq == 0
            and step > 0
            and (self.global_rank is None or self.global_rank == 0)
        ):
            for m in self._save_modules_list:
                class_name = m.__class__.__name__
                uid = m.unique_instance_id
                fn = f"{class_name}_{uid}-STEP-{step}.pt"
                if self._folder is None:
                    file_name = fn
                else:
                    file_name = os.path.join(self._folder, fn)
                logging.info(f"Saving module {class_name} in {file_name}")
                m.save_to(file_name)
                logging.info("Saved.")
            self._saved_ckpts.append(f'-{self.step}.pt')
            if len(self._saved_ckpts) > self._ckpt2keep:
                for end in self._saved_ckpts[: -self._ckpt2keep]:
                    for file in glob.glob(f'{self._folder}/*{end}'):
                        os.remove(file)
                self._saved_ckpts = self._saved_ckpts[-self._ckpt2keep :]

    def on_action_end(self):
        step = self.step
        if self.global_rank is None or self.global_rank == 0:
            for m in self._save_modules_list:
                class_name = m.__class__.__name__
                uid = m.unique_instance_id
                fn = f"{class_name}_{uid}-STEP-{step}.pt"
                if self._folder is None:
                    file_name = fn
                else:
                    file_name = os.path.join(self._folder, fn)
                logging.info(f"Saving module {class_name} in {file_name}")
                m.save_to(file_name)
                logging.info("Saved.")


class SimpleLossLoggerCallback(ActionCallback):
    """
    For callback documentation: please see
    https://nvidia.github.io/NeMo/tutorials/callbacks.html
    """

    @deprecated(version="0.12", explanation="The callback section of NeMo has been updated.")
    def __init__(
        self, tensors, print_func=None, get_tb_values=None, log_to_tb_func=None, step_freq=25, tb_writer=None,
    ):

        super().__init__()
        if not isinstance(tensors, list):
            tensors = [tensors]
        self._tensors = tensors
        self._print_func = print_func
        self._get_tb_values = get_tb_values
        self._log_to_tb_func = log_to_tb_func
        self._step_freq = step_freq
        self._swriter = tb_writer
        self._start_time = None
        self._last_epoch_start = None
        self._last_iter_start = None

    @property
    def tensors(self):
        return self._tensors

    def on_action_start(self):
        if self.global_rank is None or self.global_rank == 0:
            logging.info("Starting .....")
            self._start_time = time.time()

    def on_action_end(self):
        if self.global_rank is None or self.global_rank == 0:
            if self._swriter is not None:
                self._swriter.close()
            delta = datetime.timedelta(seconds=(time.time() - self._start_time))
            logging.info("Done in %s", delta)

    def on_epoch_start(self):
        if self.global_rank is None or self.global_rank == 0:
            logging.info(f"Starting epoch {self.epoch_num}")
            self._last_epoch_start = time.time()

    def on_epoch_end(self):
        if self.global_rank is None or self.global_rank == 0:
            step = self.step

            delta = datetime.timedelta(seconds=(time.time() - self._last_epoch_start))
            logging.info(f"Finished epoch {self.epoch_num} in {delta}")

            if self._swriter is not None:
                value = self.epoch_num
                self._swriter.add_scalar('misc/epoch', value, step)
                value = time.time() - self._last_epoch_start
                self._swriter.add_scalar('misc/epoch_time', value, step)

    def on_iteration_start(self):
        if self.global_rank is None or self.global_rank == 0:
            self._last_iter_start = time.time()

    def on_iteration_end(self):
        if self.global_rank is None or self.global_rank == 0:
            step = self.step
            if step % self._step_freq == 0:
                tensor_values = [self.registered_tensors[t.unique_name] for t in self.tensors]
                logging.info(f"Step: {step}")
                if self._print_func:
                    self._print_func(tensor_values)
                sys.stdout.flush()
                if self._swriter is not None:
                    if self._get_tb_values:
                        tb_objects = self._get_tb_values(tensor_values)
                        for name, value in tb_objects:
                            value = value.item()
                            self._swriter.add_scalar(name, value, step)
                    if self._log_to_tb_func:
                        self._log_to_tb_func(self._swriter, tensor_values, step)
                    run_time = time.time() - self._last_iter_start
                    self._swriter.add_scalar('misc/step_time', run_time, step)
                run_time = time.time() - self._last_iter_start
                logging.info(f"Step time: {run_time} seconds")

                # To keep support in line with the removal of learning rate logging from inside actions, log learning
                # rate to tensorboard. However it now logs ever self._step_freq as opposed to every step
                if self._swriter is not None:
                    self._swriter.add_scalar('param/lr', self.learning_rate, step)


class EvaluatorCallback(ActionCallback):
    """
    For callback documentation: please see
    https://nvidia.github.io/NeMo/tutorials/callbacks.html
    """

    def __init__(
        self,
        eval_tensors,
        user_iter_callback,
        user_epochs_done_callback,
        tb_writer=None,
        tb_writer_func=None,
        eval_step=1,
        eval_epoch=None,
        wandb_name=None,
        wandb_project=None,
        eval_at_start=True,
    ):
        # TODO: Eval_epoch currently does nothing
        if eval_step is None and eval_epoch is None:
            raise ValueError("Either eval_step or eval_epoch must be set. " f"But got: {eval_step} and {eval_epoch}")
        if (eval_step is not None and eval_step <= 0) or (eval_epoch is not None and eval_epoch <= 0):
            raise ValueError(f"Eval_step and eval_epoch must be > 0." f"But got: {eval_step} and {eval_epoch}")
        super().__init__()
        self._eval_tensors = eval_tensors
        self._swriter = tb_writer
        self._tb_writer_func = tb_writer_func
        self._eval_frequency = eval_step
        self._eval_at_start = eval_at_start
        # will be passed to callbacks below
        self._global_var_dict = {}

        # Callbacks
        self.user_iter_callback = user_iter_callback
        self.user_done_callback = user_epochs_done_callback

        # Weights and biases
        self._wandb_project = wandb_project
        self._wandb_name = wandb_name

    @property
    def eval_tensors(self):
        return self._eval_tensors

    @property
    def tb_writer_func(self):
        return self._tb_writer_func

    @property
    def swriter(self):
        return self._swriter

    def on_epoch_end(self):
        pass

    def on_iteration_end(self):
        if self.step == 0 and not self._eval_at_start:
            return
        if self.step % self._eval_frequency == 0:
            if self.global_rank == 0 or self.global_rank is None:
                logging.info('Doing Evaluation ' + '.' * 30)
            start_time = time.time()
            self.action._eval(self._eval_tensors, self, self.step)
            elapsed_time = time.time() - start_time
            if self.global_rank == 0 or self.global_rank is None:
                logging.info(f'Evaluation time: {elapsed_time} seconds')

    def on_action_start(self):
        if self.global_rank is None or self.global_rank == 0:
            if self._wandb_name is not None or self._wandb_project is not None:
                if _WANDB_AVAILABLE and wandb.run is None:
                    wandb.init(name=self._wandb_name, project=self._wandb_project)
                elif _WANDB_AVAILABLE and wandb.run is not None:
                    logging.info("Re-using wandb session")
                else:
                    logging.error("Could not import wandb. Did you install it (pip install --upgrade wandb)?")
                    logging.info("Will not log data to weights and biases.")
                    self._wandb_name = None
                    self._wandb_project = None

    def on_action_end(self):
        step = self.step
        if self.global_rank == 0 or self.global_rank is None:
            logging.info('Final Evaluation ' + '.' * 30)
        start_time = time.time()
        self.action._eval(self._eval_tensors, self, step)
        elapsed_time = time.time() - start_time
        if self.global_rank == 0 or self.global_rank is None:
            logging.info(f'Evaluation time: {elapsed_time} seconds')

    def clear_global_var_dict(self):
        self._global_var_dict = {}

    def wandb_log(self, tensors_logged):
        if self._wandb_name is not None and _WANDB_AVAILABLE:
            wandb.log(tensors_logged, step=self.step)


_Policy = namedtuple('Policy', 'method start end')


class _Method(ABC):
    """ Classes inherited from _Method are used for
    ValueSetterCallback below
    """

    @abstractmethod
    def __call__(self, step, total_steps):
        pass


class _Const(_Method):
    def __init__(self, value):
        super().__init__()

        self.value = value

    def __call__(self, step, total_steps):
        return self.value


class _Linear(_Method):
    def __init__(self, a, b):
        super().__init__()
        self.a, self.b = a, b

    def __call__(self, step, total_steps):
        return self.a + (step / (total_steps - 1)) * (self.b - self.a)


_Method.Const = _Const
_Method.Linear = _Linear


class ValueSetterCallback(ActionCallback):
    Policy = _Policy
    Method = _Method

    @deprecated(version="0.12", explanation="The callback section of NeMo has been updated.")
    def __init__(self, module, arg_name, policies=None, total_steps=None, tb_writer=None):
        super().__init__()

        if policies is None:
            initial_value = getattr(module, arg_name)
            policies = [_Policy(method=Const(initial_value), start=0.0, end=1.0)]

        new_policies = []
        for p in policies:
            start, end = p.start, p.end
            if isinstance(start, float):
                start = int(start * total_steps)
            if isinstance(end, float):
                end = int(end * total_steps)
            new_policies.append(_Policy(p.method, start, end))
        policies = new_policies
        assert policies[0].start == 0
        assert policies[-1].end == total_steps

        self.module = module
        self.arg_name = arg_name
        self.policies = policies
        self.total_steps = total_steps
        self.tb_writer = tb_writer

        self.cur_i = 0

    def on_iteration_start(self):
        cur_policy = self.policies[self.cur_i]
        if self.step < cur_policy.end:
            step = self.step - cur_policy.start
            total_steps = cur_policy.end - cur_policy.start
            value = cur_policy.method(step, total_steps)
            setattr(self.module, self.arg_name, value)
            if self.tb_writer is not None:
                class_name = self.module.__class__.__name__
                name = f"param/{class_name}.{self.arg_name}"
                self.tb_writer.add_scalar(name, value, self.step)
        else:
            self.cur_i += 1
            self.on_iteration_start()


class UnfreezeCallback(ActionCallback):
    @deprecated(version="0.12", explanation="The callback section of NeMo has been updated.")
    def __init__(self, modules, start_epoch=0):
        super().__init__()

        self.modules = modules
        self.start_epoch = start_epoch

    def on_iteration_start(self):
        if self.epoch_num == self.start_epoch:
            for m in self.modules:
                m.unfreeze()


class WandbCallback(ActionCallback):
    """
    Log metrics to [Weights & Biases](https://docs.wandb.com/)
    """

    @deprecated(version="0.12", explanation="The callback section of NeMo has been updated.")
    def __init__(
        self, train_tensors=[], wandb_name=None, wandb_project=None, args=None, update_freq=25,
    ):
        """
        Args:
            train_tensors: list of tensors to evaluate and log based on training batches
            wandb_name: wandb experiment name
            wandb_project: wandb project name
            args: argparse flags - will be logged as hyperparameters
            update_freq: frequency with which to log updates
        """
        super().__init__()

        if not _WANDB_AVAILABLE:
            logging.error("Could not import wandb. Did you install it (pip install --upgrade wandb)?")

        self._update_freq = update_freq
        self._train_tensors = train_tensors
        self._name = wandb_name
        self._project = wandb_project
        self._args = args

    def on_action_start(self):
        if self.global_rank is None or self.global_rank == 0:
            if _WANDB_AVAILABLE and wandb.run is None:
                wandb.init(name=self._name, project=self._project)
                if self._args is not None:
                    wandb.config.update(self._args)
            elif _WANDB_AVAILABLE and wandb.run is not None:
                logging.info("Re-using wandb session")
            else:
                logging.error("Could not import wandb. Did you install it (pip install --upgrade wandb)?")
                logging.info("Will not log data to weights and biases.")
                self._update_freq = -1

    def on_iteration_end(self):
        # log training metrics
        if self.global_rank is None or self.global_rank == 0:
            if self.step % self._update_freq == 0 and self._update_freq > 0:
                tensors_logged = {t.name: self.registered_tensors[t.unique_name].cpu() for t in self._train_tensors}
                # Always log learning rate
                tensors_logged['LR'] = self.learning_rate
                self.wandb_log(tensors_logged)

    def on_epoch_start(self):
        if self.global_rank is None or self.global_rank == 0:
            self._last_epoch_start = time.time()

    def on_epoch_end(self):
        if self.global_rank is None or self.global_rank == 0:
            # always log epoch num and epoch_time
            epoch_time = time.time() - self._last_epoch_start
            self.wandb_log({"epoch": self.epoch_num, "epoch_time": epoch_time})

    def wandb_log(self, tensors_logged):
        if _WANDB_AVAILABLE:
            wandb.log(tensors_logged, step=self.step)
