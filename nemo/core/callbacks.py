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

import datetime
import glob
import os
import sys
import time
import warnings
from abc import ABC, abstractmethod
from collections import namedtuple
from typing import Callable, List, Union

import nemo
from nemo.utils import get_checkpoint_from_dir, logging
from nemo.utils.app_state import AppState

try:
    import wandb

    _WANDB_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    _WANDB_AVAILABLE = False


class NeMoCallback(ABC):
    """The base class for callbacks inside of NeMo. It contains no __init__ which children classes are responsible for.
    Each callback contains 8 functions which are called at different stages of train(). All functions must take as the
    first argument: the current action state. This state is a StateWrapper object.
    TODO: Add a link to documentation.
    """
    def on_train_start(self, state):
        pass

    def on_epoch_start(self, state):
        pass

    def on_batch_start(self, state):
        pass

    def on_step_start(self, state):
        pass

    def on_step_end(self, state):
        pass

    def on_batch_end(self, state):
        pass

    def on_epoch_end(self, state):
        pass

    def on_train_end(self, state):
        pass


def on_train_start(func):
    """A function decorator that wraps a Callable inside the NeMoCallback object and runs the function with the
    on_train_start callback event.
    """
    class NeMoCallbackWrapper(NeMoCallback):
        def __init__(self, my_func):
            self._func = my_func

        def on_train_start(self, state):
            self._func(state)

    return NeMoCallbackWrapper(func)


def on_epoch_start(func):
    """A function decorator that wraps a Callable inside the NeMoCallback object and runs the function with the
    on_epoch_start callback event.
    """
    class NeMoCallbackWrapper(NeMoCallback):
        def __init__(self, my_func):
            self._func = my_func

        def on_epoch_start(self, state):
            self._func(state)

    return NeMoCallbackWrapper(func)


def on_batch_start(func):
    """A function decorator that wraps a Callable inside the NeMoCallback object and runs the function with the
    on_batch_start callback event.
    """
    class NeMoCallbackWrapper(NeMoCallback):
        def __init__(self, my_func):
            self._func = my_func

        def on_batch_start(self, state):
            self._func(state)

    return NeMoCallbackWrapper(func)


def on_step_start(func):
    """A function decorator that wraps a Callable inside the NeMoCallback object and runs the function with the
    on_step_start callback event.
    """
    class NeMoCallbackWrapper(NeMoCallback):
        def __init__(self, my_func):
            self._func = my_func

        def on_step_start(self, state):
            self._func(state)

    return NeMoCallbackWrapper(func)


def on_step_end(func):
    """A function decorator that wraps a Callable inside the NeMoCallback object and runs the function with the
    on_step_end callback event.
    """
    class NeMoCallbackWrapper(NeMoCallback):
        def __init__(self, my_func):
            self._func = my_func

        def on_step_end(self, state):
            self._func(state)

    return NeMoCallbackWrapper(func)


def on_batch_end(func):
    """A function decorator that wraps a Callable inside the NeMoCallback object and runs the function with the
    on_batch_end callback event.
    """
    class NeMoCallbackWrapper(NeMoCallback):
        def __init__(self, my_func):
            self._func = my_func

        def on_batch_end(self, state):
            self._func(state)

    return NeMoCallbackWrapper(func)


def on_epoch_end(func):
    """A function decorator that wraps a Callable inside the NeMoCallback object and runs the function with the
    on_epoch_end callback event.
    """
    class NeMoCallbackWrapper(NeMoCallback):
        def __init__(self, my_func):
            self._func = my_func

        def on_epoch_end(self, state):
            self._func(state)

    return NeMoCallbackWrapper(func)


def on_train_end(func):
    """A function decorator that wraps a Callable inside the NeMoCallback object and runs the function with the
    on_train_end callback event.
    """
    class NeMoCallbackWrapper(NeMoCallback):
        def __init__(self, my_func):
            self._func = my_func

        def on_train_end(self, state):
            self._func(state)

    return NeMoCallbackWrapper(func)



class SimpleLogger(NeMoCallback):
    def __init__(self, step_freq:int = 100, tensors_to_log: List[Union[str, 'NmTensor']] = ["loss"]):
        """A simple callback that prints tensors to screen. It's default option is to print the training loss every
        100 steps. Additional tensors can be printed by adding them to the tensors_to_log argument.

        args:
            step_freq (int): The frequency of printing to screen. Defaults to every 100 steps
            tensors_to_log (List of str or NmTensor): A list of either tensor names or NmTensors which will be printed
                every step_freq steps.
                Defaults to ["loss"] which only prints the loss.
        """
        self.step_freq = step_freq
        self.tensors_to_log = tensors_to_log

    def on_step_end(self, state):
        if state["step"] % self.step_freq == 0:
            for tensor_key in self.tensors_to_log:
                tensor = state["tensors"].get_tensor(tensor_key)
                logging.info("%s: %s", tensor_key, tensor)


class TensorboardLogger(NeMoCallback):
    def __init__(
            self,
            tb_writer: 'torch.utils.tensorboard.SummaryWriter',
            step_freq:int=100,
            tensors_to_log:List[Union[str, 'NmTensor']]=["loss"],
            custom_tb_log_func:Callable[[Union[str, 'NmTensor']],None]=None,
            log_epoch:bool=True
        ):
        """A tensorboard callback that logs tensors using a tensorboard writer object. It's default option is to log
        the loss every 100 steps. Additional scalar tensors can be logged by adding them to the tensors_to_log
        argument. In order to log complex tensorboard entities, the custom_tb_log_func must be passed it. By default,
        it always logs the current epoch and the time taken per epoch.

        args:
            tb_writer (required): The tensorboard logger object.
            step_freq (int): The frequency of tensorboard logging. Defaults to every 100 steps
            tensors_to_log (List of str or NmTensor): A list of either tensor names or NmTensors which will be logged
                every step_freq steps.
                Defaults to ["loss"] which only prints the loss.
            custom_tb_log_func (func): TensorboardLogger loops through tensors_to_log and passes these elements to
                custom_tb_log_func. So a custom_tb_log_func will receive one argument on each call with the arugment
                being an element from tensors_to_log.
                Defaults to None which logs each tensors_to_log as a scalar.
            log_epoch (bool): Whether to log epoch and epoch training time to tensorboard.
                Defaults to True.
        """
        self.step_freq = step_freq
        self.tensors_to_log = tensors_to_log
        self.tb_writer = tb_writer
        self.custom_tb_log_func = custom_tb_log_func
        self._last_epoch_start = None
        self._log_epoch = log_epoch

    def on_epoch_start(self, state):
        if state["global_rank"] is None or state["global_rank"] == 0:
            self._last_epoch_start = time.time()

    def on_epoch_end(self, state):
        if state["global_rank"] is None or state["global_rank"] == 0:
            if self._log_epoch:
                epoch_time = time.time() - self._last_epoch_start
                self.tb_writer.add_scalar('misc/epoch', state["epoch"], state["step"])
                self.tb_writer.add_scalar('misc/epoch_time', epoch_time, state["step"])

    def on_step_end(self, state):
        if state["global_rank"] is None or state["global_rank"] == 0:
            if state["step"] % self.step_freq == 0:
                tb_log_func = lambda x: self.tb_writer.add_scalar(x, state["tensors"].get_tensor(x), state["step"])
                if self.custom_tb_log_func is not None:
                    tb_log_func = self.custom_tb_log_func
                for tensor_key in self.tensors_to_log:
                    tb_log_func(tensor_key)


class WandBLogger(NeMoCallback):
    def __init__(
            self,
            step_freq:int=100,
            tensors_to_log:List[Union[str, 'NmTensor']]=["loss"],
            wandb_name:str=None,
            wandb_project:str=None,
            args=None,
            log_epoch:bool=True
        ):
        """A [Weights & Biases](https://docs.wandb.com/) callback that logs tensors to W&B. It's default option is to
        log the loss every 100 steps. Additional scalar tensors can be logged by adding them to the tensors_to_log
        argument. By default, it always logs the current epoch and the time taken per epoch.

        args:
            step_freq (int): The frequency of tensorboard logging. Defaults to every 100 steps
            tensors_to_log (List of str or NmTensor): A list of either tensor names or NmTensors which will be logged
                every step_freq steps.
                Defaults to ["loss"] which only prints the loss.
            wandb_name(str): wandb experiment name.
                Defaults to None
            wandb_project(str): wandb project name.
                Defaults to None
            args: argparse flags which will be logged as hyperparameters.
                Defaults to None.
            log_epoch (bool): Whether to log epoch and epoch training time to tensorboard.
                Defaults to True.
        """
        if not _WANDB_AVAILABLE:
            logging.error("Could not import wandb. Did you install it (pip install --upgrade wandb)?")
        self._step_freq = step_freq
        self._tensors_to_log = tensors_to_log
        self._name = wandb_name
        self._project = wandb_project
        self._args = args
        self._last_epoch_start = None
        self._log_epoch = log_epoch

    def on_train_start(self, state):
        if state["global_rank"] is None or state["global_rank"] == 0:
            if _WANDB_AVAILABLE and wandb.run is None:
                wandb.init(name=self._name, project=self._project)
                if self._args is not None:
                    wandb.config.update(self._args)
            elif _WANDB_AVAILABLE and wandb.run is not None:
                logging.info("Re-using wandb session")
            else:
                logging.error("Could not import wandb. Did you install it (pip install --upgrade wandb)?")
                logging.info("Will not log data to weights and biases.")
                self._step_freq = -1

    def on_step_end(self, state):
        # log training metrics
        if state["global_rank"] is None or state["global_rank"] == 0:
            if state["step"] % self._step_freq == 0 and self._step_freq > 0:
                tensors_logged = {t: state["tensors"].get_tensor(t).cpu() for t in self._tensors_to_log}
                # Always log learning rate
                tensors_logged['LR'] = state["learning_rate"]
                self._wandb_log(tensors_logged)

    def on_epoch_start(self, state):
        if state["global_rank"] is None or state["global_rank"] == 0:
            self._last_epoch_start = time.time()

    def on_epoch_end(self, state):
        if state["global_rank"] is None or state["global_rank"] == 0:
            if self._log_epoch:
                epoch_time = time.time() - self._last_epoch_start
                self._wandb_log({"epoch": state["epoch"], "epoch_time": epoch_time})

    def _wandb_log(self, tensors_logged):
        if _WANDB_AVAILABLE:
            wandb.log(tensors_logged, step=state["step"])


class CheckpointCallback(NeMoCallback):
    def __init__(
        self,
        folder:str,
        load_from_folder:str=None,
        step_freq:int=-1,
        epoch_freq:int=-1,
        checkpoints_to_keep:int=4,
        force_load:bool=False,
    ):
        """A callback that does checkpointing of module weights and trainer (incl. optimizer) status.

        args:
            folder (str, required): A path where checkpoints are to be stored and loaded from if load_from_folder is
                None.
            load_from_folder (str): A path where checkpoints can be loaded from.
                Defaults to None.
            step_freq (int): How often in terms of steps to save checkpoints. One of step_freq or epoch_freq is
                required.
            epoch_freq (int): How often in terms of epochs to save checkpoints. One of step_freq or epoch_freq is
                required.
            checkpoints_to_keep (int): Number of most recent checkpoints to keep. Older checkpoints will be deleted.
                Defaults to 4.
            force_load (bool): Whether to crash if loading is unsuccessful.
                Defaults to False
        """
        if step_freq == -1 and epoch_freq == -1:
            logging.warning("No checkpoints will be saved because step_freq and epoch_freq are both -1.")

        if step_freq > -1 and epoch_freq > -1:
            logging.warning("You config the model to save by both steps and epochs. Please use one or the other")
            epoch_freq = -1

        self._step_freq = step_freq
        self._epoch_freq = epoch_freq
        self._folder = folder
        self._load_from_folder = load_from_folder if load_from_folder else folder
        self._ckpt2keep = checkpoints_to_keep
        self._saved_ckpts = []
        # If True, run will fail if we cannot load module weights
        self._force_load = force_load

    def __save_to(self, path, state):
        if state["global_rank"] is not None and state["global_rank"] != 0:
            return
        if not os.path.isdir(path):
            logging.info(f"Creating {path} folder")
            os.makedirs(path, exist_ok=True)
        unique_mod_names = set()
        for module in AppState().modules:
            if module.num_weights > 0:
                if str(module) in unique_mod_names:
                    raise NotImplementedError(
                        "There were two instances of the same module. Please overwrite __str__() of one of the "
                        "modules."
                    )
                unique_mod_names.add(str(module))
                if self._step_freq > -1:
                    filename = f"{module}-STEP-{state['step']}.pt"
                else:
                    filename = f"{module}-EPOCH-{state['epoch']}.pt"
                module.save_to(os.path.join(path, filename))

        if self._step_freq > -1:
            filename = f"trainer-STEP-{state['step']}.pt"
            state.save_state_to(f"{path}/{filename}")
            self._saved_ckpts.append(f"-{state['step']}.pt")
        else:
            filename = f"trainer-EPOCH-{state['epoch']}.pt"
            state.save_state_to(f"{path}/{filename}")
            self._saved_ckpts.append(f"-{state['epoch']}.pt")

        if len(self._saved_ckpts) > self._ckpt2keep:
            for end in self._saved_ckpts[: -self._ckpt2keep]:
                for file in glob.glob(f'{path}/*{end}'):
                    os.remove(file)
            self._saved_ckpts = self._saved_ckpts[-self._ckpt2keep :]
        logging.info(f'Saved checkpoint: {path}/{filename}')

    def __restore_from(self, path, state):
        if not os.path.isdir(path):
            if self._force_load:
                raise ValueError("force_load was set to True for checkpoint callback but a checkpoint was not found.")
            logging.warning(f"Checkpoint folder {path} not found!")
        else:
            logging.info(f"Found checkpoint folder {path}. Will attempt to restore checkpoints from it.")
            modules_to_restore = []
            modules_to_restore_name = []
            for module in AppState().modules:
                if module.num_weights > 0:
                    modules_to_restore.append(module)
                    modules_to_restore_name.append(str(module))
            try:
                module_checkpoints = get_checkpoint_from_dir(modules_to_restore_name, path)

                for mod, checkpoint in zip(modules_to_restore, module_checkpoints):
                    mod.restore_from(checkpoint, state["local_rank"])
            except (ValueError) as e:
                if self._force_load:
                    raise ValueError(
                        "force_load was set to True for checkpoint callback but a checkpoint was not found."
                    )
                logging.warning(e)
                logging.warning(
                    f"Checkpoint folder {path} was present but nothing was restored. Continuing training from random "
                    "initialization."
                )
                return

            try:
                trainer_checkpoints = get_checkpoint_from_dir(["trainer"], path)
                state.restore_state_from(trainer_checkpoints[0])
                # for tr, checkpoint in zip([self.action], trainer_checkpoints):
            except (ValueError) as e:
                logging.warning(e)
                logging.warning(
                    "Trainer state such as optimizer state and current step/epoch was not restored. Pretrained weights"
                    " have still been restore and fine-tuning should continue fine."
                )
                return

    def on_train_start(self, state):
        num_parameters = 0
        unique_mod_names = set()
        for module in AppState().modules:
            if module.num_weights > 0:
                if str(module) in unique_mod_names:
                    raise NotImplementedError(
                        "There were two instances of the same module. Please overwrite __str__() of one of the "
                        "modules."
                    )
                unique_mod_names.add(str(module))
                num_parameters += module.num_weights
        logging.info(f"Found {len(unique_mod_names)} modules with weights:")
        for name in unique_mod_names:
            logging.info(f"{name}")
        logging.info(f"Total model parameters: {num_parameters}")
        self.__restore_from(self._load_from_folder, state)

    def on_step_end(self, state):
        step = state["step"]
        if self._step_freq > 0 and step % self._step_freq == 0 and step > 0:
            self.__save_to(self._folder, state)

    def on_train_end(self, state):
        if self._step_freq > 0 or self._epoch_freq > 0:
            self.__save_to(self._folder, state)

    def on_epoch_end(self, state):
        epoch = state["epoch"]
        if self._epoch_freq > 0 and epoch % self._epoch_freq == 0 and epoch > 0:
            self.__save_to(self._folder, state)



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
        warnings.warn("This will be deprecated in future releases. Please use nemo.logging instead")
        return nemo.logging

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
                # name = f'param/{class_name}.{self.arg_name}'
                name = f"param/{class_name}.{self.arg_name}"
                self.tb_writer.add_scalar(name, value, self.step)
        else:
            self.cur_i += 1
            self.on_iteration_start()


class UnfreezeCallback(ActionCallback):
    def __init__(self, modules, start_epoch=0):
        super().__init__()

        self.modules = modules
        self.start_epoch = start_epoch

    def on_iteration_start(self):
        if self.epoch_num == self.start_epoch:
            for m in self.modules:
                m.unfreeze()


class OldWandbCallback(ActionCallback):
    """
    Log metrics to [Weights & Biases](https://docs.wandb.com/)
    """

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
