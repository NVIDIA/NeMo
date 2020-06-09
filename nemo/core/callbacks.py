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

# __all__ = [
#     "NeMoCallback",
#     "SimpleLogger",
#     "TensorboardLogger",
#     "WandBLogger",
#     "CheckpointCallback",
#     "on_action_start",
#     "on_action_end",
#     "on_epoch_start",
#     "on_epoch_end",
#     "on_batch_start",
#     "on_batch_end",
#     "on_step_start",
#     "on_step_end",
# ]

import glob
import os
import time
from abc import ABC
from typing import Callable, List, Union

from nemo.core.deprecated_callbacks import (
    ActionCallback,
    EvaluatorCallback,
    ModuleSaverCallback,
    SimpleLossLoggerCallback,
    UnfreezeCallback,
    ValueSetterCallback,
    WandbCallback,
)
from nemo.core.neural_types import NmTensor
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

    def on_action_start(self, state):
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

    def on_action_end(self, state):
        pass


def on_action_start(func):
    """A function decorator that wraps a Callable inside the NeMoCallback object and runs the function with the
    on_action_start callback event.
    """

    class NeMoCallbackWrapper(NeMoCallback):
        def __init__(self, my_func):
            self._func = my_func

        def on_action_start(self, state):
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


def on_action_end(func):
    """A function decorator that wraps a Callable inside the NeMoCallback object and runs the function with the
    on_action_end callback event.
    """

    class NeMoCallbackWrapper(NeMoCallback):
        def __init__(self, my_func):
            self._func = my_func

        def on_action_end(self, state):
            self._func(state)

    return NeMoCallbackWrapper(func)


class SimpleLogger(NeMoCallback):
    """A simple callback that prints tensors to screen. It's default option is to print the training loss every
    100 steps. Additional tensors can be printed by adding them to the tensors_to_log argument.

    args:
        step_freq (int): The frequency of printing to screen. Defaults to every 100 steps
        tensors_to_log (List of str or NmTensor): A list of either tensor names or NmTensors which will be printed
            every step_freq steps.
            Defaults to ["loss"] which only prints the loss.
    """

    def __init__(self, step_freq: int = 100, tensors_to_log: List[Union[str, NmTensor]] = ["loss"]):
        self.step_freq = step_freq
        self.tensors_to_log = tensors_to_log

    def on_step_end(self, state):
        if state["step"] % self.step_freq == 0:
            for tensor_key in self.tensors_to_log:
                tensor = state["tensors"].get_tensor(tensor_key)
                logging.info("%s: %s", tensor_key, tensor.data.cpu().numpy())


class TensorboardLogger(NeMoCallback):
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
        custom_tb_log_func (func): custom_tb_log_func should accept three position arguments: tensors, tb_writer, and
            step. tensors is a list of pytorch tensors that correspond to the values of the NmTensors in
            tensors_to_log. tb_writer is the tensorboard logger passed to TensorboardLogger. step is the current
            step.
            Defaults to None which logs each tensors_to_log as a scalar.
        log_epoch (bool): Whether to log epoch and epoch training time to tensorboard.
            Defaults to True.
        log_lr (bool): Whether to log the learning rate to tensorboard.
            Defaults to True.
    """

    def __init__(
        self,
        tb_writer: 'torch.utils.tensorboard.SummaryWriter',
        step_freq: int = 100,
        tensors_to_log: List[Union[str, NmTensor]] = ["loss"],
        custom_tb_log_func: Callable[[Union[str, NmTensor]], None] = None,
        log_epoch: bool = True,
        log_lr: bool = True,
    ):
        self.step_freq = step_freq
        self.tensors_to_log = tensors_to_log
        self.tb_writer = tb_writer
        self.custom_tb_log_func = custom_tb_log_func
        self._last_epoch_start = None
        self._log_epoch = log_epoch
        self._log_lr = log_lr

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
                tensor_values = []
                for tensor_key in self.tensors_to_log:
                    tensor_values.append(state["tensors"].get_tensor(tensor_key))

                def tb_log_func(tensors, tb_writer, step):
                    for t, name in zip(tensors, self.tensors_to_log):
                        tb_writer.add_scalar(name, t, step)

                if self.custom_tb_log_func is not None:
                    tb_log_func = self.custom_tb_log_func
                tb_log_func(tensor_values, self.tb_writer, state["step"])

                if self._log_lr:
                    self.tb_writer.add_scalar('param/lr', state["optimizers"][0].param_groups[0]['lr'], state["step"])


class WandBLogger(NeMoCallback):
    """A [Weights & Biases](https://docs.wandb.com/) callback that logs tensors to W&B. It's default option is to
    log the loss every 100 steps. Additional scalar tensors can be logged by adding them to the tensors_to_log
    argument. By default, it always logs the current epoch and the time taken per epoch.

    args:
        step_freq (int): The frequency of Weights and Biases logging. Defaults to every 100 steps
        tensors_to_log (List of str or NmTensor): A list of either tensor names or NmTensors which will be logged
            every step_freq steps.
            Defaults to ["loss"] which only prints the loss.
        wandb_name(str): wandb experiment name.
            Defaults to None
        wandb_project(str): wandb project name.
            Defaults to None
        args: argparse flags which will be logged as hyperparameters.
            Defaults to None.
        log_epoch (bool): Whether to log epoch and epoch training time to Weights and Biases.
            Defaults to True.
        log_lr (bool): Whether to log epoch and epoch training time to Weights and Biases.
            Defaults to True.
    """

    def __init__(
        self,
        step_freq: int = 100,
        tensors_to_log: List[Union[str, NmTensor]] = ["loss"],
        wandb_name: str = None,
        wandb_project: str = None,
        args=None,
        log_epoch: bool = True,
        log_lr: bool = True,
    ):
        if not _WANDB_AVAILABLE:
            logging.error("Could not import wandb. Did you install it (pip install --upgrade wandb)?")
        self._step_freq = step_freq
        self._tensors_to_log = tensors_to_log
        self._name = wandb_name
        self._project = wandb_project
        self._args = args
        self._last_epoch_start = None
        self._log_epoch = log_epoch
        self._log_lr = log_lr

    def on_action_start(self, state):
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
                if self._log_lr:
                    tensors_logged['LR'] = state["optimizers"][0].param_groups[0]['lr']
                self._wandb_log(tensors_logged, state["step"])

    def on_epoch_start(self, state):
        if state["global_rank"] is None or state["global_rank"] == 0:
            self._last_epoch_start = time.time()

    def on_epoch_end(self, state):
        if state["global_rank"] is None or state["global_rank"] == 0:
            if self._log_epoch:
                epoch_time = time.time() - self._last_epoch_start
                self._wandb_log({"epoch": state["epoch"], "epoch_time": epoch_time}, state["step"])

    @staticmethod
    def _wandb_log(tensors_logged, step):
        if _WANDB_AVAILABLE:
            wandb.log(tensors_logged, step=step)


class CheckpointCallback(NeMoCallback):
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

    def __init__(
        self,
        folder: str,
        load_from_folder: str = None,
        step_freq: int = -1,
        epoch_freq: int = -1,
        checkpoints_to_keep: int = 4,
        force_load: bool = False,
    ):
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
            step_check = None
            try:
                module_checkpoints, steps = get_checkpoint_from_dir(modules_to_restore_name, path, return_steps=True)

                # If the steps are different, print a warning message
                for step in steps:
                    if step_check is None:
                        step_check = step
                    elif step != step_check:
                        logging.warning("Restoring from modules checkpoints where the training step does not match")
                        break

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
                trainer_checkpoints, steps = get_checkpoint_from_dir(["trainer"], path, return_steps=True)
                if step_check is not None and step_check != steps[0]:
                    logging.error(
                        "The step we are restoring from the trainer checkpoint does not match one or more steps that "
                        "are being restored from modules."
                    )
                state.restore_state_from(trainer_checkpoints[0])
            except (ValueError) as e:
                logging.warning(e)
                logging.warning(
                    "Trainer state such as optimizer state and current step/epoch was not restored. Pretrained weights"
                    " have still been restore and fine-tuning should continue fine."
                )
                return

    def on_action_start(self, state):
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

    def on_action_end(self, state):
        if self._step_freq > 0 or self._epoch_freq > 0:
            self.__save_to(self._folder, state)

    def on_epoch_end(self, state):
        epoch = state["epoch"]
        if self._epoch_freq > 0 and epoch % self._epoch_freq == 0 and epoch > 0:
            self.__save_to(self._folder, state)
