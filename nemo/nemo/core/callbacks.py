# Copyright (c) 2019 NVIDIA Corporation
from abc import ABC, abstractmethod
from collections import namedtuple
import glob
import math
import os
import sys
import time

from ..utils.exp_logging import get_logger
from ..utils import get_checkpoint_from_dir


logger = get_logger('')


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
    def action(self):
        return self._action

    @action.setter
    def action(self, action_obj):
        self._action = action_obj

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
    When step_freq = -1, don't save anything.
    """

    def __init__(self, save_modules_list, step_freq=1000, folder=None):
        super().__init__()
        self._save_modules_list = save_modules_list
        self._folder = folder
        self._step_freq = step_freq

    def on_iteration_end(self):
        step = self.step
        if (
                self._step_freq > 0
                and
                step % self._step_freq == 0
                and step > 0
                and (self.local_rank is None or self.local_rank == 0)
        ):
            for m in self._save_modules_list:
                class_name = m.__class__.__name__
                uid = m.unique_instance_id
                fn = "{0}_{1}-STEP-{2}.pt".format(class_name, uid, step)
                if self._folder is None:
                    file_name = fn
                else:
                    file_name = os.path.join(self._folder, fn)
                print(f"Saving module {class_name} in {file_name}")
                m.save_to(file_name)
                print("Saved.")

    def on_action_end(self):
        step = self.step
        if self.local_rank is None or self.local_rank == 0:
            for m in self._save_modules_list:
                class_name = m.__class__.__name__
                uid = m.unique_instance_id
                fn = "{0}_{1}-STEP-{2}.pt".format(class_name, uid, step)
                if self._folder is None:
                    file_name = fn
                else:
                    file_name = os.path.join(self._folder, fn)
                print(f"Saving module {class_name} in {file_name}")
                m.save_to(file_name)
                print("Saved.")


class SimpleLossLoggerCallback(ActionCallback):

    def __init__(self,
                 tensors,
                 print_func=None,
                 get_tb_values=None,
                 step_freq=25,
                 tb_writer=None):

        super().__init__()
        if not isinstance(tensors, list):
            tensors = [tensors]
        self._tensors = tensors
        self._print_func = print_func
        self._get_tb_values = get_tb_values
        self._step_freq = step_freq
        self._swriter = tb_writer
        self._start_time = None
        self._last_epoch_start = None
        self._last_iter_start = None

    @property
    def tensors(self):
        return self._tensors

    def on_action_start(self):
        if self.local_rank is None or self.local_rank == 0:
            print("Starting .....")
            self._start_time = time.time()

    def on_action_end(self):
        if self.local_rank is None or self.local_rank == 0:
            if self._swriter is not None:
                self._swriter.close()
            print(f"Done in {time.time() - self._start_time}")

    def on_epoch_start(self):
        if self.local_rank is None or self.local_rank == 0:
            print(f"Starting epoch {self.epoch_num}")
            self._last_epoch_start = time.time()

    def on_epoch_end(self):
        if self.local_rank is None or self.local_rank == 0:
            step = self.step
            run_time = time.time() - self._last_epoch_start
            print(f"Finished epoch {self.epoch_num} in {run_time}")
            if self._swriter is not None:
                value = self.epoch_num
                self._swriter.add_scalar('misc/epoch', value, step)
                value = time.time() - self._last_epoch_start
                self._swriter.add_scalar('misc/epoch_time', value, step)

    def on_iteration_start(self):
        if self.local_rank is None or self.local_rank == 0:
            self._last_iter_start = time.time()

    def on_iteration_end(self):
        if self.local_rank is None or self.local_rank == 0:
            step = self.step
            if step % self._step_freq == 0:
                tensor_values = [
                    self.registered_tensors[t.unique_name]
                    for t in self.tensors
                ]

                print(f"Step: {step}")
                if self._print_func:
                    self._print_func(tensor_values)
                sys.stdout.flush()
                if self._swriter is not None:
                    if self._get_tb_values:
                        tb_objects = self._get_tb_values(tensor_values)
                        for name, value in tb_objects:
                            value = value.item()
                            self._swriter.add_scalar(name, value, step)
                    run_time = time.time() - self._last_iter_start
                    self._swriter.add_scalar('misc/step_time', run_time, step)
                run_time = time.time() - self._last_iter_start
                print(f"Step time: {run_time} seconds")


class CheckpointCallback(ActionCallback):

    def __init__(self, folder, load_from_folder=None, step_freq=-1,
                 epoch_freq=-1, checkpoints_to_keep=4, force_load=False):
        super().__init__()
        if step_freq == -1 and epoch_freq == -1:
            logger.warning(
                "No checkpoints will be saved because step_freq and "
                "epoch_freq are both -1."
            )

        if step_freq > -1 and epoch_freq > -1:
            logger.warning(
                "You config the model to save by both steps and epochs. "
                "Save by step_freq only"
            )
            epoch_freq = -1

        self._step_freq = step_freq
        self._epoch_freq = epoch_freq
        self._folder = folder
        self._load_from_folder = load_from_folder if load_from_folder \
            else folder
        self._ckpt2keep = checkpoints_to_keep
        self._saved_ckpts = []
        # If True, run will fail if we cannot load module weights
        self._force_load = force_load

    def __save_to(self, path):
        if not os.path.isdir(path):
            print("Creating {0} folder".format(path))
            os.makedirs(path, exist_ok=True)
        unique_mod_names = set()
        for module in self.action.modules:
            if module.num_weights > 0:
                if str(module) in unique_mod_names:
                    raise NotImplementedError(
                        "There were two instances of the same module. Please "
                        "overwrite __str__() of one of the modules.")
                unique_mod_names.add(str(module))
                if self._step_freq > -1:
                    filename = f"{module}-STEP-{self.step}.pt"
                else:
                    filename = f"{module}-EPOCH-{self.epoch_num}.pt"
                module.save_to(os.path.join(path, filename))

        if self._step_freq > -1:
            filename = f"trainer-STEP-{self.step}.pt"
            self.action.save_state_to(f'{path}/{filename}')
            self._saved_ckpts.append(f'-{self.step}.pt')
        else:
            filename = f"trainer-EPOCH-{self.epoch_num}.pt"
            self.action.save_state_to(f'{path}/{filename}')
            self._saved_ckpts.append(f'-{self.epoch_num}.pt')

        if len(self._saved_ckpts) > self._ckpt2keep:
            for end in self._saved_ckpts[:-self._ckpt2keep]:
                for file in glob.glob(f'{path}/*{end}'):
                    os.remove(file)
            self._saved_ckpts = self._saved_ckpts[-self._ckpt2keep:]
        logger.info(f'Saved checkpoint: {path}/{filename}')

    def __restore_from(self, path):
        if not os.path.isdir(path):
            if self._force_load:
                raise ValueError("force_load was set to True for checkpoint "
                                 "callback but a checkpoint was not found.")
            logger.warning(f"Checkpoint folder {path} not found!")
        else:
            logger.info(f"Restoring checkpoint from folder {path} ...")
            modules_to_restore = []
            modules_to_restore_name = []
            for module in self.action.modules:
                if module.num_weights > 0:
                    modules_to_restore.append(module)
                    modules_to_restore_name.append(str(module))
            try:
                module_checkpoints = get_checkpoint_from_dir(
                    modules_to_restore_name, path
                )

                for mod, checkpoint in zip(modules_to_restore,
                                           module_checkpoints):
                    mod.restore_from(checkpoint, self.local_rank)
            except (BaseException, ValueError) as e:
                if self._force_load:
                    raise ValueError(
                        "force_load was set to True for checkpoint callback"
                        "but a checkpoint was not found.")
                print(e)
                print(
                    "Checkpoint folder {0} present but did not restore".format(
                        path))
                return

            try:
                trainer_checkpoints = get_checkpoint_from_dir(
                    ["trainer"], path)
                for tr, checkpoint in zip([self.action], trainer_checkpoints):
                    tr.restore_state_from(checkpoint)
            except (BaseException, ValueError) as e:
                print(e)
                print("Trainer state wasn't restored".format(
                    path))
                return

    def on_action_start(self):
        self.__restore_from(path=self._load_from_folder)

    def on_iteration_end(self):
        step = self.step
        if (
                self._step_freq > 0
                and step % self._step_freq == 0
                and step > 0
                and (self.local_rank is None or self.local_rank == 0)
        ):
            self.__save_to(path=self._folder)

    def on_action_end(self):
        if self._step_freq > 0 or self._epoch_freq > 0:
            if self.local_rank is None or self.local_rank == 0:
                self.__save_to(path=self._folder)

    def on_epoch_start(self):
        self._last_epoch_start = time.time()

    def on_epoch_end(self):
        if self._epoch_freq > 0:
            if self.local_rank is None or self.local_rank == 0:
                run_time = time.time() - self._last_epoch_start
                logger.info(f'Finished epoch {self.epoch_num} in {run_time}')
                if (self.epoch_num + 1) % self._epoch_freq == 0:
                    self.__save_to(path=self._folder)


class EvaluatorCallback(ActionCallback):

    def __init__(
            self,
            eval_tensors,
            user_iter_callback,
            user_epochs_done_callback,
            tb_writer=None,
            eval_step=1,
            eval_epoch=None,
    ):
        # TODO: Eval_epoch currently does nothing
        if eval_step is None and eval_epoch is None:
            raise ValueError(
                "Either eval_step or eval_epoch must be set. "
                "But got: {0} and {1}".format(eval_step, eval_epoch)
            )
        if (eval_step is not None and eval_step <= 0) or (
                eval_epoch is not None and eval_epoch <= 0
        ):
            raise ValueError(
                "Eval_step and eval_epoch must be > 0."
                "But got: {0} and {1}".format(eval_step, eval_epoch)
            )
        super().__init__()
        self._eval_tensors = eval_tensors
        self._swriter = tb_writer
        self._eval_frequency = eval_step
        # will be passed to callbacks below
        self._global_var_dict = {}

        # Callbacks
        self.user_iter_callback = user_iter_callback
        self.user_done_callback = user_epochs_done_callback

    @property
    def eval_tensors(self):
        return self._eval_tensors

    def on_epoch_end(self):
        pass

    def on_iteration_end(self):
        step = self.step
        if step % self._eval_frequency == 0:
            if self.local_rank == 0 or self.local_rank is None:
                print('Doing Evaluation ' + '.' * 30)
            start_time = time.time()
            self.action._eval(self._eval_tensors, self, step)
            elapsed_time = time.time() - start_time
            if self.local_rank == 0 or self.local_rank is None:
                print(f'Evaluation time: {elapsed_time} seconds')

    def on_action_end(self):
        step = self.step
        if self.local_rank == 0 or self.local_rank is None:
            print('Final Evaluation ' + '.' * 30)
        start_time = time.time()
        self.action._eval(self._eval_tensors, self, step)
        elapsed_time = time.time() - start_time
        if self.local_rank == 0 or self.local_rank is None:
            print(f'Evaluation time: {elapsed_time} seconds')

    def clear_global_var_dict(self):
        self._global_var_dict = {}


# class InferenceCallback(ActionCallback):
#     def __init__(
#             self,
#             eval_tensors,
#     ):
#         super().__init__()
#         self._eval_tensors = eval_tensors
#         # will be passed to callbacks below
#         self._global_var_dict = {}
#         self._swriter = None

#     @property
#     def eval_tensors(self):
#         return self._eval_tensors

#     def user_done_callback(self, global_var_dict):
#         pass

#     def user_iter_callback(self, tensors, global_var_dict):
#         """ Pushes evaluated tensors to var_dict """
#         for tensor in self._eval_tensors:
#             key = tensor.unique_name
#             self._global_var_dict[key] += tensors[key]

#     def clear_global_var_dict(self):
#         for tensor in self._eval_tensors:
#             self._global_var_dict[tensor.unique_name] = []


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

    def __init__(self, module, arg_name,
                 policies=None, total_steps=None, tb_writer=None):
        super().__init__()

        if policies is None:
            initial_value = getattr(module, arg_name)
            policies = [_Policy(method=Const(initial_value),
                                start=0.0, end=1.0)]

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
                name = "param/{0}.{1}".format(class_name, self.arg_name)
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
