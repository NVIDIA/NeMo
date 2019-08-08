# Copyright (c) 2019 NVIDIA Corporation
from abc import ABC, abstractmethod
from collections import namedtuple
import glob
import math
import logging
import os
import sys
import time

from ..utils import get_checkpoint_from_dir


logger = logging.getLogger('log')


class ActionCallback(ABC):
    """Abstract interface for callbacks.
    """

    def __init__(self):
        self._step = None
        self._epoch_num = None
        self._registered_tensors = None
        self._tensors_to_optimize = None
        self._tensors_to_evaluate = None
        self._local_rank = None

    @property
    def step(self):
        return self._step

    @property
    def epoch_num(self):
        return self._epoch_num

    @property
    def registered_tensors(self):
        return self._registered_tensors

    @property
    def tensors_to_optimize(self):
        return self._tensors_to_optimize

    @property
    def tensors_to_evaluate(self):
        return self._tensors_to_evaluate

    @property
    def local_rank(self):
        return self._local_rank

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
    def __init__(self, save_modules_list, step_frequency=1000, folder=None):
        super().__init__()
        self._save_modules_list = save_modules_list
        self._folder = folder
        self._step_frequency = step_frequency

    def on_iteration_end(self):
        step = self.step
        if (
                self._step_frequency > 0
                and
                step % self._step_frequency == 0
                and step > 0
                and (self._local_rank is None or self._local_rank == 0)
        ):
            for m in self._save_modules_list:
                class_name = m.__class__.__name__
                uid = m.unique_instance_id
                fn = "{0}_{1}-STEP-{2}.pt".format(class_name, uid, step)
                if self._folder is None:
                    file_name = fn
                else:
                    file_name = os.path.join(self._folder, fn)
                print("Saving module {0} in {1}".format(class_name, file_name))
                m.save_to(file_name)
                print("Saved.")

    def on_action_end(self):
        step = self.step
        if self._local_rank is None or self._local_rank == 0:
            for m in self._save_modules_list:
                class_name = m.__class__.__name__
                uid = m.unique_instance_id
                fn = "{0}_{1}-STEP-{2}.pt".format(class_name, uid, step)
                if self._folder is None:
                    file_name = fn
                else:
                    file_name = os.path.join(self._folder, fn)
                print("Saving module {0} in {1}".format(class_name, file_name))
                m.save_to(file_name)
                print("Saved.")


class SimpleLossLoggerCallback(ActionCallback):
    def __init__(
            self,
            tensor_list2string,
            tensor_list2string_evl=None,
            step_frequency=25,
            tensorboard_writer=None,
    ):
        super().__init__()
        self._tensor_list2string = tensor_list2string
        self._tensor_list2string_evl = tensor_list2string_evl
        self._step_frequency = step_frequency
        self._swriter = tensorboard_writer
        self._start_time = None
        self._last_epoch_start = None
        self._last_iter_start = None

    def on_action_start(self):
        if self.local_rank is None or self.local_rank == 0:
            print("Starting .....")
            self._start_time = time.time()

    def on_action_end(self):
        if self.local_rank is None or self.local_rank == 0:
            print("Done in {0}".format(time.time() - self._start_time))

    def on_epoch_start(self):
        if self.local_rank is None or self.local_rank == 0:
            print("Starting epoch {0}".format(self.epoch_num))
            self._last_epoch_start = time.time()

    def on_epoch_end(self):
        if self.local_rank is None or self.local_rank == 0:
            step = self.step
            print(
                "Finished epoch {0} in {1}".format(
                    self.epoch_num, time.time() - self._last_epoch_start
                )
            )
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
            if step % self._step_frequency == 0:
                loss_values = [
                    self.registered_tensors[t.name + "~~~" + t.producer._uuid]
                    for t in self.tensors_to_optimize
                ]
                print("Step: {0}".format(step))
                print("Train Loss: {0}"
                      .format(self._tensor_list2string(loss_values)))
                sys.stdout.flush()
                if self._swriter is not None:
                    value = float(self._tensor_list2string(loss_values))
                    self._swriter.add_scalar('loss', value, step)
                    try:
                        value = math.exp(value)
                        self._swriter.add_scalar('perplexity', value, step)
                    except OverflowError:
                        pass
                    value = time.time() - self._last_iter_start
                    self._swriter.add_scalar('misc/step_time', value, step)

                if self.tensors_to_evaluate is not None \
                        and self._tensor_list2string_evl is not None:
                    e_tensors = [
                        self.registered_tensors[t.name +
                                                "~~~" + t.producer._uuid]
                        for t in self.tensors_to_evaluate
                    ]
                    w_t = self._tensor_list2string_evl(e_tensors)
                    if w_t is not None:
                        print("Watched tensors: {0}".format(w_t))
                print(
                    "Step time: {0} seconds".format(
                        time.time() - self._last_iter_start)
                )


class CheckpointCallback(ActionCallback):
    def __init__(self, folder, step_freq=-1, epoch_freq=-1,
                 checkpoints_to_keep=4):
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
        self._ckpt2keep = checkpoints_to_keep
        self._saved_ckpts = []

        # These will be set in action
        self.call_chain = None
        self.action = None

    def __save_to(self, path):
        if not os.path.isdir(path):
            print("Creating {0} folder".format(path))
            os.makedirs(path, exist_ok=True)
        for ind, (module, _) in enumerate(self.call_chain):
            if module.num_weights > 0:
                class_name = module.__class__.__name__
                if self._step_freq > -1:
                    filename = f"{class_name}_{ind}-STEP-{self.step}.pt"
                else:
                    filename = f"{class_name}_{ind}-EPOCH-{self.epoch_num}.pt"
                module.save_to(os.path.join(path, filename))

        if self._step_freq > -1:
            filename = f"trainer_{ind}-STEP-{self.step}.pt"
        else:
            filename = f"trainer_{ind}-EPOCH-{self.epoch_num}.pt"

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
            logger.warning(f"Checkpoint folder {path} not found!")
        else:
            logger.info("Restoring checkpoint from folder {path} ...")
            modules_to_restore = []
            modules_to_restore_name = []
            for ind, (module, _) in enumerate(self.call_chain):
                if module.num_weights > 0:
                    modules_to_restore.append(module)
                    modules_to_restore_name.append(
                        "{0}_{1}".format(module.__class__.__name__, ind)
                    )
            try:
                module_checkpoints = get_checkpoint_from_dir(
                    modules_to_restore_name, path, None
                )

                for mod, checkpoint in zip(modules_to_restore,
                                           module_checkpoints):
                    mod.restore_from(checkpoint, self._local_rank)
            except BaseException as e:
                print(e)
                print(
                    "Checkpoint folder {0} present but did not restore".format(
                        path))
                return

            trainer_chekpoints = get_checkpoint_from_dir(
                ["trainer"], path, None)
            for tr, checkpoint in zip([self.action], trainer_chekpoints):
                tr.restore_state_from(checkpoint)

    def on_action_start(self):
        self.__restore_from(path=self._folder)

    def on_iteration_end(self):
        step = self.step
        if (
                self._step_freq > 0
                and step % self._step_freq == 0
                and step > 0
                and (self._local_rank is None or self._local_rank == 0)
        ):
            self.__save_to(path=self._folder)

    def on_action_end(self):
        if self._local_rank is None or self._local_rank == 0:
            self.__save_to(path=self._folder)

    def on_epoch_start(self):
        self._last_epoch_start = time.time()

    def on_epoch_end(self):
        if self._epoch_freq > 0:
            if self.local_rank is None or self.local_rank == 0:
                logger.info(
                    "Finished epoch {0} in {1}".format(
                        self.epoch_num, time.time() - self._last_epoch_start
                    )
                )
                if (self.epoch_num + 1) % self._epoch_freq == 0:
                    self.__save_to(path=self._folder)


class EvaluatorCallback(ActionCallback):
    def __init__(
            self,
            eval_tensors,
            user_iter_callback,
            user_epochs_done_callback,
            tensorboard_writer=None,
            eval_step=None,
            eval_epoch=None,
    ):
        if eval_step is None and eval_epoch is None:
            raise ValueError(
                "Either eval_step or eval_epoch must be set. "
                "But got: {0} and {1}".format(eval_step, eval_epoch)
            )
        if (eval_step is not None and eval_step <= 0) or (
                eval_epoch is not None and eval_epoch <= 0
        ):
            raise ValueError(
                "eval_step and eval_epoch must be >0."
                "But got: {0} and {1}".format(eval_step, eval_epoch)
            )
        super().__init__()
        self._eval_tensors = eval_tensors
        self._swriter = tensorboard_writer
        self._eval_frequency = eval_step
        # will be passed to callbacks below
        self._global_var_dict = {}

        # Callbacks
        self._compute_callback = None
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
            if self._compute_callback is not None:
                if self._local_rank == 0 or self._local_rank is None:
                    print("Doing Evaluation .................................")
                start_time = time.time()
                self._compute_callback(self._eval_tensors, self, step)
                elapsed_time = time.time() - start_time
                if self._local_rank == 0 or self._local_rank is None:
                    print("Evaluation time: {0} seconds".format(elapsed_time))

    def on_action_end(self):
        step = self.step
        if self._compute_callback is not None:
            print(
                "Final Evaluation ....................... ......  ... .. . .")
            self._compute_callback(self._eval_tensors, self, step)

    def clear_global_var_dict(self):
        self._global_var_dict = {}


class InferenceCallback(ActionCallback):
    def __init__(
            self,
            eval_tensors,
    ):
        super().__init__()
        self._eval_tensors = eval_tensors
        # will be passed to callbacks below
        self._global_var_dict = {}
        self._swriter = None

    @property
    def eval_tensors(self):
        return self._eval_tensors

    def user_done_callback(self, global_var_dict):
        pass

    def user_iter_callback(self, tensors, global_var_dict):
        """ Pushes evaluated tensors to var_dict """
        for tensor in self._eval_tensors:
            key = tensor.unique_name
            self._global_var_dict[key] += tensors[key]

    def clear_global_var_dict(self):
        for tensor in self._eval_tensors:
            self._global_var_dict[tensor.unique_name] = []


Policy = namedtuple('Policy', 'method start end')


class _Method(ABC):
    @abstractmethod
    def __call__(self, step, total_steps):
        pass


class Const(_Method):
    def __init__(self, value):
        super().__init__()

        self.value = value

    def __call__(self, step, total_steps):
        return self.value


class Linear(_Method):
    def __init__(self, a, b):
        super().__init__()

        self.a = a
        self.b = b

    def __call__(self, step, total_steps):
        return self.a + (step / (total_steps - 1)) * (self.b - self.a)


class ValueSetterCallback(ActionCallback):
    def __init__(self, module, arg_name,
                 policies=None, total_steps=None, tb_writer=None):
        super().__init__()

        if policies is None:
            initial_value = getattr(module, arg_name)
            policies = [Policy(method=Const(initial_value),
                               start=0.0, end=1.0)]

        new_policies = []
        for p in policies:
            start, end = p.start, p.end
            if isinstance(start, float):
                start = int(start * total_steps)
            if isinstance(end, float):
                end = int(end * total_steps)
            new_policies.append(Policy(p.method, start, end))
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
