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

__all__ = [
    'OperationMode',
    'Optimization',
    'DeviceType',
    'NeuralModuleFactory',
    'DeploymentFormat',
]

import random
from enum import Enum
from typing import List, Optional, Union

import numpy as np

import nemo
from nemo.core.callbacks import ActionCallback, EvaluatorCallback, NeMoCallback
from nemo.core.neural_types import NmTensor
from nemo.utils import ExpManager, logging


class DeploymentFormat(Enum):
    """Which format to use when exporting a Neural Module for deployment"""

    AUTO = 0
    PYTORCH = 1
    TORCHSCRIPT = 2
    ONNX = 3
    TRTONNX = 4
    JARVIS = 5


class OperationMode(Enum):
    """Training or Inference (Evaluation) mode"""

    training = 0
    evaluation = 1
    both = 2


class Optimization(Enum):
    """Various levels of Apex/amp Optimization.
    WARNING: This might have effect on model accuracy."""

    mxprO0 = 0
    mxprO1 = 1
    mxprO2 = 2
    mxprO3 = 3


class DeviceType(Enum):
    """Device types where Neural Modules can be placed."""

    GPU = 1
    CPU = 2
    AllGpu = 3


def _str_to_opt_level(opt_str: str) -> Optimization:
    number = int(opt_str[1:])
    if number not in Optimization._value2member_map_:
        raise ValueError(f"Unknown optimization value {opt_str}")
    return Optimization(number)


class NeuralModuleFactory(object):
    _DEFAULT = None

    """
    Neural Module Factory instance is used to create neural modules and
    trainers

    Args:
        local_rank (int): Process rank. Should be set by distributed runner
        optimization_level (Optimization): Level of optimization to use. Will
            be passed to neural modules and actions created by this factory.
        placement (DeviceType: where to place NeuralModule instances by default
        cudnn_benchmark (bool): (default False) If set to True it will use
            cudnnFind method to find the best kernels instead of using
            heuristics. If the shapes of your inputs are constant this
            should help, for various shapes it can slow things down. Give it
            few iterations to warmup if set to True. Currently only supported
            by PyTorch backend.
        random_seed (int): (default None) Sets random seed to control for
            randomness. This should be used for debugging purposes as it might
            have negative impact on performance. Can't be used when
            `cudnn_benchmark=True`.
        master_process (bool): (default True) Flag for master process
            indication
        set_default (bool): (default True) True if should set this instance as
            default factory for modules instantiating.
    """

    def __init__(
        self,
        local_rank=None,
        optimization_level=Optimization.mxprO0,
        placement=None,
        cudnn_benchmark=False,
        random_seed=None,
        set_default=True,
        log_dir=None,
        checkpoint_dir=None,
        tensorboard_dir=None,
        create_tb_writer=False,
        files_to_copy=None,
        add_time_to_log_dir=False,
    ):
        self._local_rank = local_rank
        self._global_rank = None

        if isinstance(optimization_level, str):
            optimization_level = _str_to_opt_level(optimization_level)
        self._optim_level = optimization_level

        if placement is None:
            if local_rank is not None:
                device = DeviceType.AllGpu
            else:
                device = DeviceType.GPU

            self._placement = device
        else:
            self._placement = placement

        self._world_size = 1
        broadcast_func = None
        if True:  # backend == Backend.PyTorch:
            # TODO: Move all framework specific code from this file
            import torch

            if self._placement != DeviceType.CPU:
                if not torch.cuda.is_available():
                    raise ValueError(
                        "You requested to use GPUs but CUDA is "
                        "not installed. You can try running using"
                        " CPU-only. To do this, instantiate your"
                        " factory with placement=DeviceType.CPU"
                        "\n"
                        "Note that this is slow and is not "
                        "well supported."
                    )

            torch.backends.cudnn.benchmark = cudnn_benchmark
            if random_seed is not None and cudnn_benchmark:
                raise ValueError("cudnn_benchmark can not be set to True when random_seed is not None.")
            if random_seed is not None:
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
                torch.manual_seed(random_seed)
                np.random.seed(random_seed)
                random.seed(random_seed)

            # logging.info("Random seeds")
            # logging.info("torch: %d", torch.initial_seed())
            # logging.info("numpy: %d", )
            # logging.info("random: %d", )

            if self._local_rank is not None:
                torch.distributed.init_process_group(backend="nccl", init_method="env://")

                cuda_set = True
                # Try to set cuda device. This should fail if self._local_rank
                # is greater than the number of available GPUs
                try:
                    torch.cuda.set_device(self._local_rank)
                except RuntimeError:
                    # Note in this case, all tensors are now sent to GPU 0
                    # who could crash because of OOM. Thus init_process_group()
                    # must be done before any cuda tensors are allocated
                    cuda_set = False
                cuda_set_t = torch.cuda.IntTensor([cuda_set])

                # Do an all_reduce to ensure all workers obtained a GPU
                # For the strangest reason, BAND doesn't work so I am resorting
                # to MIN.
                torch.distributed.all_reduce(cuda_set_t, op=torch.distributed.ReduceOp.MIN)
                if cuda_set_t.item() == 0:
                    raise RuntimeError(
                        "There was an error initializing distributed training."
                        " Perhaps you specified more gpus than you have "
                        "available"
                    )

                del cuda_set_t
                torch.cuda.empty_cache()
                # Remove test tensor from memory

                self._world_size = torch.distributed.get_world_size()
                self._global_rank = torch.distributed.get_rank()

                def torch_broadcast_wrapper(str_len=None, string=None, src=0):
                    """Wrapper function to broadcast string values across all
                    workers
                    """
                    # Create byte cuda torch tensor
                    if string is not None:
                        string_tensor = torch.tensor(list(string.encode()), dtype=torch.uint8).cuda()
                    else:
                        string_tensor = torch.tensor([0] * str_len, dtype=torch.uint8).cuda()
                    # Run broadcast
                    torch.distributed.broadcast(string_tensor, src)
                    # turn byte tensor back to string
                    return_string = string_tensor.cpu().numpy()
                    return_string = b''.join(return_string).decode()
                    return return_string

                broadcast_func = torch_broadcast_wrapper
        else:
            raise NotImplementedError("Only Pytorch backend is currently supported.")

        # Create ExpManager
        # if log_dir is None, only create logger
        self._exp_manager = ExpManager(
            work_dir=log_dir,
            ckpt_dir=checkpoint_dir,
            use_tb=create_tb_writer,
            tb_dir=tensorboard_dir,
            local_rank=local_rank,
            global_rank=self._global_rank,
            files_to_copy=files_to_copy,
            add_time=add_time_to_log_dir,
            exist_ok=True,
            broadcast_func=broadcast_func,
        )
        self._tb_writer = self._exp_manager.tb_writer

        # Create trainer
        self._trainer = self._get_trainer(tb_writer=self._tb_writer)

        if set_default:
            NeuralModuleFactory.set_default_factory(self)

    @classmethod
    def get_default_factory(cls):
        return cls._DEFAULT

    @classmethod
    def set_default_factory(cls, factory):
        cls._DEFAULT = factory

    @classmethod
    def reset_default_factory(cls):
        cls._DEFAULT = None

    @staticmethod
    def __name_import(name):
        components = name.split(".")
        mod = __import__(components[0])
        for comp in components[1:]:
            mod = getattr(mod, comp)
        return mod

    def create_optimizer(self, optimizer, things_to_optimize, optimizer_params):
        return self._trainer.create_optimizer(
            optimizer=optimizer, things_to_optimize=things_to_optimize, optimizer_params=optimizer_params,
        )

    def train(
        self,
        tensors_to_optimize=None,
        training_graph=None,
        optimizer=None,
        optimization_params=None,
        callbacks: Optional[List[Union[ActionCallback, NeMoCallback]]] = None,
        lr_policy=None,
        batches_per_step=None,
        stop_on_nan_loss=False,
        synced_batchnorm=False,
        synced_batchnorm_groupsize=0,
        gradient_predivide=False,
        amp_max_loss_scale=2.0 ** 24,
        reset=False,
    ):
        if reset:
            self.reset_trainer()
        return self._trainer.train(
            tensors_to_optimize=tensors_to_optimize,
            training_graph=training_graph,
            optimizer=optimizer,
            optimization_params=optimization_params,
            callbacks=callbacks,
            lr_policy=lr_policy,
            batches_per_step=batches_per_step,
            stop_on_nan_loss=stop_on_nan_loss,
            synced_batchnorm=synced_batchnorm,
            synced_batchnorm_groupsize=synced_batchnorm_groupsize,
            gradient_predivide=gradient_predivide,
            amp_max_loss_scale=amp_max_loss_scale,
        )

    def eval(self, callbacks: List[EvaluatorCallback]):
        if callbacks is None or len(callbacks) == 0:
            raise ValueError(f"You need to provide at lease one evaluation" f"callback to eval")
        for callback in callbacks:
            if not isinstance(callback, EvaluatorCallback):
                raise TypeError(f"All callbacks passed to the eval action must" f"be inherited from EvaluatorCallback")
        self.train(
            tensors_to_optimize=None, optimizer='sgd', callbacks=callbacks, optimization_params={'num_epochs': 1},
        )

    def deployment_export(
        self, module, output: str, d_format: DeploymentFormat, input_example=None, output_example=None
    ):
        """Exports Neural Module instance for deployment.

        Args:
            module: neural module to export
            output (str): where export results should be saved
            d_format (DeploymentFormat): which deployment format to use
            input_example: sometimes tracing will require input examples
            output_example: Should match inference on input_example
        """
        if d_format == DeploymentFormat.JARVIS:
            logging.info("Exporting model to Jarvis.")
            module.deploy_to_jarvis(output=output)
            logging.info(f"Exported to {output}")
            return

        _inexample, _out_example = module._prepare_for_deployment()

        if input_example is not None:
            _inexample = input_example
        if output_example is not None:
            _out_example = output_example

        return self._trainer.deployment_export(
            module=module, output=output, d_format=d_format, input_example=_inexample, output_example=_out_example,
        )

    def infer(
        self,
        tensors: List[NmTensor],
        checkpoint_dir=None,
        ckpt_pattern='',
        verbose=True,
        cache=False,
        use_cache=False,
        offload_to_cpu=True,
        modules_to_restore=None,
    ):
        """Runs inference to obtain values for tensors

        Args:
            tensors (list[NmTensor]): List of NeMo tensors that we want to get
                values of.
            checkpoint_dir (str): Path to checkpoint directory. Default is None
                which does not load checkpoints.
            ckpt_pattern (str): Pattern used to check for checkpoints inside
                checkpoint_dir. Default is '' which matches any checkpoints
                inside checkpoint_dir.
            verbose (bool): Controls printing. Defaults to True.
            cache (bool): If True, cache all `tensors` and intermediate tensors
                so that future calls that have use_cache set will avoid
                computation. Defaults to False.
            use_cache (bool): Values from `tensors` will be always re-computed.
                It will re-use intermediate tensors from the DAG leading to
                `tensors`. If you want something to be re-computed, put it into
                `tensors` list. Defaults to False.
            offload_to_cpu (bool): If True, all evaluated tensors are moved to
                cpu memory after each inference batch. Defaults to True.
            modules_to_restore (list): Defaults to None, in which case all
                NMs inside callchain with weights will be restored. If
                specified only the modules inside this list will be restored.

        Returns:
            List of evaluated tensors. Each element in the list is also a list
            where each element is now a batch of tensor values.
        """
        return self._trainer.infer(
            tensors=tensors,
            checkpoint_dir=checkpoint_dir,
            ckpt_pattern=ckpt_pattern,
            verbose=verbose,
            cache=cache,
            use_cache=use_cache,
            offload_to_cpu=offload_to_cpu,
            modules_to_restore=modules_to_restore,
        )

    def clear_cache(self):
        """Helper function to clean inference cache."""
        self._trainer.clear_cache()

    def _get_trainer(self, tb_writer=None):
        if True:  # self._backend == Backend.PyTorch:
            constructor = NeuralModuleFactory.__name_import("nemo.backends.pytorch.PtActions")
            instance = constructor(
                local_rank=self._local_rank,
                global_rank=self._global_rank,
                tb_writer=tb_writer,
                optimization_level=self._optim_level,
            )
            return instance
        else:
            raise ValueError("Only PyTorch backend is currently supported.")

    def reset_trainer(self):
        del self._trainer
        self._trainer = self._get_trainer(tb_writer=self._tb_writer)

    def sync_all_processes(self, status=True):
        """ Helper function for testing that allows proccess 0 to inform all
        other processes of failures. Does nothing if not using distributed
        training. Usage example can be seen in examples/asr/jasper_an4.py

        Args:
            status (bool): Defaults to True. If any proccess passes False, it
                will trigger a graceful exit on all other processes. It is
                assumed that the process that passed False will print an error
                message on its own and exit
        """
        if self._world_size == 1:
            logging.info("sync_all_processes does nothing if there is one process")
            return
        if True:  # self._backend == Backend.PyTorch:
            import torch

            status_tensor = torch.cuda.IntTensor([status])
            torch.distributed.all_reduce(status_tensor, op=torch.distributed.ReduceOp.MIN)
            if status_tensor.item() == 0:
                logging.error("At least one process had a failure")
                if status:
                    raise ValueError(
                        f"Process with global rank {self._global_rank} entered"
                        " sync_all_processes with a passing status, but "
                        "another process indicated a failure"
                    )

    @property
    def world_size(self):
        return self._world_size

    @property
    def tb_writer(self):
        return self._tb_writer

    @property
    def placement(self):
        return self._placement

    @property
    def optim_level(self):
        return self._optim_level

    @property
    def checkpoint_dir(self):
        return self._exp_manager.ckpt_dir

    @property
    def work_dir(self):
        return self._exp_manager.work_dir

    @property
    def global_rank(self):
        return self._global_rank
