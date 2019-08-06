# Copyright (c) 2019 NVIDIA Corporation
import logging
import numpy as np
from abc import ABC, abstractmethod
from typing import List, Optional

from .callbacks import ActionCallback
from .neural_types import *


class Backend(Enum):
    """Supported backends. For now, it is only PyTorch."""

    PyTorch = 1
    NotSupported = 2


class ModelMode(Enum):
    """Training Mode or Evaluation/Inference"""

    train = 0
    eval = 1


class Optimization(Enum):
    """Various levels of Optimization.
    WARNING: This might have effect on model accuracy."""

    nothing = 0
    mxprO0 = 1
    mxprO1 = 2
    mxprO2 = 3
    mxprO3 = 4


class DeviceType(Enum):
    """Device types where Neural Modules can be placed."""

    GPU = 1
    CPU = 2
    AllGpu = 3


class Actions(ABC):
    """Basic actions allowed on graphs of Neural Modules"""

    def __init__(self, params, local_rank):
        self._parameters = params
        self._local_rank = local_rank
        if "optimization_level" in params:
            self._optim_level = params["optimization_level"]
        else:
            self._optim_level = Optimization.nothing
        self.step = None
        self.epoch_num = None

    @property
    def local_rank(self):
        """Local rank during distributed execution. None if single GPU/CPU

        Returns:
            (int) rank or worker or None if not in distributed model
        """
        return self._local_rank

    @abstractmethod
    def train(
            self,
            tensors_to_optimize: List[NmTensor],
            tensors_to_evaluate: Optional[List[NmTensor]],
            callbacks: Optional[List[ActionCallback]],
            lr_policy=None,
            batches_per_step=None,
            stop_on_nan_loss=False
    ):
        """This action executes training and (optionally) evaluation.

        Args:
            tensors_to_optimize: which tensors to optimize. Typically this is
                single loss tesnor.
            tensors_to_evaluate: which tensors to compute during evaluation.
            callbacks: list of callback objects
            lr_policy: function which should take (initial_lr, step, epoch) and
                return learning rate
            batches_per_step: number of mini-batches to process before one
                optimizer step. (default: None, same as 1). Use this
                to simulate larger batch sizes on hardware which could not fit
                larger batch in memory otherwise. Effectively, this will make
                "algorithmic" batch size per GPU/worker = batches_per_step*
                batch_size
            stop_on_nan_loss: (default: False) If set to True, the training
                will stop if loss=nan. If set to False, the training will
                continue, but the gradients will be zeroed before next
                mini-batch.

        Returns:
            None
        """
        pass

    @abstractmethod
    def infer(self, tensors: List[NmTensor]):
        """This action executes inference. Nothing is optimized.
        Args:
          tensors: which tensors to evaluate.

        Returns:
          None
        """
        pass

    @abstractmethod
    def save_state_to(self, path: str):
        """
        Saves current state such as step, epoch and optimizer parameters
        Args:
          path:

        Returns:

        """
        pass

    @abstractmethod
    def restore_state_from(self, path: str):
        """
        Restores state such as step, epoch and optimizer parameters
        Args:
          path:

        Returns:

        """
        pass

    def _perform_on_iteration_start(self, callbacks):
        if callbacks is not None and isinstance(callbacks, List) and len(
                callbacks) > 0:
            for callback in callbacks:
                callback._local_rank = self.local_rank
                callback.on_iteration_start()

    def _perform_on_iteration_end(self, callbacks):
        if callbacks is not None and isinstance(callbacks, List) and len(
                callbacks) > 0:
            for callback in callbacks:
                callback._local_rank = self.local_rank
                callback.on_iteration_end()

    def _perform_on_action_start(self, callbacks):
        if callbacks is not None and isinstance(callbacks, List) and len(
                callbacks) > 0:
            for callback in callbacks:
                callback._local_rank = self.local_rank
                callback.on_action_start()

    def _perform_on_action_end(self, callbacks):
        if callbacks is not None and isinstance(callbacks, List) and len(
                callbacks) > 0:
            for callback in callbacks:
                callback._local_rank = self.local_rank
                callback.on_action_end()

    def _perform_on_epoch_start(self, callbacks):
        if callbacks is not None and isinstance(callbacks, List) and len(
                callbacks) > 0:
            for callback in callbacks:
                callback._local_rank = self.local_rank
                callback.on_epoch_start()

    def _perform_on_epoch_end(self, callbacks):
        if callbacks is not None and isinstance(callbacks, List) and len(
                callbacks) > 0:
            for callback in callbacks:
                callback._local_rank = self.local_rank
                callback.on_epoch_end()

    def _fill_callbacks(
            self,
            callbacks=None,
            tensors_to_optimize=None,
            tensors_to_evaluate=None,
            registered_tensors=None,
    ):
        # if self.local_rank is None or self.local_rank == 0:
        if callbacks is not None and isinstance(callbacks, List) and len(
                callbacks) > 0:
            for callback in callbacks:
                callback._step = self.step
                callback._epoch_num = self.epoch_num
                callback._tensors_to_optimize = tensors_to_optimize
                callback._tensors_to_evaluate = tensors_to_evaluate
                callback._registered_tensors = registered_tensors
                callback._local_rank = self.local_rank


class NeuralModuleFactory(object):
    """
    Neural Module Factory instance is used to create neural modules and
    trainers

    Args:
        backend (Backend): Currently only Backend.PyTorch is supported
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
    """

    def __init__(
            self,
            backend=Backend.PyTorch,
            local_rank=None,
            optimization_level=Optimization.nothing,
            placement=DeviceType.GPU,
            cudnn_benchmark=False,
            random_seed=None,
            master_process=True,
    ):
        self._local_rank = local_rank
        self._optim_level = optimization_level
        self._placement = placement
        self._backend = backend
        self._world_size = 1
        self._master_process = master_process
        if backend == Backend.PyTorch:
            # TODO: Move all framework specific code from this file
            import torch
            torch.backends.cudnn.benchmark = cudnn_benchmark
            if random_seed is not None and cudnn_benchmark:
                raise ValueError("cudnn_benchmark can not be set to True"
                                 "when random_seed is not None.")
            if random_seed is not None:
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
                torch.manual_seed(random_seed)
                np.random.seed(random_seed)

            if self._local_rank is not None:
                torch.cuda.set_device(self._local_rank)
                torch.distributed.init_process_group(
                    backend="nccl", init_method="env://"
                )
                self._world_size = torch.distributed.get_world_size()
        else:
            raise NotImplementedError(
                "Only Pytorch backend is currently supported.")

    @staticmethod
    def __name_import(name):
        components = name.split(".")
        mod = __import__(components[0])
        for comp in components[1:]:
            mod = getattr(mod, comp)
        return mod

    def __get_pytorch_module(self, name, params, collection, pretrained):
        params["factory"] = self
        if collection == "toys" or collection == "tutorials" or collection \
                == "other":
            constructor = NeuralModuleFactory.__name_import(
                "nemo.backends.pytorch.tutorials." + name
            )
        elif collection == "nemo_nlp":
            constructor = NeuralModuleFactory.__name_import(
                "nemo_nlp." + name
            )
            if name == "BERT" and pretrained is True:
                params["pretrained"] = True
        elif collection == "nemo_asr":
            constructor = NeuralModuleFactory.__name_import(
                "nemo_asr." + name
            )
        elif collection == "nemo_lpr":
            constructor = NeuralModuleFactory.__name_import(
                "nemo_lpr." + name
            )
        elif collection == 'common':
            constructor = NeuralModuleFactory.__name_import(
                'nemo.backends.pytorch.common.' + name
            )
        elif collection == "torchvision":
            import torchvision.models as tv_models
            import nemo.backends.pytorch.module_wrapper as mw
            import torch.nn as nn

            if name == "ImageFolderDataLayer":
                constructor = NeuralModuleFactory.__name_import(
                    "nemo.backends.pytorch.torchvision.data." + name
                )
                instance = constructor(**params)
                return instance
            else:
                _nm_name = name.lower()
                if _nm_name == "resnet18":
                    input_ports = {
                        "x": NeuralType(
                            {
                                0: AxisType(BatchTag),
                                1: AxisType(ChannelTag),
                                2: AxisType(HeightTag, 224),
                                3: AxisType(WidthTag, 224),
                            }
                        )
                    }
                    output_ports = {
                        "output": NeuralType(
                            {0: AxisType(BatchTag), 1: AxisType(ChannelTag)}
                        )
                    }

                    pt_model = tv_models.resnet18(pretrained=pretrained)
                    num_classes = params.get("num_classes", None)
                    if num_classes is not None:
                        pt_model.fc = nn.Linear(512, params["num_classes"])
                    return mw.TrainableNeuralModuleWrapper(
                        pt_nn_module=pt_model,
                        input_ports_dict=input_ports,
                        output_ports_dict=output_ports,
                        **params,
                    )
                elif _nm_name == "resnet50":
                    input_ports = {
                        "x": NeuralType(
                            {
                                0: AxisType(BatchTag),
                                1: AxisType(ChannelTag),
                                2: AxisType(HeightTag, 224),
                                3: AxisType(WidthTag, 224),
                            }
                        )
                    }
                    output_ports = {
                        "output": NeuralType(
                            {0: AxisType(BatchTag), 1: AxisType(ChannelTag)}
                        )
                    }

                    pt_model = tv_models.resnet50(pretrained=pretrained)
                    num_classes = params.get("num_classes", None)
                    if num_classes is not None:
                        pt_model.fc = nn.Linear(2048, params["num_classes"])
                    return mw.TrainableNeuralModuleWrapper(
                        pt_nn_module=pt_model,
                        input_ports_dict=input_ports,
                        output_ports_dict=output_ports,
                        **params,
                    )
        else:
            collection_path = "nemo.collections." + collection + "." + name
            constructor = NeuralModuleFactory.__name_import(collection_path)
            if name == "BERT" and pretrained is True:
                params["pretrained"] = True

        if "placement" not in params:
            params["placement"] = self._placement
        instance = constructor(**params)
        return instance

    def get_module(self, name, params, collection, pretrained=False):
        """
        Creates NeuralModule instance

        Args:
          name (str): name of NeuralModule which instance should be returned.
          params (dict): local parameters which should be passed to
          NeuralModule's constructor.
          collection (str): in which collection to look for
          `neural_module_name`
          pretrained (bool): return pre-trained instance or randomly
          initialized (default)

        Returns:
          NeuralModule instance
        """
        if params is not None and "optimization_level" in params:
            if params["optimization_level"] != self._optim_level:
                if self._master_process:
                    logging.warning(
                        "Module's {0} requested optimization level {1} is"
                        "different from the one specified by factory - {2}."
                        "Using: {3} for this module".format(
                            name,
                            params["optimization_level"],
                            self._optim_level,
                            params["optimization_level"],
                        )
                    )
        else:
            if params is None:
                params = {}
            params["optimization_level"] = self._optim_level

        if self._backend == Backend.PyTorch:
            return self.__get_pytorch_module(
                name=name, params=params, collection=collection,
                pretrained=pretrained
            )
        else:
            return None

    def get_trainer(self, params, tb_writer=None):
        if self._backend == Backend.PyTorch:
            params["optimization_level"] = self._optim_level
            constructor = NeuralModuleFactory.__name_import(
                "nemo.backends.pytorch.PtActions"
            )
            instance = constructor(params=params,
                                   local_rank=self._local_rank,
                                   tb_writer=tb_writer)
            return instance
        else:
            raise ValueError("Only PyTorch backend is currently supported.")

    @property
    def world_size(self):
        return self._world_size

    @property
    def placement(self):
        return self._placement

    @property
    def optim_level(self):
        return self._optim_level

    @property
    def master_process(self):
        return self._master_process
