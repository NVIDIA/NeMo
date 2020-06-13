# Copyright (c) 2019 NVIDIA Corporation
import copy
import importlib
import itertools
import json
import os
from collections import defaultdict
from contextlib import ExitStack
from pathlib import Path
from typing import List, Optional

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP

from nemo import logging
from nemo.backends.pytorch.module_wrapper import TrainableNeuralModuleWrapper
from nemo.backends.pytorch.nm import DataLayerNM, TrainableNM
from nemo.backends.pytorch.optimizers import AdamW, Novograd, master_params
from nemo.core import DeploymentFormat, DeviceType, NeuralModule, NeuralModuleFactory, NmTensor
from nemo.core.actions import Actions, TrainingState, topological_sort_from_leaves
from nemo.core.callbacks import ActionCallback, NeMoCallback, SimpleLossLoggerCallback
from nemo.core.neural_factory import OperationMode, Optimization
from nemo.core.neural_types import AxisKind, NeuralType
from nemo.utils.app_state import AppState
from nemo.utils.decorators import deprecated
from nemo.utils.helpers import get_checkpoint_from_dir

# these imports will happen on as-needed basis
amp = None
LARC = None
FusedLAMB = None
FusedAdam = None
FusedNovoGrad = None

AmpOptimizations = {
    Optimization.mxprO0: "O0",
    Optimization.mxprO1: "O1",
    Optimization.mxprO2: "O2",
    Optimization.mxprO3: "O3",
}

_float_2_half_req = {
    Optimization.mxprO1,
    Optimization.mxprO2,
    Optimization.mxprO3,
}


class PtActions(Actions):
    def __init__(
        self, local_rank=None, global_rank=None, tb_writer=None, optimization_level=Optimization.mxprO0,
    ):
        need_apex = local_rank is not None or optimization_level != Optimization.mxprO0
        if need_apex:
            try:
                apex = importlib.import_module('apex')
                if optimization_level != Optimization.mxprO0:
                    global amp
                    amp = importlib.import_module('apex.amp')
                if local_rank is not None:
                    global LARC
                    global FusedLAMB
                    global FusedAdam
                    global FusedNovoGrad
                    parallel = importlib.import_module('apex.parallel')
                    apex_optimizer = importlib.import_module('apex.optimizers')
                    LARC = parallel.LARC
                    FusedLAMB = apex_optimizer.FusedLAMB
                    FusedAdam = apex_optimizer.FusedAdam
                    FusedNovoGrad = apex_optimizer.FusedNovoGrad

            except ImportError:
                raise ImportError(
                    "NVIDIA Apex is necessary for distributed training and"
                    "mixed precision training. It only works on GPUs."
                    "Please install Apex from "
                    "https://www.github.com/nvidia/apex"
                )

        super(PtActions, self).__init__(
            local_rank=local_rank, global_rank=global_rank, optimization_level=optimization_level,
        )

        self._step = 0
        self._epoch = 0
        self._optimizers = []
        self.tb_writer = tb_writer
        self.cache = None
        self.amp_initialized = False
        self.ddp_initialized = False
        self.ddp_module_dict = {}
        self._train_called = False

    @property
    def step(self):
        return self._step

    @step.setter
    def step(self, step):
        self._step = step

    @property
    def epoch(self):
        return self._epoch

    @epoch.setter
    def epoch(self, epoch):
        self._epoch = epoch

    @property
    @deprecated(version="0.12", explanation="epoch_num has been deprecated in favour of epoch.")
    def epoch_num(self):
        return self._epoch

    @epoch_num.setter
    @deprecated(version="0.12", explanation="epoch_num has been deprecated in favour of epoch.")
    def epoch_num(self):
        return self._epoch

    @property
    def optimizers(self):
        return self._optimizers

    def __get_top_sorted_modules_and_dataloader(self, hook: List[NmTensor]):
        """A function that accepts a list of NmTensors that need to be computed and constructs a call DAG that starts
        from a datalayerNM and can be used to compute the NmTensors.

        args:
            leaf_nmtensors (List[NmTensors]): The tensors to be computed

        returns:
            top_sorted_modules: the callchain DAG
            tdataset: the datalayer at the top of the callchain
        """
        top_sorted_modules = topological_sort_from_leaves(hook)

        if not isinstance(top_sorted_modules[0][0], DataLayerNM):
            raise ValueError("The first module in your DAG was not a DataLayer NeuralModule.")

        tdataset = top_sorted_modules[0][0].dataset

        for m in top_sorted_modules:
            if m[0].factory is None and self._local_rank is not None:
                raise ValueError(
                    "Neural module {0} was created without NeuralModuleFactory, but you are trying to run in "
                    "distributed mode. Please instantiate NeuralModuleFactory first and pass its instance as "
                    "`factory` parameter to all your Neural Module objects.".format(str(m[0]))
                )

        return top_sorted_modules, tdataset

    def create_optimizer(self, optimizer, things_to_optimize, optimizer_params=None):
        """
        Wrapper function around __setup_optimizer()

        Args:
            optimizer : A instantiated PyTorch optimizer or string. For
                currently supported strings, see __setup_optimizer().
            things_to_optimize (list): Must be a list of Neural Modules and/or
                parameters. If a Neural Module is passed, all trainable
                parameters are extracted and passed to the optimizer.
            optimizer_params (dict): Optional parameters dictionary.

        Returns:
            Optimizer
        """

        optimizer_instance = None
        optimizer_class = None
        if isinstance(optimizer, str):
            optimizer_class = optimizer
        elif isinstance(optimizer, torch.optim.Optimizer):
            optimizer_instance = optimizer
        else:
            raise ValueError("`optimizer` must be a string or an instance of torch.optim.Optimizer")

        modules_to_optimize = []
        tensors_to_optimize = []
        if not isinstance(things_to_optimize, list):
            things_to_optimize = [things_to_optimize]
        for thing in things_to_optimize:
            if isinstance(thing, NeuralModule):
                modules_to_optimize.append(thing)
            elif isinstance(thing, NmTensor):
                tensors_to_optimize.append(thing)
            else:
                raise ValueError(
                    "{} passed to create_optimizer() was neither a neural module nor a neural module tensor"
                )

        if tensors_to_optimize:
            call_chain, _ = self.__get_top_sorted_modules_and_dataloader(tensors_to_optimize)

            for module in call_chain:
                if module[0] not in modules_to_optimize:
                    modules_to_optimize.append(module[0])

        # Extract trainable weights which will be optimized
        params_list = [p.parameters() for p in modules_to_optimize if isinstance(p, TrainableNM) or p.is_trainable()]
        params_to_optimize = itertools.chain(*params_list)

        if optimizer_params is None:
            optimizer_params = {}
        # Init amp
        optimizer = self.__setup_optimizer(
            optimizer_instance=optimizer_instance,
            optimizer_class=optimizer_class,
            optimization_params=optimizer_params,
            params_to_optimize=params_to_optimize,
        )

        self._optimizers.append(optimizer)
        return optimizer

    @staticmethod
    def __setup_optimizer(
        optimizer_instance, optimizer_class, optimization_params, params_to_optimize,
    ):

        if optimizer_instance is None:
            # Setup optimizer instance, by default it is SGD
            lr = optimization_params["lr"]
            if optimizer_class.lower() == "sgd":
                optimizer = optim.SGD(
                    params_to_optimize,
                    lr=lr,
                    momentum=optimization_params.get("momentum", 0.9),
                    weight_decay=optimization_params.get("weight_decay", 0.0),
                )
            elif optimizer_class.lower() == "adam":
                optimizer = optim.Adam(
                    params=params_to_optimize, lr=lr, betas=optimization_params.get("betas", (0.9, 0.999)),
                )
            elif optimizer_class.lower() == "fused_adam":
                if not FusedAdam:
                    raise ValueError("FusedAdam works only with torch DDP.")
                optimizer = FusedAdam(
                    params=params_to_optimize,
                    lr=lr,
                    weight_decay=optimization_params.get("weight_decay", 0.0),
                    betas=optimization_params.get("betas", (0.9, 0.999)),
                )
            elif optimizer_class.lower() == "adam_w":
                optimizer = AdamW(
                    params=params_to_optimize,
                    lr=lr,
                    weight_decay=optimization_params.get("weight_decay", 0.0),
                    betas=optimization_params.get("betas", (0.9, 0.999)),
                )
            elif optimizer_class.lower() == "novograd":
                optimizer = Novograd(
                    params_to_optimize,
                    lr=lr,
                    weight_decay=optimization_params.get("weight_decay", 0.0),
                    luc=optimization_params.get("luc", False),
                    luc_trust=optimization_params.get("luc_eta", 1e-3),
                    betas=optimization_params.get("betas", (0.95, 0.25)),
                )
            elif optimizer_class.lower() == "fused_novograd":
                if not FusedNovoGrad:
                    raise ValueError("FusedNovoGrad works only with torch DDP.")
                optimizer = FusedNovoGrad(
                    params_to_optimize,
                    lr=lr,
                    weight_decay=optimization_params.get("weight_decay", 0.0),
                    reg_inside_moment=True,
                    grad_averaging=False,
                    betas=optimization_params.get("betas", (0.95, 0.25)),
                )
            elif optimizer_class.lower() == "fused_lamb":
                if not FusedLAMB:
                    raise ValueError("FusedLAMB works only with torch DDP.")
                optimizer = FusedLAMB(params_to_optimize, lr=lr,)
            else:
                raise ValueError("Unknown optimizer class: {0}".format(optimizer_class))

            if optimization_params.get("larc", False):
                if not LARC:
                    raise ValueError("LARC works only with torch DDP.")
                logging.info("Enabling larc")
                optimizer = LARC(optimizer, trust_coefficient=optimization_params.get("larc_eta", 2e-2),)
        else:
            logging.info("Optimizer instance: {0} is provided.")
            if optimizer_class is not None and optimizer_class != "":
                logging.warning("Ignoring `optimizer_class` parameter because `optimizer_instance` is provided")
            if optimization_params is not None and optimization_params != {}:
                logging.warning(
                    "Ignoring `optimization_params` parameter for optimizer because `optimizer_instance` is provided"
                )
            optimizer = optimizer_instance
        return optimizer

    def __initialize_amp(
        self, optimizer, optim_level, amp_max_loss_scale=2.0 ** 24, amp_min_loss_scale=1.0,
    ):
        if optim_level not in AmpOptimizations:
            raise ValueError(f"__initialize_amp() was called with unknown optim_level={optim_level}")
        # in this case, nothing to do here
        if optim_level == Optimization.mxprO0:
            return optimizer

        if len(AppState().modules) < 1:
            raise ValueError("There were no modules to initialize")
        pt_modules = []
        for module in AppState().modules:
            if isinstance(module, nn.Module):
                pt_modules.append(module)
            elif isinstance(module, TrainableNeuralModuleWrapper):
                pt_modules.append(module._pt_module)

        _, optimizer = amp.initialize(
            max_loss_scale=amp_max_loss_scale,
            min_loss_scale=amp_min_loss_scale,
            models=pt_modules,
            optimizers=optimizer,
            opt_level=AmpOptimizations[optim_level],
        )
        self.amp_initialized = True
        return optimizer

    def nm_graph_forward_pass(self, callchain, registered_tensors):
        self.__nm_graph_forward_pass(callchain, registered_tensors)

    def __nm_graph_forward_pass(
        self, call_chain, registered_tensors, mode=OperationMode.training, use_cache=False,
    ):
        for ind in range(1, len(call_chain)):
            if use_cache:
                in_cache = True
                for tensor in call_chain[ind][2].values():
                    if tensor is None:
                        # NM has an output tensor that is not used in the
                        # current call chain, so we don't care if it's not in
                        # cache
                        continue
                    if tensor.unique_name not in registered_tensors:
                        in_cache = False
                if in_cache:
                    continue
            call_args = call_chain[ind][1]
            m_id = call_chain[ind][0].unique_instance_id
            pmodule = self.ddp_module_dict[m_id] if self.ddp_initialized else call_chain[ind][0]

            if mode == OperationMode.training:
                # if module.is_trainable():
                if isinstance(pmodule, nn.Module):
                    pmodule.train()
            elif mode == OperationMode.evaluation:
                # if module.is_trainable():
                if isinstance(pmodule, nn.Module):
                    pmodule.eval()
            else:
                raise ValueError("Unknown OperationMode")
            # prepare call signature for `module`
            call_set = {}
            for tensor_name, nmtensor in call_args.items():
                key = nmtensor.unique_name
                call_set[tensor_name] = registered_tensors[key]
            new_tensors = pmodule(force_pt=True, **call_set)

            if not isinstance(new_tensors, List):
                if not isinstance(new_tensors, tuple):
                    new_tensors = [new_tensors]
                else:
                    new_tensors = list(new_tensors)
            for t_tensor, nm_tensor in zip(new_tensors, call_chain[ind][2].values()):
                if nm_tensor is None:
                    continue
                t_name = nm_tensor.unique_name
                if t_name not in registered_tensors or registered_tensors[t_name] is None:
                    registered_tensors[t_name] = t_tensor
                else:
                    raise ValueError(f"A NMTensor was produced twice in the same DAG. {t_name}")

    @staticmethod
    def pad_tensor(t: torch.Tensor, target_size: torch.Size):
        padded_shape = target_size.cpu().data.numpy().tolist()
        padded_t = torch.zeros(padded_shape).cuda().type_as(t)
        t_size = t.size()
        if len(t_size) == 0:
            padded_t = t
        elif len(t_size) == 1:
            padded_t[: t_size[0]] = t
        elif len(t_size) == 2:
            padded_t[: t_size[0], : t_size[1]] = t
        elif len(t_size) == 3:
            padded_t[: t_size[0], : t_size[1], : t_size[2]] = t
        elif len(t_size) == 4:
            padded_t[: t_size[0], : t_size[1], : t_size[2], : t_size[3]] = t
        else:
            raise NotImplementedError
        return padded_t

    @staticmethod
    def depad_tensor(t: torch.Tensor, original_size: torch.Size):
        t_size = original_size
        if len(t_size) == 0:
            depadded_t = t
        elif len(t_size) == 1:
            depadded_t = t[: t_size[0]]
        elif len(t_size) == 2:
            depadded_t = t[: t_size[0], : t_size[1]]
        elif len(t_size) == 3:
            depadded_t = t[: t_size[0], : t_size[1], : t_size[2]]
        elif len(t_size) == 4:
            depadded_t = t[: t_size[0], : t_size[1], : t_size[2], : t_size[3]]
        else:
            raise NotImplementedError
        return depadded_t

    def _eval(self, tensors_2_evaluate, callback, step, verbose=False):
        """
        Evaluation process.
        WARNING THIS function assumes that all tensors_2_evaluate are based
        on a single datalayer
        Args:
          tensors_2_evaluate: list of NmTensors to evaluate
          callback: instance of EvaluatorCallback
          step: current training step, used for logging

        Returns:
          None
        """
        with torch.no_grad():
            # each call chain corresponds to a tensor in tensors_2_evaluate
            call_chain, _ = self.__get_top_sorted_modules_and_dataloader(hook=tensors_2_evaluate)
            # "Retrieve" data layer from call chain.
            dl_nm = call_chain[0][0]

            # Prepare eval_dataloader
            # For distributed training it should have disjoint subsets of
            # all data on every worker
            is_distributed = False
            world_size = None
            if dl_nm.placement == DeviceType.AllGpu:
                assert dist.is_initialized()
                is_distributed = True
                world_size = torch.distributed.get_world_size()

                if dl_nm.dataset is not None:
                    sampler = None
                    if not isinstance(dl_nm.dataset, torch.utils.data.IterableDataset):
                        sampler = torch.utils.data.distributed.DistributedSampler(
                            dataset=dl_nm.dataset, shuffle=dl_nm.shuffle
                        )
                    dataloader_params = {
                        'dataset': dl_nm.dataset,
                        'sampler': sampler,
                        'num_workers': dl_nm.num_workers,
                        'batch_size': dl_nm.batch_size,
                        'shuffle': False,
                        'pin_memory': dl_nm.pin_memory,
                    }
                    if hasattr(dl_nm, 'collate_fn'):
                        dataloader_params['collate_fn'] = dl_nm.collate_fn
                    eval_dataloader = torch.utils.data.DataLoader(**dataloader_params)
                else:
                    eval_dataloader = dl_nm.data_iterator

                if hasattr(eval_dataloader, 'sampler'):
                    eval_dataloader.sampler.set_epoch(0)
            else:  # Not distributed
                if dl_nm.dataset is not None:
                    # Todo: remove local_parameters
                    dataloader_params = {
                        'dataset': dl_nm.dataset,
                        'sampler': None,  # not distributed sampler
                        'num_workers': dl_nm.num_workers,
                        'batch_size': dl_nm.batch_size,
                        'shuffle': dl_nm.shuffle,
                        'pin_memory': dl_nm.pin_memory,
                    }
                    if hasattr(dl_nm, 'collate_fn'):
                        dataloader_params['collate_fn'] = dl_nm.collate_fn
                    eval_dataloader = torch.utils.data.DataLoader(**dataloader_params)
                else:
                    eval_dataloader = dl_nm.data_iterator
            # after this eval_dataloader is ready to be used
            # reset global_var_dict - results of evaluation will be stored
            # there

            callback.clear_global_var_dict()
            dl_device = dl_nm._device

            # Evaluation mini-batch for loop
            num_batches = None
            if hasattr(eval_dataloader, "__len__"):
                num_batches = len(eval_dataloader)
            for epoch_i, data in enumerate(eval_dataloader, 0):
                if (
                    verbose
                    and num_batches is not None
                    and (num_batches < 10 or (epoch_i % int(num_batches / 10) == 0))
                ):
                    logging.info(f"Evaluating batch {epoch_i} out of {num_batches}")
                tensors = []
                if isinstance(data, torch.Tensor):
                    data = (data,)
                for d in data:
                    if isinstance(d, torch.Tensor):
                        tensors.append(d.to(dl_device))
                    else:
                        tensors.append(d)

                registered_e_tensors = {
                    t.unique_name: d for t, d in zip(call_chain[0][2].values(), tensors) if t is not None
                }
                self.__nm_graph_forward_pass(
                    call_chain=call_chain, registered_tensors=registered_e_tensors, mode=OperationMode.evaluation,
                )

                if not is_distributed or self.global_rank == 0:
                    values_dict = {}
                # If distributed. For the outer loop, we need to ensure that
                # all processes loop through the elements in the same order
                for t2e in tensors_2_evaluate:
                    key = t2e.unique_name
                    if key not in registered_e_tensors.keys():
                        logging.info("WARNING: Tensor {} was not found during eval".format(key))
                        continue
                    if is_distributed:
                        # where we will all_gather results from all workers
                        tensors_list = []
                        # where we will all_gather tensor sizes
                        tensor_on_worker = registered_e_tensors[key]

                        if not isinstance(tensor_on_worker, torch.Tensor):  # For string and other.
                            if self.global_rank == 0:
                                values_dict[key] = [tensor_on_worker] + ([None] * (world_size - 1))
                            continue

                        # https://github.com/pytorch/pytorch/issues/24137
                        is_bool = False
                        if tensor_on_worker.dtype == torch.bool:
                            is_bool = True
                            tensor_on_worker = tensor_on_worker.to(dtype=torch.long)

                        if tensor_on_worker.shape != torch.Size([]):
                            tensor_on_worker_size_as_tensor = torch.tensor(tensor_on_worker.shape).cuda()
                            sizes = []
                            for ind in range(world_size):
                                sizes.append(torch.empty_like(tensor_on_worker_size_as_tensor))
                            dist.all_gather(sizes, tensor_on_worker_size_as_tensor)
                            mx_dim, _ = torch.max(torch.stack(sizes), dim=0)
                        else:  # this is a singleton. For example, loss value
                            sizes = [torch.Size([])] * world_size
                            mx_dim = None
                        for ind in range(world_size):
                            # we have to use max shape for all_gather
                            if mx_dim is None:  # singletons
                                tensors_list.append(torch.tensor(2).cuda().type_as(tensor_on_worker))
                            else:  # non-singletons
                                tensors_list.append(
                                    torch.empty(mx_dim.cpu().data.numpy().tolist()).cuda().type_as(tensor_on_worker)
                                )

                        if mx_dim is not None:
                            t_to_send = self.pad_tensor(tensor_on_worker, mx_dim)
                        else:
                            t_to_send = tensor_on_worker
                        dist.all_gather(tensors_list, t_to_send)
                        tensors_list = [self.depad_tensor(t, size) for t, size in zip(tensors_list, sizes)]
                        if self.global_rank == 0:
                            values_dict["IS_FROM_DIST_EVAL"] = True
                            if is_bool:
                                tensors_list = [t.to(dtype=torch.bool) for t in tensors_list]
                            values_dict[key] = tensors_list
                    else:  # NON-DISTRIBUTED TRAINING
                        values_dict["IS_FROM_DIST_EVAL"] = False
                        values_dict[key] = [registered_e_tensors[key]]
                if callback.user_iter_callback and (self.global_rank is None or self.global_rank == 0):
                    # values_dict will contain results from all workers
                    callback.user_iter_callback(values_dict, callback._global_var_dict)

            # final aggregation (over minibatches) and logging of results
            # should happend only on one worker
            if callback.user_done_callback and (self.global_rank is None or self.global_rank == 0):
                vals_to_log = callback.user_done_callback(callback._global_var_dict)
                # log results to Tensorboard or Weights & Biases
                if vals_to_log is not None:
                    if hasattr(callback, 'swriter') and callback.swriter is not None:
                        if hasattr(callback, 'tb_writer_func') and callback.tb_writer_func is not None:
                            callback.tb_writer_func(callback.swriter, vals_to_log, step)
                        else:
                            for key, val in vals_to_log.items():
                                callback.swriter.add_scalar(key, val, step)
                    if hasattr(callback, 'wandb_log'):
                        callback.wandb_log(vals_to_log)

    def _infer(
        self, tensors_to_return, verbose=False, cache=False, use_cache=False, offload_to_cpu=True,
    ):
        """
        Does the same as _eval() just with tensors instead of eval callback.
        """
        # Checking that cache is used properly
        if cache and use_cache:
            raise ValueError(
                "cache and use_cache were both set. However cache must first be created prior to using it."
            )
        if cache:
            if self.cache is not None:
                raise ValueError("cache was set but was not empty")
            self.cache = []
        if use_cache:
            if not self.cache:
                raise ValueError("use_cache was set, but cache was empty")

        with torch.no_grad():
            # each call chain corresponds to a tensor in tensors_2_evaluate
            dl_nm = None
            call_chain, _ = self.__get_top_sorted_modules_and_dataloader(hook=tensors_to_return)
            dl_nm = call_chain[0][0]

            # Prepare eval_dataloader
            # For distributed training it should have disjoint subsets of
            # all data on every worker
            is_distributed = False
            world_size = None
            if dl_nm.placement == DeviceType.AllGpu:
                if self.cache or use_cache:
                    raise NotImplementedError("Caching is not available for distributed training.")
                assert dist.is_initialized()
                is_distributed = True
                world_size = torch.distributed.get_world_size()
                if dl_nm.dataset is not None:
                    sampler = None
                    if not isinstance(dl_nm.dataset, torch.utils.data.IterableDataset):
                        sampler = torch.utils.data.distributed.DistributedSampler(
                            dataset=dl_nm.dataset, shuffle=dl_nm.shuffle
                        )
                    dataloader_params = {
                        'dataset': dl_nm.dataset,
                        'sampler': sampler,
                        'num_workers': dl_nm.num_workers,
                        'batch_size': dl_nm.batch_size,
                        'shuffle': False,
                        'pin_memory': dl_nm.pin_memory,
                    }
                    if hasattr(dl_nm, 'collate_fn'):
                        dataloader_params['collate_fn'] = dl_nm.collate_fn
                    eval_dataloader = torch.utils.data.DataLoader(**dataloader_params)
                else:
                    eval_dataloader = dl_nm.data_iterator
                eval_dataloader.sampler.set_epoch(0)
            elif not use_cache:  # Not distributed and not using cache
                # Dataloaders are only used if use_cache is False
                # When caching, the DAG must cache all outputs from dataloader
                if dl_nm.dataset is not None:
                    # Todo: remove local_parameters
                    dataloader_params = {
                        'dataset': dl_nm.dataset,
                        'sampler': None,  # not distributed sampler
                        'num_workers': dl_nm.num_workers,
                        'batch_size': dl_nm.batch_size,
                        'shuffle': dl_nm.shuffle,
                        'pin_memory': dl_nm.pin_memory,
                    }
                    if hasattr(dl_nm, 'collate_fn'):
                        dataloader_params['collate_fn'] = dl_nm.collate_fn
                    eval_dataloader = torch.utils.data.DataLoader(**dataloader_params)
                else:
                    eval_dataloader = dl_nm.data_iterator
            # after this eval_dataloader is ready to be used
            # reset global_var_dict - results of evaluation will be stored
            # there

            if not is_distributed or self.global_rank == 0:
                values_dict = {}
                for t in tensors_to_return:
                    values_dict[t.unique_name] = []
            dl_device = dl_nm._device

            # Evaluation mini-batch for loop
            if use_cache:
                num_batches = len(self.cache)
                loop_iterator = self.cache
            else:
                num_batches = len(eval_dataloader)
                loop_iterator = eval_dataloader

            for epoch_i, data in enumerate(loop_iterator, 0):
                if verbose and (num_batches < 10 or (epoch_i % int(num_batches / 10) == 0)):
                    logging.info(f"Evaluating batch {epoch_i} out of {num_batches}")
                tensors = []
                if use_cache:
                    registered_e_tensors = data
                    # delete tensors_to_return
                    for t in tensors_to_return:
                        if t.unique_name in registered_e_tensors:
                            del registered_e_tensors[t.unique_name]
                    # Need to check for device type mismatch
                    for t in registered_e_tensors:
                        registered_e_tensors[t].to(dl_device)
                else:
                    if isinstance(data, torch.Tensor):
                        data = (data,)
                    for d in data:
                        if isinstance(d, torch.Tensor):
                            tensors.append(d.to(dl_device))
                        else:
                            tensors.append(d)

                    registered_e_tensors = {
                        t.unique_name: d for t, d in zip(call_chain[0][2].values(), tensors) if t is not None
                    }
                self.__nm_graph_forward_pass(
                    call_chain=call_chain,
                    registered_tensors=registered_e_tensors,
                    mode=OperationMode.evaluation,
                    use_cache=use_cache,
                )

                if cache:
                    self.append_to_cache(registered_e_tensors, offload_to_cpu)

                # If distributed. For the outer loop, we need to ensure that
                # all processes loop through the elements in the same order
                for t2e in tensors_to_return:
                    key = t2e.unique_name
                    if key not in registered_e_tensors.keys():
                        logging.info("WARNING: Tensor {} was not found during eval".format(key))
                        continue
                    if is_distributed:
                        # where we will all_gather results from all workers
                        tensors_list = []
                        # where we will all_gather tensor sizes
                        tensor_on_worker = registered_e_tensors[key]
                        if tensor_on_worker.shape != torch.Size([]):
                            tensor_on_worker_size_as_tensor = torch.tensor(tensor_on_worker.shape).cuda()
                            sizes = []
                            for ind in range(world_size):
                                sizes.append(torch.empty_like(tensor_on_worker_size_as_tensor))
                            dist.all_gather(sizes, tensor_on_worker_size_as_tensor)
                            mx_dim, _ = torch.max(torch.stack(sizes), dim=0)
                        else:  # this is a singleton. For example, loss value
                            sizes = [torch.Size([])] * world_size
                            mx_dim = None
                        for ind in range(world_size):
                            # we have to use max shape for all_gather
                            if mx_dim is None:  # singletons
                                tensors_list.append(torch.tensor(2).cuda().type_as(tensor_on_worker))
                            else:  # non-singletons
                                tensors_list.append(
                                    torch.empty(mx_dim.cpu().data.numpy().tolist()).cuda().type_as(tensor_on_worker)
                                )

                        if mx_dim is not None:
                            t_to_send = self.pad_tensor(tensor_on_worker, mx_dim)
                        else:
                            t_to_send = tensor_on_worker
                        dist.all_gather(tensors_list, t_to_send)
                        tensors_list = [self.depad_tensor(t, size) for t, size in zip(tensors_list, sizes)]
                        if offload_to_cpu:
                            tensors_list = [t.cpu() for t in tensors_list]
                        if self.global_rank == 0:
                            values_dict[key] += tensors_list
                    else:  # NON-DISTRIBUTED TRAINING
                        tensor = registered_e_tensors[key]
                        if offload_to_cpu and isinstance(tensor, torch.Tensor):
                            tensor = tensor.cpu()
                        values_dict[key] += [tensor]

            if not is_distributed or self.global_rank == 0:
                inferred_tensors = []
                for t in tensors_to_return:
                    inferred_tensors.append(values_dict[t.unique_name])
                return inferred_tensors

            # For all other ranks
            return None

    def append_to_cache(self, registered_tensors: dict, offload_to_cpu):
        """Simpler helper function to add results of __nm_graph_forward_pass to
        current cache.
        """
        if offload_to_cpu:
            for t in registered_tensors:
                registered_tensors[t] = registered_tensors[t].cpu()
        self.cache.append(registered_tensors)

    def clear_cache(self):
        """ Simple helpful function to clear cache by setting self.cache to
        None
        """
        self.cache = None

    def save_state_to(self, path: str):
        """
        Saves current state such as step, epoch and optimizer parameters
        Args:
          path:

        Returns:

        """
        state = {
            "step": self.step,
            "epoch": self.epoch,
            "optimizer_state": [opt.state_dict() for opt in self._optimizers],
        }
        torch.save(state, path)

    def restore_state_from(self, path: str):
        """
        Restores state such as step, epoch and optimizer parameters
        Args:
          path:

        Returns:

        """
        if os.path.isfile(path):
            # map_location could be cuda:<device_id> but cpu seems to be more
            # general since we are also saving step and epoch
            # load_state_dict should move the variables to the relevant device
            checkpoint = torch.load(path, map_location="cpu")
            self.step = checkpoint["step"]
            self.epoch = checkpoint["epoch"]
            if checkpoint["optimizer_state"]:
                for opt, opt_chkpt in zip(self._optimizers, checkpoint["optimizer_state"]):
                    opt.load_state_dict(opt_chkpt)
        else:
            raise FileNotFoundError("Could not find checkpoint file: {0}".format(path))

    @staticmethod
    def _check_all_tensors(list_of_tensors):
        """Method that checks if the passed list contains all NmTensors
        """
        if not isinstance(list_of_tensors, list):
            return False
        for tensor in list_of_tensors:
            if not isinstance(tensor, NmTensor):
                return False
        return True

    @staticmethod
    def _check_tuples(list_of_tuples):
        """Method that checks if the passed tuple contains an optimizer in the
        first element, and a list of NmTensors in the second.
        """
        for tup in list_of_tuples:
            if not (isinstance(tup[0], torch.optim.Optimizer) and PtActions._check_all_tensors(tup[1])):
                return False
        return True

    @staticmethod
    def __module_export(module, output, d_format: DeploymentFormat, input_example=None, output_example=None):
        # Check if output already exists
        destination = Path(output)
        if destination.exists():
            raise FileExistsError(f"Destination {output} already exists. " f"Aborting export.")

        input_names = list(module.input_ports.keys())
        output_names = list(module.output_ports.keys())
        dynamic_axes = defaultdict(list)

        def __extract_dynamic_axes(port_name: str, ntype: NeuralType, dynamic_axes: defaultdict):
            if ntype.axes:
                for ind, axis in enumerate(ntype.axes):
                    if axis.kind == AxisKind.Batch or axis.kind == AxisKind.Time:
                        dynamic_axes[port_name].append(ind)

        # extract dynamic axes and remove unnecessary inputs/outputs
        # for input_ports
        for port_name, ntype in module.input_ports.items():
            if port_name in module._disabled_deployment_input_ports:
                input_names.remove(port_name)
                continue
            __extract_dynamic_axes(port_name, ntype, dynamic_axes)
        # for output_ports
        for port_name, ntype in module.output_ports.items():
            if port_name in module._disabled_deployment_output_ports:
                output_names.remove(port_name)
                continue
            __extract_dynamic_axes(port_name, ntype, dynamic_axes)

        if len(dynamic_axes) == 0:
            dynamic_axes = None

        # Make a deep copy of init parameters.
        init_params_copy = copy.deepcopy(module._init_params)

        # Reset standard instance field - making the file (probably) lighter.
        module._init_params = None
        module._placement = None
        module._factory = None
        module._device = None

        module.eval()
        try:
            # Remove NeMo-related things from the module
            # We need to change __call__ method. Note that this will change the
            # whole class, not just this object! Which is why we need to repair it
            # in the finally block
            __orig_call__ = type(module).__call__
            type(module).__call__ = torch.nn.Module.__call__

            if d_format == DeploymentFormat.TORCHSCRIPT:
                if input_example is None:
                    # Route 1 - via torch.jit.script
                    traced_m = torch.jit.script(module)
                    traced_m.save(output)
                else:
                    # Route 2 - via tracing
                    traced_m = torch.jit.trace(module, input_example)
                    traced_m.save(output)
            elif d_format == DeploymentFormat.ONNX or d_format == DeploymentFormat.TRTONNX:
                if input_example is None:
                    raise ValueError(f'Example input is None, but ONNX tracing was' f' attempted')
                if output_example is None:
                    if isinstance(input_example, tuple):
                        output_example = module.forward(*input_example)
                    else:
                        output_example = module.forward(input_example)
                with torch.jit.optimized_execution(True):
                    jitted_model = torch.jit.trace(module, input_example)

                torch.onnx.export(
                    jitted_model,
                    input_example,
                    output,
                    input_names=input_names,
                    output_names=output_names,
                    verbose=False,
                    export_params=True,
                    do_constant_folding=True,
                    dynamic_axes=dynamic_axes,
                    opset_version=11,
                    example_outputs=output_example,
                )
                # fn = output + ".readable"
                # with open(fn, 'w') as f:
                #     tempModel = onnx.load(output)
                #     onnx.save(tempModel, output + ".copy")
                #     onnx.checker.check_model(tempModel)
                #     pgraph = onnx.helper.printable_graph(tempModel.graph)
                #     f.write(pgraph)

            elif d_format == DeploymentFormat.PYTORCH:
                torch.save(module.state_dict(), output)
                with open(output + ".json", 'w') as outfile:
                    json.dump(init_params_copy, outfile)

            else:
                raise NotImplementedError(f"Not supported deployment format: {d_format}")
        except Exception as e:  # nopep8
            logging.error(f'module export failed for {module} ' f'with exception {e}')
        finally:
            type(module).__call__ = __orig_call__

    @staticmethod
    def deployment_export(module, output: str, d_format: DeploymentFormat, input_example=None, output_example=None):
        """Exports Neural Module instance for deployment.

        Args:
            module: neural module to export
            output (str): where export results should be saved
            d_format (DeploymentFormat): which deployment format to use
            input_example: sometimes tracing will require input examples
            output_example: Should match inference on input_example
            amp_max_loss_scale (float): Max value for amp loss scaling.
                Defaults to 2.0**24.
        """

        with torch.no_grad():
            PtActions.__module_export(
                module=module,
                output=output,
                d_format=d_format,
                input_example=input_example,
                output_example=output_example,
            )

    def train(
        self,
        tensors_to_optimize=None,
        training_graph=None,
        optimizer=None,
        optimization_params=None,
        callbacks: Optional[List[ActionCallback]] = None,
        lr_policy=None,
        batches_per_step=None,
        stop_on_nan_loss=False,
        synced_batchnorm=False,
        synced_batchnorm_groupsize=0,
        gradient_predivide=False,
        amp_max_loss_scale=2.0 ** 24,
    ):
        def _perform_on_step_start(callbacks, state):
            # TODO: Most of these checks can be relaxed since we enforce callbacks
            # to be a list of ActionCallback objects
            if callbacks is not None and isinstance(callbacks, List) and len(callbacks) > 0:
                for callback in callbacks:
                    if isinstance(callback, ActionCallback):
                        callback.on_iteration_start()
                    elif isinstance(callback, NeMoCallback):
                        callback.on_step_start(state)
                    else:
                        raise ValueError(
                            "Callback was not a child of ActionCallback nor NeMoCallback and was not understood"
                        )

        def _perform_on_step_end(callbacks, state):
            if callbacks is not None and isinstance(callbacks, List) and len(callbacks) > 0:
                for callback in callbacks:
                    if isinstance(callback, ActionCallback):
                        callback.on_iteration_end()
                    elif isinstance(callback, NeMoCallback):
                        callback.on_step_end(state)
                    else:
                        raise ValueError(
                            "Callback was not a child of ActionCallback nor NeMoCallback and was not understood"
                        )

        def _perform_on_action_start(callbacks, state):
            if callbacks is not None and isinstance(callbacks, List) and len(callbacks) > 0:
                for callback in callbacks:
                    if isinstance(callback, ActionCallback):
                        callback.on_action_start()
                    elif isinstance(callback, NeMoCallback):
                        callback.on_action_start(state)
                    else:
                        raise ValueError(
                            "Callback was not a child of ActionCallback nor NeMoCallback and was not understood"
                        )

        def _perform_on_action_end(callbacks, state):
            if callbacks is not None and isinstance(callbacks, List) and len(callbacks) > 0:
                for callback in callbacks:
                    if isinstance(callback, ActionCallback):
                        callback.on_action_end()
                    elif isinstance(callback, NeMoCallback):
                        callback.on_action_end(state)
                    else:
                        raise ValueError(
                            "Callback was not a child of ActionCallback nor NeMoCallback and was not understood"
                        )

        def _perform_on_epoch_start(callbacks, state):
            if callbacks is not None and isinstance(callbacks, List) and len(callbacks) > 0:
                for callback in callbacks:
                    if isinstance(callback, ActionCallback):
                        callback.on_epoch_start()
                    elif isinstance(callback, NeMoCallback):
                        callback.on_epoch_start(state)
                    else:
                        raise ValueError(
                            "Callback was not a child of ActionCallback nor NeMoCallback and was not understood"
                        )

        def _perform_on_epoch_end(callbacks, state):
            if callbacks is not None and isinstance(callbacks, List) and len(callbacks) > 0:
                for callback in callbacks:
                    if isinstance(callback, ActionCallback):
                        callback.on_epoch_end()
                    elif isinstance(callback, NeMoCallback):
                        callback.on_epoch_end(state)
                    else:
                        raise ValueError(
                            "Callback was not a child of ActionCallback nor NeMoCallback and was not understood"
                        )

        def _perform_on_batch_start(callbacks, state):
            if callbacks is not None and isinstance(callbacks, List) and len(callbacks) > 0:
                for callback in callbacks:
                    if isinstance(callback, ActionCallback):
                        continue
                    elif isinstance(callback, NeMoCallback):
                        callback.on_batch_start(state)
                    else:
                        raise ValueError(
                            "Callback was not a child of ActionCallback nor NeMoCallback and was not understood"
                        )

        def _perform_on_batch_end(callbacks, state):
            if callbacks is not None and isinstance(callbacks, List) and len(callbacks) > 0:
                for callback in callbacks:
                    if isinstance(callback, ActionCallback):
                        continue
                    elif isinstance(callback, NeMoCallback):
                        callback.on_batch_end(state)
                    else:
                        raise ValueError(
                            "Callback was not a child of ActionCallback nor NeMoCallback and was not understood"
                        )

        def _init_callbacks(callbacks, action):
            if callbacks is not None and isinstance(callbacks, List) and len(callbacks) > 0:
                for callback in callbacks:
                    if isinstance(callback, ActionCallback):
                        callback.action = action

        def _update_callbacks(callbacks=None, registered_tensors=None, final_loss=None):
            # if self.local_rank is None or self.local_rank == 0:
            if callbacks is not None and isinstance(callbacks, List) and len(callbacks) > 0:
                for callback in callbacks:
                    if isinstance(callback, ActionCallback):
                        callback._registered_tensors = registered_tensors
                    else:  # For now, we can use the old callback function. In the future we should improve this
                        registered_tensors["loss"] = final_loss

        def get_state(action: 'PtAction'):
            """Helper function used to create a state for callbacks
            """

            class StateWrapper(dict):
                def __init__(self, action):
                    """A class that wraps a dictionary but adds the functions: restore_state_from and save_state_to
                    which are helper functions for CheckpointCallback to use.
                    The StateWrapper is a dictionary that contains the following mapping:
                        "step" (int): the current training step
                        "epoch" (int): the current epoch step
                        "local_rank" (int): the local rank that the process is running on
                        "global_rank" (int): the global rank that the process is running on
                        "optimizers" (list): a list of optimizers defined during the training process
                        "tensors" (TrainingState): A TrainingState object that can be used to access tensor values
                    """
                    self.action = action
                    super().__init__(
                        {
                            "step": action.step,
                            "tensors": action._training_state,
                            "epoch": action.epoch,
                            "local_rank": action.local_rank,
                            "global_rank": action.global_rank,
                            "optimizers": action.optimizers,
                        }
                    )

                def restore_state_from(self, path):
                    if os.path.isfile(path):
                        # map_location could be cuda:<device_id> but cpu seems to be more
                        # general since we are also saving step and epoch
                        # load_state_dict should move the variables to the relevant device
                        checkpoint = torch.load(path, map_location="cpu")
                        action.step = checkpoint["step"]
                        self["step"] = action.step
                        epoch = checkpoint.get("epoch", None)
                        if epoch is None:
                            epoch = checkpoint.get("epoch_num", None)
                        if epoch is None:
                            raise ValueError("Epoch was not found in the trainer checkpoint")
                        action.epoch = epoch
                        self["epoch"] = action.epoch
                        if checkpoint["optimizer_state"]:
                            for opt, opt_chkpt in zip(self["optimizers"], checkpoint["optimizer_state"]):
                                opt.load_state_dict(opt_chkpt)
                    else:
                        raise FileNotFoundError("Could not find checkpoint file: {0}".format(path))

                def save_state_to(self, path):
                    state = {
                        "step": self["step"],
                        "epoch": self["epoch"],
                        "optimizer_state": [opt.state_dict() for opt in self["optimizers"]],
                    }
                    torch.save(state, path)

            return StateWrapper(action)

        if self._train_called:
            logging.warning(
                "You called train twice. Please note that we do not support calling training twice in one script if "
                "amp or ddp is used. If you wish to call train twice, you need to run "
                "`nemo.utils.app_state.AppState().modules.clear(); neural_factory.reset_trainer()` and then "
                "reinstantiate all Neural Modules prior to calling train()"
            )
        self._train_called = True

        self._training_state = TrainingState(self)
        # Analyse the arguments passed to train.
        if tensors_to_optimize is not None and training_graph is not None:
            raise ValueError("Cannot pass both `tensors_to_optimize` and `training_graph` to the train() function")
        # if tensors_to_optimize is None and training_graph is None:
        #    raise ValueError(
        #        "One of the `tensors_to_optimize` or `training_graph` values must be passed to the train() function"
        #    )
        # Finally, unify.
        if training_graph is not None:
            # Get device - from NF.
            device_type = NeuralModuleFactory.get_default_factory().placement
            # Move the graph to device.
            training_graph.to(device_type=device_type)
            # To keep the "compatibility with old NeMo": get output tensors.
            tensors_to_optimize = training_graph.outputs.tensor_list

        if gradient_predivide:
            logging.error(
                "gradient_predivide is currently disabled, and is under consideration for removal in future versions. "
                "If this functionality is needed, please raise a github issue."
            )
        if not optimization_params:
            optimization_params = {}
        num_epochs = optimization_params.get("num_epochs", None)
        max_steps = optimization_params.get("max_steps", None)
        if num_epochs is None and max_steps is None:
            raise ValueError("You must specify either max_steps or num_epochs")
        grad_norm_clip = optimization_params.get('grad_norm_clip', None)

        if batches_per_step is None:
            batches_per_step = 1
        # this is necessary because we average gradients over batch
        bps_scale = torch.FloatTensor([1.0 / batches_per_step]).squeeze()

        if tensors_to_optimize is None:
            # This is Evaluation Mode
            _init_callbacks(callbacks, self)
            # Do action start callbacks
            _perform_on_action_end(callbacks, get_state(self))
            return
        # Check if tensors_to_optimize is just a list of NmTensors
        elif tensors_to_optimize is not None and (
            isinstance(tensors_to_optimize[0], NmTensor) and PtActions._check_all_tensors(tensors_to_optimize)
        ):
            # Parse graph into a topologically sorted sequence of neural
            # modules' calls
            (opt_call_chain, t_dataset,) = self.__get_top_sorted_modules_and_dataloader(hook=tensors_to_optimize)
            # Extract trainable weights which will be optimized
            params_list = [
                p[0].parameters() for p in opt_call_chain if isinstance(p[0], TrainableNM) or p[0].is_trainable()
            ]
            params_to_optimize = itertools.chain(*params_list)

            # Setup optimizer instance. By default it is SGD
            optimizer_instance = None
            optimizer_class = None
            if isinstance(optimizer, str):
                optimizer_class = optimizer
            elif isinstance(optimizer, torch.optim.Optimizer):
                optimizer_instance = optimizer
            else:
                raise ValueError("optimizer was not understood")
            optimizer = self.__setup_optimizer(
                optimizer_instance=optimizer_instance,
                optimizer_class=optimizer_class,
                optimization_params=optimization_params,
                params_to_optimize=params_to_optimize,
            )

            training_loop = [(optimizer, tensors_to_optimize, opt_call_chain)]

            self._optimizers.append(optimizer)
            assert len(self._optimizers) == 1, (
                "There was more than one optimizer, was create_optimizer() called before train()? Are you calling "
                "train() twice in one script, If so you need to call NeuralModuleFactory.reset_trainer() first."
            )

        elif PtActions._check_tuples(tensors_to_optimize):
            if batches_per_step != 1:
                raise ValueError("Gradient accumlation with multiple optimizers is not supported")
            datasets = []
            training_loop = []
            for step in tensors_to_optimize:
                (step_call_chain, dataset,) = self.__get_top_sorted_modules_and_dataloader(hook=step[1])
                datasets.append(dataset)
                training_loop.append((step[0], step[1], step_call_chain))

            t_dataset = datasets[0]
            for dataset in datasets:
                if type(dataset) is not type(t_dataset):
                    raise ValueError("There were two training datasets, we only support 1.")
        else:
            raise ValueError("tensors_to_optimize was not understood")

        logging_callchain = None
        # callbacks setup
        if callbacks is not None:
            for callback in callbacks:
                if not isinstance(callback, ActionCallback) and not isinstance(callback, NeMoCallback):
                    raise ValueError(
                        "A callback was received that was not a child of ActionCallback nor a child of NeMoCallback"
                    )
                elif isinstance(callback, SimpleLossLoggerCallback):
                    if logging_callchain:
                        raise ValueError("We only support one logger callback but more than one were found")
                    logger_step_freq = callback._step_freq
                    logging_tensors = callback.tensors
                    all_tensors = logging_tensors
                    for step in training_loop:
                        all_tensors = all_tensors + step[1]
                    (logging_callchain, _,) = self.__get_top_sorted_modules_and_dataloader(hook=all_tensors)

        # Intialize Amp if needed
        if self._optim_level in AmpOptimizations:
            # Store mapping of self.optimizers to optimizer in callchain
            training_loop_opts = []
            for opt in training_loop:
                training_loop_opts.append(self._optimizers.index(opt[0]))
            self._optimizers = self.__initialize_amp(
                optimizer=self._optimizers,
                optim_level=self._optim_level,
                amp_max_loss_scale=amp_max_loss_scale,
                amp_min_loss_scale=optimization_params.get('amp_min_loss_scale', 1.0),
            )
            # Use stored mapping to map amp_init opts to training loop
            for i, step in enumerate(training_loop):
                training_loop[i] = (
                    self._optimizers[training_loop_opts[i]],
                    step[1],
                    step[2],
                )

        dataNM = training_loop[0][2][0][0]
        placement_gpu = dataNM.placement == DeviceType.AllGpu
        if placement_gpu:
            logging.info("Doing distributed training")
            if t_dataset is not None:
                train_sampler = None
                if not isinstance(t_dataset, torch.utils.data.IterableDataset):
                    train_sampler = torch.utils.data.distributed.DistributedSampler(
                        dataset=t_dataset, shuffle=dataNM.shuffle
                    )
                dataloader_params = {
                    'dataset': t_dataset,
                    'sampler': train_sampler,
                    'num_workers': dataNM.num_workers,
                    'batch_size': dataNM.batch_size,
                    'shuffle': False,
                    'pin_memory': dataNM.pin_memory,
                }
                if hasattr(dataNM, 'collate_fn'):
                    dataloader_params['collate_fn'] = dataNM.collate_fn
                train_dataloader = torch.utils.data.DataLoader(**dataloader_params)
            else:
                train_dataloader = dataNM.data_iterator
                if hasattr(train_dataloader, 'sampler'):
                    train_sampler = train_dataloader.sampler
                else:
                    train_sampler = None

            self.ddp_initialized = True
            module_list = [mod.name for mod in AppState().modules]
            module_list = sorted(module_list)
            for module_name in module_list:
                module = AppState().modules[module_name]
                key = module.unique_instance_id
                num_trainable_weights = module.num_weights
                self.ddp_module_dict[key] = module
                if not isinstance(module, DDP) and isinstance(module, torch.nn.Module) and num_trainable_weights > 0:
                    # Per pytorch docs, convert sync bn prior to DDP
                    if synced_batchnorm:
                        world_size = dist.get_world_size()
                        sync_batchnorm_group = None
                        if synced_batchnorm_groupsize > 0:
                            if world_size % synced_batchnorm_groupsize != 0:
                                raise ValueError(
                                    f"Synchronized batch norm group size ({synced_batchnorm_groupsize}) must be 0"
                                    f" or divide total number of GPUs ({world_size})."
                                )
                            # Find ranks of other nodes in the same batchnorm group
                            rank = torch.distributed.get_rank()
                            group = rank // synced_batchnorm_groupsize
                            group_rank_ids = range(
                                group * synced_batchnorm_groupsize, (group + 1) * synced_batchnorm_groupsize
                            )
                            sync_batchnorm_group = torch.distributed.new_group(group_rank_ids)

                        module = nn.SyncBatchNorm.convert_sync_batchnorm(module, process_group=sync_batchnorm_group)

                    # By default, disable broadcast_buffers. This disables batch norm synchronization on forward
                    # pass
                    module = DDP(
                        module, device_ids=[self.local_rank], broadcast_buffers=False, find_unused_parameters=True
                    )
                    self.ddp_module_dict[key] = module

        # single GPU/CPU training
        else:
            if t_dataset is not None:
                train_sampler = None
                dataloader_params = {
                    'dataset': t_dataset,
                    'sampler': None,
                    'num_workers': dataNM.num_workers,
                    'batch_size': dataNM.batch_size,
                    'shuffle': dataNM.shuffle,
                    'pin_memory': dataNM.pin_memory,
                }
                if hasattr(dataNM, 'collate_fn'):
                    dataloader_params['collate_fn'] = dataNM.collate_fn

                train_dataloader = torch.utils.data.DataLoader(**dataloader_params)
            else:
                train_dataloader = dataNM.data_iterator
                train_sampler = None

        _init_callbacks(callbacks, self)
        # Do action start callbacks
        _perform_on_action_start(callbacks, get_state(self))

        # MAIN TRAINING LOOP
        # iteration over epochs
        while num_epochs is None or self.epoch < num_epochs:
            if train_sampler is not None:
                train_sampler.set_epoch(self.epoch)
            if max_steps is not None and self.step >= max_steps:
                break

            # Register epochs start with callbacks
            _perform_on_epoch_start(callbacks, get_state(self))

            # iteration over batches in epoch
            batch_counter = 0
            for _, data in enumerate(train_dataloader, 0):
                if max_steps is not None and self.step >= max_steps:
                    break

                if batch_counter == 0:
                    # Started step, zero gradients
                    curr_optimizer = training_loop[self.step % len(training_loop)][0]
                    curr_optimizer.zero_grad()
                    # Register iteration start with callbacks
                    _perform_on_step_start(callbacks, get_state(self))

                # Perform batch start callbacks
                _perform_on_batch_start(callbacks, get_state(self))

                # set learning rate policy
                if lr_policy is not None:
                    adjusted_lr = lr_policy(optimization_params["lr"], self.step, self.epoch)
                    for param_group in curr_optimizer.param_groups:
                        param_group["lr"] = adjusted_lr

                # TODO: Remove below loop when ActionCallback is removed
                if callbacks is not None:
                    for callback in callbacks:
                        if isinstance(callback, ActionCallback):
                            callback.learning_rate = curr_optimizer.param_groups[0]['lr']

                # registered_tensors will contain created tensors
                # named by output port and uuid of module which created them
                # Get and properly name tensors returned by data layer
                curr_call_chain = training_loop[self.step % len(training_loop)][2]
                dl_device = curr_call_chain[0][0]._device
                if logging_callchain and self.step % logger_step_freq == 0:
                    curr_call_chain = logging_callchain
                tensors = []
                if isinstance(data, torch.Tensor):
                    data = (data,)
                for d in data:
                    if isinstance(d, torch.Tensor):
                        tensors.append(d.to(dl_device))
                    else:
                        tensors.append(d)

                for t, d in zip(curr_call_chain[0][2].values(), tensors):
                    if t is not None:
                        self._training_state.set_tensor(t, d)
                disable_allreduce = batch_counter < (batches_per_step - 1)
                self.__nm_graph_forward_pass(
                    call_chain=curr_call_chain, registered_tensors=self._training_state.tensor_dict,
                )

                curr_tensors_to_optimize = training_loop[self.step % len(training_loop)][1]
                final_loss = 0
                for tensor in curr_tensors_to_optimize:
                    final_loss += self._training_state.tensor_dict[tensor.unique_name]

                # Check for NaN/inf loss (across workers if applicable)
                loss_nan_inf_checker = final_loss.clone()
                if placement_gpu:
                    dist.all_reduce(loss_nan_inf_checker, torch.distributed.ReduceOp.MAX)
                if torch.isnan(loss_nan_inf_checker).any() or torch.isinf(loss_nan_inf_checker).any():
                    if stop_on_nan_loss:
                        raise ValueError('Loss is NaN or inf - exiting')
                    if self._optim_level in AmpOptimizations and self._optim_level != Optimization.mxprO0:
                        logging.warning('Loss is NaN or inf.')
                    else:
                        # Skip this step across workers if loss is NaN/inf and using fp32
                        logging.warning('Loss is NaN or inf. Skipping update.')
                        self._training_state.clear_dict()  # Clear state dict here
                        continue

                if self._optim_level in AmpOptimizations and self._optim_level != Optimization.mxprO0:
                    with amp.scale_loss(final_loss, curr_optimizer, delay_unscale=disable_allreduce) as scaled_loss:
                        if disable_allreduce:
                            with ExitStack() as stack:
                                for mod in self.get_DDP_modules(curr_call_chain):
                                    stack.enter_context(mod.no_sync())
                                scaled_loss.backward(bps_scale.to(scaled_loss.get_device()))
                        else:
                            scaled_loss.backward(bps_scale.to(scaled_loss.get_device()))
                # no AMP optimizations needed
                else:
                    # multi-GPU, float32
                    if self._local_rank is not None:
                        if disable_allreduce:
                            with ExitStack() as stack:
                                for mod in self.get_DDP_modules(curr_call_chain):
                                    stack.enter_context(mod.no_sync())
                                final_loss.backward(bps_scale.to(final_loss.get_device()))
                        else:
                            final_loss.backward(bps_scale.to(final_loss.get_device()))
                    # single device (CPU or GPU)
                    else:
                        # Fix (workaround?) enabling to backpropagate gradients on CPUs.
                        if final_loss.get_device() < 0:
                            final_loss.backward(bps_scale)
                        else:
                            final_loss.backward(bps_scale.to(final_loss.get_device()))

                # Register batch end with callbacks
                _update_callbacks(
                    callbacks, registered_tensors=self._training_state.tensor_dict, final_loss=final_loss
                )
                # Perform batch end callbacks
                _perform_on_batch_end(callbacks, get_state(self))

                batch_counter += 1
                if batch_counter == batches_per_step:
                    # Ended step. Do optimizer update
                    if grad_norm_clip is not None:
                        torch.nn.utils.clip_grad_norm_(master_params(curr_optimizer), grad_norm_clip)
                    curr_optimizer.step()
                    batch_counter = 0
                    _perform_on_step_end(callbacks, get_state(self))
                    self.step += 1
                self._training_state.clear_dict()
            # End of epoch for loop
            # Register epochs end with callbacks
            _perform_on_epoch_end(callbacks, get_state(self))
            self.epoch += 1
        _perform_on_action_end(callbacks, get_state(self))

    def infer(
        self,
        tensors,
        checkpoint_dir=None,
        ckpt_pattern='',
        verbose=True,
        cache=False,
        use_cache=False,
        offload_to_cpu=True,
        modules_to_restore=None,
    ):
        """See NeuralModuleFactory.infer()
        """

        call_chain, _ = self.__get_top_sorted_modules_and_dataloader(hook=tensors)
        if checkpoint_dir:
            # Find all modules that need to be restored
            if modules_to_restore is None:
                modules_to_restore = []
                modules_to_restore_name = []
                for op in call_chain:
                    if op[0].num_weights > 0:
                        modules_to_restore.append(op[0])

            if not isinstance(modules_to_restore, list):
                modules_to_restore = [modules_to_restore]
            modules_to_restore_name = []
            for mod in modules_to_restore:
                if not isinstance(mod, NeuralModule):
                    raise ValueError("Found something that was not a Neural Module inside modules_to_restore")
                elif mod.num_weights == 0:
                    raise ValueError("Found a Neural Module with 0 weights inside modules_to_restore")
                modules_to_restore_name.append(str(mod))

            module_checkpoints = get_checkpoint_from_dir(modules_to_restore_name, checkpoint_dir, ckpt_pattern)

            for mod, checkpoint in zip(modules_to_restore, module_checkpoints):
                logging.info(f"Restoring {mod} from {checkpoint}")
                mod.restore_from(checkpoint, self._local_rank)

        # Init Amp
        if (
            self._optim_level in AmpOptimizations
            and self._optim_level != Optimization.mxprO0
            and not self.amp_initialized
        ):
            pt_modules = []
            for i in range(len(call_chain)):
                if isinstance(call_chain[i][0], nn.Module):
                    pt_modules.append(call_chain[i][0])
                elif isinstance(call_chain[i][0], TrainableNeuralModuleWrapper):
                    pt_modules.append(call_chain[i][0]._pt_module)

            amp.initialize(
                min_loss_scale=1.0, models=pt_modules, optimizers=None, opt_level=AmpOptimizations[self._optim_level],
            )
            self.amp_initialized = True

        # Run infer
        return self._infer(
            tensors_to_return=tensors,
            verbose=verbose,
            cache=cache,
            use_cache=use_cache,
            offload_to_cpu=offload_to_cpu,
        )

    def get_DDP_modules(self, call_chain):
        modules = []
        for ind in range(1, len(call_chain)):
            m_id = call_chain[ind][0].unique_instance_id
            module = self.ddp_module_dict[m_id]
            if isinstance(module, DDP):
                modules.append(module)

        return modules
