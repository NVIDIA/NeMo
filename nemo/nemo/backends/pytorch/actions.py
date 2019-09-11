# Copyright (c) 2019 NVIDIA Corporation
import itertools
import logging
import os
from typing import List, Optional

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from nemo.backends.pytorch.nm import TrainableNM

from .module_wrapper import TrainableNeuralModuleWrapper
from .nm import DataLayerNM
from .optimizers import Novograd, AdamW, Lamb
from ...core import NmTensor, DeviceType, NeuralModule
from ...core.callbacks import (
    ActionCallback,
    EvaluatorCallback,
    SimpleLossLoggerCallback,
)
from ...core.neural_factory import Actions, ModelMode, Optimization
from ...utils.helpers import get_checkpoint_from_dir

try:
    import apex
    from apex.parallel import DistributedDataParallel as DDP
    from apex.parallel.LARC import LARC
    from apex import amp
except ImportError:
    raise ImportError(
        "Please install apex from https://www.github.com/nvidia/apex")

AmpOptimizations = {
    Optimization.mxprO0: "O0",
    Optimization.mxprO1: "O1",
    Optimization.mxprO2: "O2",
    Optimization.mxprO3: "O3",
}

_float_2_half_req = {Optimization.mxprO1,
                     Optimization.mxprO2,
                     Optimization.mxprO3}


class PtActions(Actions):
    def __init__(self, local_rank=None, tb_writer=None,
                 optimization_level=Optimization.mxprO0):
        super(PtActions, self).__init__(
            local_rank=local_rank,
            optimization_level=optimization_level)

        # will be [unique_instance_id -> (NMModule, PTModule)]
        self.module_reference_table = {}
        self.step = 0
        self.epoch_num = 0
        self.optimizers = []
        self.tb_writer = tb_writer
        self._modules = set()

    @property
    def modules(self):
        return self._modules

    def __get_top_sorted_modules_and_dataloader(self, hook):
        """
        Constracts DAG leading to hook and creates its topological order.
        It also populates self.module_reference_table.
        Args:
          hook: an NmTensor or a list of NmTensors representing leaf nodes
          in DAG

        Returns:
          list of modules with their call arguments and outputs, and dataset
        """

        def create_node(producer, producer_args):
            if producer_args is None:
                return tuple((producer, ()))
            else:
                return tuple(
                    (
                        producer,
                        tuple([(k, v) for k, v in producer_args.items()]))
                )

        def is_in_degree_zero(node, processed_nodes):
            """A node has in degree of zero"""
            if node[1] == ():
                return True
            for portname, nmtensor in node[1]:
                nd = create_node(nmtensor.producer, nmtensor.producer_args)
                if nd not in processed_nodes:
                    return False
            return True

        if not isinstance(hook, list):
            hooks = [hook]
        else:
            hooks = hook

        # ensures that no tensors are processed twice
        processed_nmtensors = set()

        indices_to_remove = []
        # Check for duplicates in hook
        for i, nmtensor in enumerate(hook):
            if nmtensor in processed_nmtensors:
                indices_to_remove.append(i)
            else:
                processed_nmtensors.add(nmtensor)

        for i in reversed(indices_to_remove):
            hook.pop(i)

        _top_sorted_modules = []
        all_nodes = {}

        # extract all nodes to all_nodes set
        hooks_lst = list(hooks)
        while len(hooks_lst) > 0:
            # take nmtensor from the end of the list
            nmtensor = hooks_lst.pop()
            node = create_node(nmtensor.producer, nmtensor.producer_args)
            # Store nmtensor as an output of its producer
            # first make sure all keys are present per output port
            # and nm is inside all_nodes
            if node not in all_nodes:
                all_nodes[node] = {
                    k: None for k in nmtensor.producer._output_ports}
            # second, populate output port with current nmtensor
            # where applicable
            all_nodes[node][nmtensor.name] = nmtensor
            processed_nmtensors.add(nmtensor)
            if (nmtensor.producer_args is not None
                    and nmtensor.producer_args != {}):
                for _, new_nmtensor in nmtensor.producer_args.items():
                    if new_nmtensor not in processed_nmtensors:
                        # put in the start of list
                        hooks_lst.insert(0, new_nmtensor)

        all_node_with_output = []
        # Iterate over all_nodes to create new nodes that include its output
        # now all nodes have (module, input tensors, output tensors)
        for node in all_nodes:
            all_node_with_output.append(tuple((
                node[0],
                node[1],
                all_nodes[node]
            )))

        processed_nodes = []
        while len(all_node_with_output) > 0:
            for node in all_node_with_output.copy():
                # if node's in_degree is zero it can be added to
                # _top_sorted_modules
                # this will also reduce in_degree of its children
                if is_in_degree_zero(node, processed_nodes):
                    _top_sorted_modules.append(node)
                    processed_nodes.append((node[0], node[1]))
                    all_node_with_output.remove(node)

        # Create top_sorted_modules aka callchain
        top_sorted_modules = []
        for i, m in enumerate(_top_sorted_modules):
            top_sorted_modules.append((m[0], dict(m[1]), m[2]))
            # Ensure that there is only one dataset in callchain
            if i > 0 and isinstance(m[0], DataLayerNM):
                raise ValueError(
                    "There were more than one DataLayer NeuralModule inside "
                    "your DAG.")

        if not isinstance(top_sorted_modules[0][0], DataLayerNM):
            raise ValueError(
                "The first module in your DAG was not a DataLayer "
                "NeuralModule.")

        tdataset = top_sorted_modules[0][0].dataset

        # populate self.module_reference_table
        for m in top_sorted_modules:
            if m[0].factory is None and self._local_rank is not None:
                raise ValueError("Neural module {0} was created without "
                                 "NeuralModuleFactory, but you are trying to"
                                 "run in distributed mode. Please instantiate"
                                 "NeuralModuleFactory first and pass its "
                                 "instance as `factory` parameter to all your"
                                 "Neural Module objects."
                                 "".format(m[0].__class__.__name__))
            key = m[0].unique_instance_id
            if key not in self.module_reference_table:
                if isinstance(m[0], TrainableNeuralModuleWrapper):
                    self.module_reference_table[key] = (m[0], m[0]._pt_module)
                else:
                    self.module_reference_table[key] = (m[0], m[0])

        return top_sorted_modules, tdataset

    def create_optimizer(
            self,
            optimizer,
            things_to_optimize,
            optimizer_params=None,
    ):
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
            raise ValueError("`optimizer` must be a string or an instance "
                             "of torch.optim.Optimizer")

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
                raise ValueError("{} passed to create_optimizer() was neither "
                                 "a neural module nor a neural module tensor")

        if tensors_to_optimize:
            call_chain, _ = self.__get_top_sorted_modules_and_dataloader(
                tensors_to_optimize)

            for module in call_chain:
                if module[0] not in modules_to_optimize:
                    modules_to_optimize.append(module[0])

        # Extract trainable weights which will be optimized
        params_list = [
            p.parameters() for p in modules_to_optimize
            if isinstance(p, TrainableNM) or p.is_trainable()
        ]
        params_to_optimize = itertools.chain(*params_list)

        if optimizer_params is None:
            optimizer_params = {}
        # Init amp
        optimizer = self.__setup_optimizer(
            optimizer_instance=optimizer_instance,
            optimizer_class=optimizer_class,
            optimization_params=optimizer_params,
            params_to_optimize=params_to_optimize)

        self.optimizers.append(optimizer)
        return optimizer

    @staticmethod
    def __setup_optimizer(
            optimizer_instance,
            optimizer_class,
            optimization_params,
            params_to_optimize,
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
                    params=params_to_optimize, lr=lr,
                    betas=optimization_params.get("betas", (0.9, 0.999)))
            elif optimizer_class.lower() == "fused_adam":
                optimizer = apex.optimizers.FusedAdam(
                    params=params_to_optimize,
                    lr=lr)
            elif optimizer_class.lower() == "adam_w":
                optimizer = AdamW(
                    params=params_to_optimize,
                    lr=lr,
                    weight_decay=optimization_params.get("weight_decay", 0.0),
                )
            elif optimizer_class.lower() == "novograd":
                optimizer = Novograd(
                    params_to_optimize,
                    lr=lr,
                    weight_decay=optimization_params.get("weight_decay", 0.0),
                    luc=optimization_params.get("luc", False),
                    luc_trust=optimization_params.get("luc_eta", 1e-3),
                    betas=optimization_params.get("betas", (0.95, 0.98)),
                )
            elif optimizer_class.lower() == "fused_novograd":
                optimizer = apex.optimizers.FusedNovoGrad(
                    params_to_optimize,
                    lr=lr,
                    weight_decay=optimization_params.get("weight_decay", 0.0),
                    reg_inside_moment=True,
                    grad_averaging=False,
                    betas=optimization_params.get("betas", (0.95, 0.25)),
                )
            elif optimizer_class.lower() == "lamb":
                optimizer = Lamb(
                    params_to_optimize,
                    lr=lr,
                    weight_decay=optimization_params.get("weight_decay", 0.0),
                )
            else:
                raise ValueError(
                    "Unknown optimizer class: {0}".format(optimizer_class))

            if optimization_params.get("larc", False):
                print("Enabling larc")
                optimizer = LARC(
                    optimizer,
                    trust_coefficient=optimization_params.get("larc_eta", 2e-2)
                )
        else:
            logging.info("Optimizer instance: {0} is provided.")
            if optimizer_class is not None and optimizer_class != "":
                logging.warning("Ignoring `optimizer_class` parameter because"
                                "`optimizer_instance` is provided")
            if optimization_params is not None and optimization_params != {}:
                logging.warning("Ignoring `optimization_params` parameter for "
                                "optimizer because `optimizer_instance` "
                                "is provided")
            optimizer = optimizer_instance
        return optimizer

    def __initialize_amp(
            self, optimizer, optim_level, amp_min_loss_scale=1.0
    ):
        if optim_level not in AmpOptimizations:
            raise ValueError("__initialize_amp() was called but optim_level "
                             "was set to float32.")
        if len(self.modules) < 1:
            raise ValueError("There were no modules to initialize")
        pt_modules = []
        for module in self.modules:
            if isinstance(module, nn.Module):
                pt_modules.append(module)
            elif isinstance(module,
                            TrainableNeuralModuleWrapper):
                pt_modules.append(module._pt_module)

        _, optimizer = amp.initialize(
            min_loss_scale=amp_min_loss_scale,
            models=pt_modules,
            optimizers=optimizer,
            opt_level=AmpOptimizations[optim_level],
        )
        return optimizer

    def __nm_graph_forward_pass(
            self,
            call_chain,
            registered_tensors,
            mode=ModelMode.train,
            disable_allreduce=False
    ):
        for ind in range(1, len(call_chain)):
            call_args = call_chain[ind][1]
            # module = call_chain[ind][0]
            m_id = call_chain[ind][0].unique_instance_id
            pmodule = self.module_reference_table[m_id][1]

            if isinstance(pmodule, DDP):
                if disable_allreduce:
                    pmodule.disable_allreduce()
                else:
                    pmodule.enable_allreduce()

            if mode == ModelMode.train:
                # if module.is_trainable():
                if isinstance(pmodule, nn.Module):
                    pmodule.train()
            elif mode == ModelMode.eval:
                # if module.is_trainable():
                if isinstance(pmodule, nn.Module):
                    pmodule.eval()
            else:
                raise ValueError("Unknown ModelMode")
            # prepare call signature for `module`
            call_set = {}
            for tensor_name, nmtensor in call_args.items():
                # _add_uuid_2_name(nmtensor.name, nmtensor.producer._uuid)
                key = nmtensor.unique_name
                call_set[tensor_name] = registered_tensors[key]
            # actual PyTorch module call with signature
            if isinstance(
                    self.module_reference_table[m_id][0],
                    TrainableNeuralModuleWrapper
            ):
                new_tensors = pmodule(**call_set)
            else:
                new_tensors = pmodule(force_pt=True, **call_set)

            if not isinstance(new_tensors, List):
                if not isinstance(new_tensors, tuple):
                    new_tensors = [new_tensors]
                else:
                    new_tensors = list(new_tensors)
            for t_tensor, nm_tensor in zip(
                    new_tensors, call_chain[ind][2].values()):
                if nm_tensor is None:
                    continue
                t_name = nm_tensor.unique_name
                if t_name not in registered_tensors:
                    registered_tensors[t_name] = t_tensor
                else:
                    raise ValueError(
                        "A NMTensor was produced twice in the same DAG. "
                        "{}".format(t_name))

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
            dl_nm = None
            call_chain, _ = self.__get_top_sorted_modules_and_dataloader(
                hook=tensors_2_evaluate
            )
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
                # print(
                #     "Doing distributed evaluation. Rank {0} of {1}".format(
                #         self.local_rank, world_size
                #     )
                # )
                if dl_nm.dataset is not None:
                    sampler = torch.utils.data.distributed.DistributedSampler(
                        dl_nm.dataset
                    )
                    eval_dataloader = torch.utils.data.DataLoader(
                        dataset=dl_nm.dataset,
                        sampler=sampler,
                        num_workers=dl_nm.local_parameters.get(
                            "num_workers", os.cpu_count()
                        ),
                        batch_size=dl_nm.local_parameters["batch_size"],
                        shuffle=(sampler is None),
                    )
                else:
                    eval_dataloader = dl_nm.data_iterator
                eval_dataloader.sampler.set_epoch(0)
            else:  # Not distributed
                if dl_nm.dataset is not None:
                    # Todo: remove local_parameters
                    eval_dataloader = torch.utils.data.DataLoader(
                        dataset=dl_nm.dataset,
                        sampler=None,  # not distributed sampler
                        num_workers=call_chain[0][0].local_parameters.get(
                            "num_workers", os.cpu_count()
                        ),
                        batch_size=call_chain[0][0].local_parameters[
                            "batch_size"],
                        shuffle=call_chain[0][0].local_parameters.get(
                            "shuffle",
                            False),
                    )
                else:
                    eval_dataloader = dl_nm.data_iterator
            # after this eval_dataloader is ready to be used
            # reset global_var_dict - results of evaluation will be stored
            # there

            callback.clear_global_var_dict()
            dl_device = dl_nm._device

            # Evaluation mini-batch for loop
            num_batches = len(eval_dataloader)
            for epoch_i, data in enumerate(eval_dataloader, 0):
                if verbose and (
                        num_batches < 10 or (
                        epoch_i % int(num_batches / 10) == 0)
                ):
                    print(
                        f"Evaluating batch {epoch_i} out of {num_batches}")
                tensors = []
                if isinstance(data, torch.Tensor):
                    data = (data,)
                for d in data:
                    if isinstance(d, torch.Tensor):
                        tensors.append(d.to(dl_device))
                    else:
                        tensors.append(d)

                registered_e_tensors = {t.unique_name: d for t, d in
                                        zip(call_chain[0][2].values(), tensors)
                                        if t is not None
                                        }
                self.__nm_graph_forward_pass(
                    call_chain=call_chain,
                    registered_tensors=registered_e_tensors,
                    mode=ModelMode.eval,
                )

                if not is_distributed or self.local_rank == 0:
                    values_dict = {}
                # If distributed. For the outer loop, we need to ensure that
                # all processes loop through the elements in the same order
                for t2e in tensors_2_evaluate:
                    key = t2e.unique_name
                    if key not in registered_e_tensors.keys():
                        print(
                            "WARNING: Tensor {} was not found during "
                            "eval".format(
                                key)
                        )
                        continue
                    if is_distributed:
                        # where we will all_gather results from all workers
                        tensors_list = []
                        # where we will all_gather tensor sizes
                        tensor_on_worker = registered_e_tensors[key]
                        if tensor_on_worker.shape != torch.Size([]):
                            tensor_on_worker_size_as_tensor = torch.tensor(
                                tensor_on_worker.shape
                            ).cuda()
                            sizes = []
                            for ind in range(world_size):
                                sizes.append(
                                    torch.empty_like(
                                        tensor_on_worker_size_as_tensor)
                                )
                            dist.all_gather(sizes,
                                            tensor_on_worker_size_as_tensor)
                            mx_dim, _ = torch.max(torch.stack(sizes), dim=0)
                        else:  # this is a singleton. For example, loss value
                            sizes = [torch.Size([])] * world_size
                            mx_dim = None
                        for ind in range(world_size):
                            # we have to use max shape for all_gather
                            if mx_dim is None:  # singletons
                                tensors_list.append(
                                    torch.tensor(2).cuda().type_as(
                                        tensor_on_worker)
                                )
                            else:  # non-singletons
                                tensors_list.append(torch.empty(
                                    mx_dim.cpu().data.numpy().tolist()).cuda()
                                    .type_as(
                                    tensor_on_worker))

                        if mx_dim is not None:
                            t_to_send = self.pad_tensor(tensor_on_worker,
                                                        mx_dim)
                        else:
                            t_to_send = tensor_on_worker
                        dist.all_gather(tensors_list, t_to_send)
                        tensors_list = [
                            self.depad_tensor(t, size)
                            for t, size in zip(tensors_list, sizes)
                        ]
                        if self.local_rank == 0:
                            values_dict["IS_FROM_DIST_EVAL"] = True
                            values_dict[key] = tensors_list
                    else:  # NON-DISTRIBUTED TRAINING
                        values_dict["IS_FROM_DIST_EVAL"] = False
                        values_dict[key] = [registered_e_tensors[key]]
                if callback.user_iter_callback and (
                        self.local_rank is None or self.local_rank == 0
                ):
                    # values_dict will contain results from all workers
                    callback.user_iter_callback(values_dict,
                                                callback._global_var_dict)

            # final aggregation (over minibatches) and logging of results
            # should happend only on one worker
            if callback.user_done_callback and (
                    self.local_rank is None or self.local_rank == 0
            ):
                vals_to_log = callback.user_done_callback(
                    callback._global_var_dict)
                # log results to Tensorboard
                if vals_to_log is not None and callback._swriter is not None:
                    for key, val in vals_to_log.items():
                        callback._swriter.add_scalar(key, val, step)

    def _infer(self, tensors_to_return, step, verbose=False):
        """
        Does the same as _eval() just with tensors instead of eval callback.
        """
        with torch.no_grad():
            # each call chain corresponds to a tensor in tensors_2_evaluate
            dl_nm = None
            call_chain, _ = self.__get_top_sorted_modules_and_dataloader(
                hook=tensors_to_return
            )
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
                # print(
                #     "Doing distributed evaluation. Rank {0} of {1}".format(
                #         self.local_rank, world_size
                #     )
                # )
                if dl_nm.dataset is not None:
                    sampler = torch.utils.data.distributed.DistributedSampler(
                        dl_nm.dataset
                    )
                    eval_dataloader = torch.utils.data.DataLoader(
                        dataset=dl_nm.dataset,
                        sampler=sampler,
                        num_workers=dl_nm.local_parameters.get(
                            "num_workers", os.cpu_count()
                        ),
                        batch_size=dl_nm.local_parameters["batch_size"],
                        shuffle=(sampler is None),
                    )
                else:
                    eval_dataloader = dl_nm.data_iterator
                eval_dataloader.sampler.set_epoch(0)
            else:  # Not distributed
                if dl_nm.dataset is not None:
                    # Todo: remove local_parameters
                    eval_dataloader = torch.utils.data.DataLoader(
                        dataset=dl_nm.dataset,
                        sampler=None,  # not distributed sampler
                        num_workers=call_chain[0][0].local_parameters.get(
                            "num_workers", os.cpu_count()
                        ),
                        batch_size=call_chain[0][0].local_parameters[
                            "batch_size"],
                        shuffle=call_chain[0][0].local_parameters.get(
                            "shuffle",
                            False),
                    )
                else:
                    eval_dataloader = dl_nm.data_iterator
            # after this eval_dataloader is ready to be used
            # reset global_var_dict - results of evaluation will be stored
            # there

            if not is_distributed or self.local_rank == 0:
                values_dict = {}
                for t in tensors_to_return:
                    values_dict[t.unique_name] = []
            dl_device = dl_nm._device

            # Evaluation mini-batch for loop
            num_batches = len(eval_dataloader)
            for epoch_i, data in enumerate(eval_dataloader, 0):
                if verbose and (
                        num_batches < 10 or (
                        epoch_i % int(num_batches / 10) == 0)
                ):
                    print(
                        f"Evaluating batch {epoch_i} out of {num_batches}")
                tensors = []
                if isinstance(data, torch.Tensor):
                    data = (data,)
                for d in data:
                    if isinstance(d, torch.Tensor):
                        tensors.append(d.to(dl_device))
                    else:
                        tensors.append(d)

                registered_e_tensors = {t.unique_name: d for t, d in
                                        zip(call_chain[0][2].values(), tensors)
                                        if t is not None
                                        }
                self.__nm_graph_forward_pass(
                    call_chain=call_chain,
                    registered_tensors=registered_e_tensors,
                    mode=ModelMode.eval,
                )

                # If distributed. For the outer loop, we need to ensure that
                # all processes loop through the elements in the same order
                for t2e in tensors_to_return:
                    key = t2e.unique_name
                    if key not in registered_e_tensors.keys():
                        print(
                            "WARNING: Tensor {} was not found during "
                            "eval".format(
                                key)
                        )
                        continue
                    if is_distributed:
                        # where we will all_gather results from all workers
                        tensors_list = []
                        # where we will all_gather tensor sizes
                        tensor_on_worker = registered_e_tensors[key]
                        if tensor_on_worker.shape != torch.Size([]):
                            tensor_on_worker_size_as_tensor = torch.tensor(
                                tensor_on_worker.shape
                            ).cuda()
                            sizes = []
                            for ind in range(world_size):
                                sizes.append(
                                    torch.empty_like(
                                        tensor_on_worker_size_as_tensor)
                                )
                            dist.all_gather(sizes,
                                            tensor_on_worker_size_as_tensor)
                            mx_dim, _ = torch.max(torch.stack(sizes), dim=0)
                        else:  # this is a singleton. For example, loss value
                            sizes = [torch.Size([])] * world_size
                            mx_dim = None
                        for ind in range(world_size):
                            # we have to use max shape for all_gather
                            if mx_dim is None:  # singletons
                                tensors_list.append(
                                    torch.tensor(2).cuda().type_as(
                                        tensor_on_worker)
                                )
                            else:  # non-singletons
                                tensors_list.append(torch.empty(
                                    mx_dim.cpu().data.numpy().tolist()).cuda()
                                    .type_as(
                                    tensor_on_worker))

                        if mx_dim is not None:
                            t_to_send = self.pad_tensor(tensor_on_worker,
                                                        mx_dim)
                        else:
                            t_to_send = tensor_on_worker
                        dist.all_gather(tensors_list, t_to_send)
                        tensors_list = [
                            self.depad_tensor(t, size)
                            for t, size in zip(tensors_list, sizes)
                        ]
                        if self.local_rank == 0:
                            values_dict[key] += tensors_list
                    else:  # NON-DISTRIBUTED TRAINING
                        values_dict[key] += [registered_e_tensors[key]]

            if not is_distributed or self.local_rank == 0:
                inferred_tensors = []
                for t in tensors_to_return:
                    inferred_tensors.append(values_dict[t.unique_name])
                return inferred_tensors

            # For all other ranks
            return None

    def save_state_to(self, path: str):
        """
        Saves current state such as step, epoch and optimizer parameters
        Args:
          path:

        Returns:

        """
        state = {
            "step": self.step,
            "epoch_num": self.epoch_num,
            "optimizer_state": [opt.state_dict() for opt in self.optimizers],
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
            # general since we are also saving step and epoch_num
            # load_state_dict should move the variables to the relevant device
            checkpoint = torch.load(path, map_location="cpu")
            self.step = checkpoint["step"]
            self.epoch_num = checkpoint["epoch_num"]
            if checkpoint["optimizer_state"]:
                for opt, opt_chkpt in zip(
                        self.optimizers, checkpoint["optimizer_state"]):
                    opt.load_state_dict(opt_chkpt)
        else:
            raise FileNotFoundError(
                "Could not find checkpoint file: {0}".format(path))

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
            if not (isinstance(tup[0], torch.optim.Optimizer)
                    and PtActions._check_all_tensors(tup[1])):
                return False
        return True

    def _get_all_modules(
            self, training_loop, callbacks, logging_callchain=None):
        """Gets all neural modules that will be used by train() and eval() via
        EvaluatorCallbacks. Saves all modules to self.modules
        """
        # If there is a SimpleLossLoggerCallback, create an logger_callchain
        # with all callchains from training_loop and
        # SimpleLossLoggerCallback.tensors
        if logging_callchain:
            for module in logging_callchain:
                self.modules.add(module[0])

        # Else grab all callchains from training_loop
        else:
            for step in training_loop:
                for module in step[2]:
                    self.modules.add(module[0])

        # Lastly, grab all eval modules
        if callbacks is not None:
            for callback in callbacks:
                if isinstance(callback, EvaluatorCallback):
                    callchain, _ = \
                        self.__get_top_sorted_modules_and_dataloader(
                            hook=callback.eval_tensors)
                    for module in callchain:
                        self.modules.add(module[0])

    def train(
            self,
            tensors_to_optimize,
            optimizer=None,
            optimization_params=None,
            callbacks: Optional[List[ActionCallback]] = None,
            lr_policy=None,
            batches_per_step=None,
            stop_on_nan_loss=False
    ):
        if not optimization_params:
            optimization_params = {}
        num_epochs = optimization_params.get("num_epochs", 1)
        max_steps = optimization_params.get("max_steps", None)
        grad_norm_clip = optimization_params.get('grad_norm_clip', None)

        if batches_per_step is None:
            batches_per_step = 1
        # this is necessary because we average gradients over batch
        bps_scale = torch.FloatTensor([1.0 / batches_per_step])

        if tensors_to_optimize is None:
            # This is Evaluation Mode
            self._init_callbacks(callbacks)
            # Do action start callbacks
            self._perform_on_action_end(callbacks=callbacks)
            return
        # Check if tensors_to_optimize is just a list of NmTensors
        elif tensors_to_optimize is not None and (
                isinstance(tensors_to_optimize[0],
                           NmTensor) and PtActions._check_all_tensors(
                tensors_to_optimize)):
            # Parse graph into a topologically sorted sequence of neural
            # modules' calls
            opt_call_chain, t_dataset = \
                self.__get_top_sorted_modules_and_dataloader(
                    hook=tensors_to_optimize
                )

            # Extract trainable weights which will be optimized
            params_list = [
                p[0].parameters() for p in opt_call_chain
                if isinstance(p[0], TrainableNM) or p[0].is_trainable()
            ]
            params_to_optimize = itertools.chain(*params_list)

            # Setup optimizer instance. By default it is SGD
            optimizer_instance = None
            optimizer_class = None
            if isinstance(optimizer, str):
                optimizer_class = optimizer
            elif isinstance(optimizer, torch.optim.optimizer):
                optimizer_instance = optimizer
            else:
                raise ValueError("optimizer was not understood")
            optimizer = self.__setup_optimizer(
                optimizer_instance=optimizer_instance,
                optimizer_class=optimizer_class,
                optimization_params=optimization_params,
                params_to_optimize=params_to_optimize,
            )

            training_loop = [
                (optimizer, tensors_to_optimize, opt_call_chain)
            ]

            self.optimizers.append(optimizer)
            assert len(self.optimizers) == 1, \
                ("There was more than one optimizer, was create_optimizer() "
                 "called before train()?")

        elif PtActions._check_tuples(tensors_to_optimize):
            if batches_per_step != 1:
                raise ValueError("Gradient accumlation with multiple "
                                 "optimizers is not supported")
            datasets = []
            training_loop = []
            for step in tensors_to_optimize:
                step_call_chain, dataset = \
                    self.__get_top_sorted_modules_and_dataloader(
                        hook=step[1]
                    )
                datasets.append(dataset)
                training_loop.append(
                    (step[0], step[1], step_call_chain))

            t_dataset = datasets[0]
            for dataset in datasets:
                if type(dataset) is not type(t_dataset):
                    raise ValueError(
                        "There were two training datasets, we only support 1.")
        else:
            raise ValueError("tensors_to_optimize was not understood")

        logging_callchain = None
        # callbacks setup
        if callbacks is not None:
            for callback in callbacks:
                if not isinstance(callback, ActionCallback):
                    raise ValueError("A callback was received that was not a "
                                     "child of ActionCallback")
                elif isinstance(callback, SimpleLossLoggerCallback):
                    if logging_callchain:
                        raise ValueError("We only support one logger callback "
                                         "but more than one were found")
                    logger_step_freq = callback._step_freq
                    logging_tensors = callback.tensors
                    all_tensors = logging_tensors
                    for step in training_loop:
                        all_tensors = all_tensors + step[1]
                    logging_callchain, _ = \
                        self.__get_top_sorted_modules_and_dataloader(
                            hook=all_tensors)

        self._get_all_modules(training_loop, callbacks, logging_callchain)

        # Intialize Amp if needed
        if self._optim_level in AmpOptimizations:
            # Store mapping of self.optimizers to optimizer in callchain
            training_loop_opts = []
            for opt in training_loop:
                training_loop_opts.append(self.optimizers.index(opt[0]))
            self.optimizers = self.__initialize_amp(
                optimizer=self.optimizers,
                optim_level=self._optim_level,
                amp_min_loss_scale=optimization_params.get(
                    'amp_min_loss_scale', 1.0))
            # Use stored mapping to map amp_init opts to training loop
            for i, step in enumerate(training_loop):
                training_loop[i] = (
                    self.optimizers[training_loop_opts[i]], step[1], step[2])

        dataNM = training_loop[0][2][0][0]
        if dataNM.placement == DeviceType.AllGpu:
            # if len(training_loop) > 1:
            #     raise NotImplementedError(
            #         "Distributed training does nor work with multiple "
            #         "optimizers")
            print("Doing distributed training")
            if t_dataset is not None:
                train_sampler = \
                    torch.utils.data.distributed.DistributedSampler(
                        t_dataset
                    )
                train_dataloader = torch.utils.data.DataLoader(
                    dataset=t_dataset,
                    sampler=train_sampler,
                    num_workers=dataNM.local_parameters.get(
                        "num_workers", os.cpu_count()
                    ),
                    batch_size=dataNM.local_parameters["batch_size"],
                    shuffle=(train_sampler is None),
                )
            else:
                train_dataloader = dataNM.data_iterator
                train_sampler = train_dataloader.sampler

            for train_iter in training_loop:
                call_chain = train_iter[2]
                for i in range(1, len(call_chain) - 1):
                    key = call_chain[i][0].unique_instance_id
                    pmodule = self.module_reference_table[key][1]
                    if (not isinstance(pmodule, DDP) and
                            isinstance(pmodule, torch.nn.Module)):
                        pmodule = DDP(pmodule)
                    self.module_reference_table[key] = (
                        self.module_reference_table[key][0], pmodule
                    )
        # single GPU/CPU training
        else:
            if t_dataset is not None:
                train_sampler = None
                train_dataloader = torch.utils.data.DataLoader(
                    dataset=t_dataset,
                    sampler=None,
                    num_workers=dataNM.local_parameters.get(
                        "num_workers", os.cpu_count()
                    ),
                    batch_size=dataNM.local_parameters["batch_size"],
                    shuffle=dataNM.local_parameters.get("shuffle", True),
                )
            else:
                train_dataloader = dataNM.data_iterator
                train_sampler = None

        self._init_callbacks(callbacks)
        # Do action start callbacks
        self._perform_on_action_start(callbacks=callbacks)

        # MAIN TRAINING LOOP
        # iteration over epochs
        for epoch_ind in range(self.epoch_num, num_epochs):
            self.epoch_num = epoch_ind
            if train_sampler is not None:
                train_sampler.set_epoch(self.epoch_num)
            if max_steps is not None and self.step >= max_steps:
                break

            # Register epochs start with callbacks
            self._perform_on_epoch_start(callbacks=callbacks)

            # iteration over batches in epoch
            batch_counter = 0
            for _, data in enumerate(train_dataloader, 0):
                if max_steps is not None and self.step >= max_steps:
                    break

                if batch_counter == 0:
                    # Started step, zero gradients
                    curr_optimizer = training_loop[
                        self.step % len(training_loop)][0]
                    curr_optimizer.zero_grad()
                    # Register iteration start with callbacks
                    self._perform_on_iteration_start(callbacks=callbacks)

                # set learning rate policy
                if lr_policy is not None:
                    adjusted_lr = lr_policy(
                        optimization_params["lr"], self.step, self.epoch_num
                    )
                    for param_group in curr_optimizer.param_groups:
                        param_group["lr"] = adjusted_lr
                if self.tb_writer is not None:
                    value = curr_optimizer.param_groups[0]['lr']
                    self.tb_writer.add_scalar('param/lr', value, self.step)

                # registered_tensors will contain created tensors
                # named by output port and uuid of module which created them
                # Get and properly name tensors returned by data layer
                curr_call_chain = training_loop[
                    self.step % len(training_loop)][2]
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

                registered_tensors = {
                    t.unique_name: d for t, d in
                    zip(curr_call_chain[0][2].values(), tensors)
                    if t is not None
                }
                disable_allreduce = batch_counter < (batches_per_step - 1)
                self.__nm_graph_forward_pass(
                    call_chain=curr_call_chain,
                    registered_tensors=registered_tensors,
                    disable_allreduce=disable_allreduce
                )

                curr_tensors_to_optimize = training_loop[
                    self.step % len(training_loop)][1]
                final_loss = 0
                nan = False
                for tensor in curr_tensors_to_optimize:
                    if torch.isnan(
                            registered_tensors[tensor.unique_name]).any():
                        if stop_on_nan_loss:
                            raise ValueError('Loss is NaN exiting')
                        print('WARNING: Loss is NaN')
                        curr_optimizer.zero_grad()
                        nan = True
                        break
                    final_loss += registered_tensors[tensor.unique_name]
                if nan:
                    continue
                if self._optim_level in AmpOptimizations:
                    with amp.scale_loss(
                            final_loss,
                            curr_optimizer,
                            delay_unscale=disable_allreduce
                    ) as scaled_loss:
                        if torch.isnan(scaled_loss).any():
                            if stop_on_nan_loss:
                                raise ValueError('Loss is NaN exiting')
                            print('WARNING: Loss is NaN')
                            curr_optimizer.zero_grad()
                            continue
                        scaled_loss.backward(
                            bps_scale.to(scaled_loss.get_device()))
                else:
                    final_loss.backward(
                        bps_scale.to(
                            final_loss.get_device()))

                batch_counter += 1

                if batch_counter == batches_per_step:
                    # Ended step. Do optimizer update
                    if grad_norm_clip is not None:
                        torch.nn.utils.clip_grad_norm_(
                            amp.master_params(curr_optimizer), grad_norm_clip)
                    curr_optimizer.step()
                    batch_counter = 0
                    # Register iteration end with callbacks
                    self._update_callbacks(
                        callbacks=callbacks,
                        registered_tensors=registered_tensors,
                    )
                    self._perform_on_iteration_end(callbacks=callbacks)
                    self.step += 1
            # End of epoch for loop

            # Register epochs end with callbacks
            self._perform_on_epoch_end(callbacks=callbacks)
        self._perform_on_action_end(callbacks=callbacks)

    def infer(self, tensors, checkpoint_dir=None, ckpt_pattern='',
              logger=None):

        if checkpoint_dir:
            # Find all modules that need to be restored
            call_chain, _ = self.__get_top_sorted_modules_and_dataloader(
                hook=tensors
            )
            modules_to_restore = []
            modules_to_restore_name = []
            for op in call_chain:
                if op[0].num_weights > 0:
                    modules_to_restore.append(op[0])
                    modules_to_restore_name.append(op[0].__class__.__name__)

            module_checkpoints = get_checkpoint_from_dir(
                modules_to_restore_name, checkpoint_dir, ckpt_pattern
            )

            for mod, checkpoint in zip(modules_to_restore, module_checkpoints):
                if logger:
                    logger.info(f"Restoring {mod} from {checkpoint}")
                mod.restore_from(checkpoint, self._local_rank)

            # Init Amp
            if self._optim_level in AmpOptimizations:
                pt_modules = []
                for i in range(len(call_chain)):
                    if isinstance(call_chain[i][0], nn.Module):
                        pt_modules.append(call_chain[i][0])
                    elif isinstance(call_chain[i][0],
                                    TrainableNeuralModuleWrapper):
                        pt_modules.append(call_chain[i][0]._pt_module)

                amp.initialize(
                    min_loss_scale=1.0,
                    models=pt_modules,
                    optimizers=None,
                    opt_level=AmpOptimizations[self._optim_level],
                )

        # Run infer
        return self._infer(tensors_to_return=tensors, step=0, verbose=True)
