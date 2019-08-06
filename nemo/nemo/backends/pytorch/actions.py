# Copyright (c) 2019 NVIDIA Corporation
import itertools
import os
import logging
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim

from typing import List, Optional, Dict, Set
from .module_wrapper import TrainableNeuralModuleWrapper
from .nm import DataLayerNM
from .optimizers import Novograd, AdamW, Lamb
from ...core import NmTensor, DeviceType
from ...core.callbacks import (
    ActionCallback,
    EvaluatorCallback,
    SimpleLossLoggerCallback,
    ModuleSaverCallback,
    CheckpointCallback,
)
from ...core.neural_factory import Actions, ModelMode, Optimization
from ...utils.helpers import get_latest_checkpoint_from_dir
from nemo.core.callbacks import ValueSetterCallback

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

_float_2_half_req = {Optimization.mxprO1, Optimization.mxprO2,
                     Optimization.mxprO3}


def _add_uuid_2_name(name, uuid):
    return name + "~~~" + uuid


def _remove_uuid_from_name(name):
    return name[: name.index("~~~")]


def _filter_dict(d: Dict, keys: Set) -> Set:
    res = {}
    for k, v in d.items():
        if k in keys:
            res[_remove_uuid_from_name(k)] = v
    return res


class PtActions(Actions):
    def __init__(self, params, local_rank=None, tb_writer=None):
        super(PtActions, self).__init__(params=params, local_rank=local_rank)

        # will be [unique_instance_id -> (NMModule, PTModule)]
        self.module_reference_table = {}
        self.step = 0
        self.epoch_num = 0
        self.optimizer = None
        self.tb_writer = tb_writer

    def __get_top_sorted_modules_and_dataloader(self, hook):
        """
        Constracts DAG leading to hook and creates its topological order.
        It also populates self.module_reference_table.
        Args:
          hook: an NmTensor or a list of NmTensors representing leaf nodes
          in DAG

        Returns:
          list of modules with their call arguments and dataset
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

        _top_sorted_modules = []
        all_nodes = set()

        # extract all nodes to all_nodes set
        hooks_lst = list(hooks)
        while len(hooks_lst) > 0:
            # take hook from the end of the list
            hook = hooks_lst.pop()
            node = create_node(hook.producer, hook.producer_args)
            all_nodes.add(node)
            if hook.producer_args is not None and hook.producer_args != {}:
                for _, nmtensor in hook.producer_args.items():
                    hooks_lst.insert(0, nmtensor)

        while len(all_nodes) > 0:
            for node in all_nodes.copy():
                # if node's in_degree is zero it can be added to
                # _top_sorted_modules
                # this will also reduce in_degree of its children
                if is_in_degree_zero(node, _top_sorted_modules):
                    _top_sorted_modules.append(node)
                    all_nodes.remove(node)

        tdataset = _top_sorted_modules[0][0].dataset
        top_sorted_modules = [(m[0], dict(m[1])) for m in _top_sorted_modules]

        # populate self.module_reference_table
        for m in _top_sorted_modules:
            if m[0].factory is None and self._local_rank is not None:
                raise ValueError("Neural module {0} was created without "
                                 "NeuralModuleFactory, but you are trying to"
                                 "run in distributed mode. Please instantiate"
                                 "NeuralModuleFactory first and pass it's "
                                 "instance as `factory` parameter to all your"
                                 "Neural Module objects.")
            key = m[0].unique_instance_id
            if key not in self.module_reference_table:
                if isinstance(m[0], TrainableNeuralModuleWrapper):
                    self.module_reference_table[key] = (m[0], m[0]._pt_module)
                else:
                    self.module_reference_table[key] = (m[0], m[0])

        return top_sorted_modules, tdataset

    @staticmethod
    def __setup_optimizer(
            optimizer_instance,
            optimizer_class,
            optimization_params,
            params_to_optimize,
            call_chain,
            optim_level=Optimization.nothing,
    ):
        amp_min_loss_scale = 1.0
        if optimization_params is not None:
            amp_min_loss_scale = optimization_params.get('min_loss_scale', 1.0)
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
                optimizer = optim.Adam(params=params_to_optimize, lr=lr)
            elif optimizer_class.lower() == "fuzed_adam":
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

        if optim_level in AmpOptimizations:
            inds = []
            pt_modules = []
            for i in range(len(call_chain)):
                if isinstance(call_chain[i][0], nn.Module):
                    inds.append([i, False])
                    pt_modules.append(call_chain[i][0])
                elif isinstance(call_chain[i][0],
                                TrainableNeuralModuleWrapper):
                    inds.append([i, True])
                    pt_modules.append(call_chain[i][0]._pt_module)

            pt_modules, optimizer = amp.initialize(
                min_loss_scale=amp_min_loss_scale,
                max_loss_scale=32768.0,
                models=pt_modules,
                optimizers=optimizer,
                opt_level=AmpOptimizations[optim_level],
            )

            for ind in range(len(pt_modules)):
                if inds[ind][1]:
                    call_chain[inds[ind][0]][0]._pt_module = pt_modules[ind]
                else:
                    call_chain[inds[ind][0]] = (
                        pt_modules[ind],
                        call_chain[inds[ind][0]][1],
                    )
        else:
            return optimizer, call_chain
        return optimizer, call_chain

    def __nm_graph_forward_pass(
            self, call_chain, registered_tensors, mode=ModelMode.train
    ):
        for ind in range(1, len(call_chain)):
            call_args = call_chain[ind][1]
            # module = call_chain[ind][0]
            m_id = call_chain[ind][0].unique_instance_id
            pmodule = self.module_reference_table[m_id][1]
            module_output_port_names = call_chain[ind][0]._output_ports.keys()

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
            # module_output_port_names = module._output_ports.keys()
            # now pack it according module's output port names
            new_tensors_packed = dict(
                zip(
                    [
                        _add_uuid_2_name(port_name, m_id)
                        for port_name in module_output_port_names
                    ],
                    new_tensors,
                )
            )
            for t_name, t_tensor in new_tensors_packed.items():
                if t_name not in registered_tensors:
                    registered_tensors[t_name] = t_tensor

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
            padded_t[: t_size[0], : t_size[1], : t_size[2], : t.size[3]] = t
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

            if not isinstance(dl_nm, DataLayerNM):
                raise ValueError(
                    "The evaluation callchain did not start with a DataLayerNM"
                )

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
            data_layer_output_port_names = dl_nm._output_ports.keys()
            dl_device = dl_nm._device

            # Evaluation mini-batch for loop
            num_batches = len(eval_dataloader)
            for epoch_i, data in enumerate(eval_dataloader, 0):
                if verbose and (
                        num_batches < 10 or (
                        epoch_i % int(num_batches / 10) == 0)
                ):
                    print("Evaluating batch {} out of {}".format(epoch_i,
                                                                 num_batches))
                tensors = []
                for d in data:
                    if isinstance(d, torch.Tensor):
                        tensors.append(d.to(dl_device))
                    else:
                        tensors.append(d)

                registered_e_tensors = dict(
                    zip(
                        [
                            _add_uuid_2_name(dl_port_name,
                                             call_chain[0][0]._uuid)
                            for dl_port_name in data_layer_output_port_names
                        ],
                        tensors,
                    )
                )
                self.__nm_graph_forward_pass(
                    call_chain=call_chain,
                    registered_tensors=registered_e_tensors,
                    mode=ModelMode.eval,
                )

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
                        values_dict["IS_FROM_DIST_EVAL"] = True
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
            "optimizer_state": self.optimizer.state_dict() if self.optimizer
            else None,
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
            if checkpoint["optimizer_state"] is not None:
                self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        else:
            raise FileNotFoundError(
                "Could not find checkpoint file: {0}".format(path))

    def train(
            self,
            tensors_to_optimize: List[NmTensor],
            tensors_to_evaluate: Optional[List[NmTensor]] = None,
            callbacks: Optional[List[ActionCallback]] = None,
            lr_policy=None,
            batches_per_step=None,
            stop_on_nan_loss=False
    ):
        if len(tensors_to_optimize) != 1:
            raise NotImplementedError(
                "Currently we can only optimize single loss")
        if batches_per_step is None:
            batches_per_step = 1
        # this is necessary because we average gradients over batch
        bps_scale = torch.FloatTensor([1.0/batches_per_step])

        # Parse graph into a topologically sorted sequence of neural
        # modules' calls
        opt_call_chain, t_dataset = \
            self.__get_top_sorted_modules_and_dataloader(
                hook=tensors_to_optimize
            )
        opteval_call_chain = None
        if tensors_to_evaluate is not None:
            opteval_call_chain, _ = \
                self.__get_top_sorted_modules_and_dataloader(
                    hook=tensors_to_optimize + tensors_to_evaluate
                )

        # Extract trainable weights which will be optimized
        params_list = [p[0].parameters() for p in opt_call_chain if
                       p[0].is_trainable()]
        params_to_optimize = itertools.chain(*params_list)

        # Setup optimizer instance. By default it is SGD
        optimizer_instance = self._parameters.get("optimizer_instance", None)
        optimizer_class = self._parameters.get("optimizer_kind", "sgd")
        optimization_params = self._parameters.get(
            "optimization_params", {"lr": 0.0003}
        )
        grad_norm_clip = optimization_params.get('grad_norm_clip', None)
        num_epochs = optimization_params.get("num_epochs", 1)
        max_steps = optimization_params.get("max_steps", None)
        self.optimizer, opt_call_chain = self.__setup_optimizer(
            optimizer_instance=optimizer_instance,
            optimizer_class=optimizer_class,
            optimization_params=optimization_params,
            params_to_optimize=params_to_optimize,
            call_chain=opt_call_chain,
            optim_level=self._optim_level,
        )

        dataNM = opt_call_chain[0][0]
        if dataNM.placement == DeviceType.AllGpu:
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

            for i in range(1, len(opt_call_chain) - 1):
                key = opt_call_chain[i][0].unique_instance_id
                self.module_reference_table[key] = (
                    self.module_reference_table[key][0],
                    DDP(self.module_reference_table[key][1]) if isinstance(
                        self.module_reference_table[key][1],
                        torch.nn.Module) else self.module_reference_table[key][
                        1],
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

        data_layer_output_port_names = opt_call_chain[0][0]._output_ports\
            .keys()
        eval_tensors_debug_freq = 1000000000

        # callbacks setup
        if callbacks is not None:
            for callback in callbacks:
                if isinstance(callback, EvaluatorCallback):
                    callback.__setattr__("_compute_callback", self._eval)
                elif isinstance(callback, SimpleLossLoggerCallback):
                    eval_tensors_debug_freq = min(callback._step_frequency,
                                                  eval_tensors_debug_freq)
                elif isinstance(callback, CheckpointCallback):
                    callback.__setattr__("call_chain", opt_call_chain)
                    callback.__setattr__("action", self)
                elif isinstance(callback, (ModuleSaverCallback,
                                           ValueSetterCallback)):
                    pass
                else:
                    raise TypeError("Callback of unknown type")
        # Register action start with callbacks
        self._fill_callbacks(
            callbacks=callbacks,
            tensors_to_optimize=tensors_to_optimize,
            tensors_to_evaluate=tensors_to_evaluate,
        )
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
            self._fill_callbacks(
                callbacks=callbacks,
                tensors_to_optimize=tensors_to_optimize,
                tensors_to_evaluate=tensors_to_evaluate,
            )
            self._perform_on_epoch_start(callbacks=callbacks)

            # iteration over batches in epoch
            batch_counter = 0
            for epoch_i, data in enumerate(train_dataloader, 0):
                if max_steps is not None and self.step >= max_steps:
                    break

                if batch_counter == 0:
                    # Started step, zero gradients
                    self.optimizer.zero_grad()
                    # Register iteration start with callbacks
                    self._fill_callbacks(
                        callbacks=callbacks,
                        tensors_to_optimize=tensors_to_optimize,
                        tensors_to_evaluate=tensors_to_evaluate,
                    )
                    self._perform_on_iteration_start(callbacks=callbacks)

                # set learning rate policy
                if lr_policy is not None:
                    adjusted_lr = lr_policy(
                        optimization_params["lr"], self.step, self.epoch_num
                    )
                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = adjusted_lr
                if self.tb_writer is not None:
                    value = self.optimizer.param_groups[0]['lr']
                    self.tb_writer.add_scalar('param/lr', value, self.step)

                # registered_tensors will contain created tensors
                # named by output port and uuid of module which created them
                # Get and properly name tensors returned by data layer
                dl_device = opt_call_chain[0][0]._device
                tensors = []
                if isinstance(data, torch.Tensor):
                    data = (data,)
                for d in data:
                    if isinstance(d, torch.Tensor):
                        if self._optim_level in _float_2_half_req:
                            if isinstance(d, torch.FloatTensor) or isinstance(
                                    d, torch.cuda.FloatTensor
                            ):
                                tensors.append(d.to(dl_device).half())
                            else:
                                tensors.append(d.to(dl_device))
                        else:
                            tensors.append(d.to(dl_device))
                    else:
                        tensors.append(d)

                registered_tensors = dict(
                    zip(
                        [
                            _add_uuid_2_name(dl_port_name,
                                             opt_call_chain[0][0]._uuid)
                            for dl_port_name in data_layer_output_port_names
                        ],
                        tensors,
                    )
                )

                # Run opteval_call_chain as needed, otherwise run
                # opt_call_chain
                if (
                        self.step % eval_tensors_debug_freq == 0
                        and opteval_call_chain is not None
                ):
                    self.__nm_graph_forward_pass(
                        call_chain=opteval_call_chain,
                        registered_tensors=registered_tensors,
                    )
                else:
                    self.__nm_graph_forward_pass(
                        call_chain=opt_call_chain,
                        registered_tensors=registered_tensors
                    )

                tto_len = len(tensors_to_optimize)
                for ind in range(tto_len):
                    registered_name = tensors_to_optimize[ind].unique_name
                    if self._optim_level in AmpOptimizations and ind == \
                            tto_len - 1:
                        with amp.scale_loss(
                                registered_tensors[registered_name],
                                self.optimizer
                        ) as scaled_loss:
                            if torch.isnan(scaled_loss).any():
                                if stop_on_nan_loss:
                                    raise ValueError('Loss is NaN exiting')
                                else:
                                    print('WARNING: Loss is NaN')
                                    self.optimizer.zero_grad()
                            scaled_loss.backward(
                                bps_scale.to(scaled_loss.get_device()))
                    else:
                        if torch.isnan(registered_tensors[registered_name])\
                                .any():
                            if stop_on_nan_loss:
                                raise ValueError('Loss is NaN exiting')
                            else:
                                print('WARNING: Loss is NaN')
                                self.optimizer.zero_grad()

                        registered_tensors[registered_name].backward(
                            bps_scale.to(registered_tensors[
                                             registered_name].get_device()))

                batch_counter += 1

                if batch_counter == batches_per_step:
                    # Ended step. Do optimizer update
                    if grad_norm_clip is not None:
                        torch.nn.utils.clip_grad_norm_(
                            amp.master_params(self.optimizer), grad_norm_clip)
                    self.optimizer.step()
                    batch_counter = 0
                    # Register iteration end with callbacks
                    self._fill_callbacks(
                        callbacks=callbacks,
                        tensors_to_optimize=tensors_to_optimize,
                        tensors_to_evaluate=tensors_to_evaluate,
                        registered_tensors=registered_tensors,
                    )
                    self._perform_on_iteration_end(callbacks=callbacks)
                    self.step += 1
            # End of epoch for loop

            # Register epochs end with callbacks
            self._fill_callbacks(
                callbacks=callbacks,
                tensors_to_optimize=tensors_to_optimize,
                tensors_to_evaluate=tensors_to_evaluate,
            )
            self._perform_on_epoch_end(callbacks=callbacks)
        self._fill_callbacks(
            callbacks=callbacks,
            tensors_to_optimize=tensors_to_optimize,
            tensors_to_evaluate=tensors_to_evaluate,
        )
        self._perform_on_action_end(callbacks=callbacks)

    def infer(self, callback, checkpoint_dir=None, step_to_restore_from=None):

        if checkpoint_dir:
            # Find all modules that need to be restored
            call_chain, _ = self.__get_top_sorted_modules_and_dataloader(
                hook=callback.eval_tensors
            )
            modules_to_restore = []
            modules_to_restore_name = []
            for op in call_chain:
                if op[0].num_weights > 0:
                    modules_to_restore.append(op[0])
                    modules_to_restore_name.append(op[0].__class__.__name__)

            module_checkpoints = get_latest_checkpoint_from_dir(
                modules_to_restore_name, checkpoint_dir, step_to_restore_from
            )

            for mod, checkpoint in zip(modules_to_restore, module_checkpoints):
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
                    max_loss_scale=8192.0,
                    models=pt_modules,
                    optimizers=None,
                    opt_level=AmpOptimizations[self._optim_level],
                )

        # Run infer
        self._eval(callback.eval_tensors, callback, step=0, verbose=True)

        evaluated_tensors = []
        for tensor in callback.eval_tensors:
            evaluated_tensors.append(
                callback._global_var_dict[tensor.unique_name])
        return evaluated_tensors
