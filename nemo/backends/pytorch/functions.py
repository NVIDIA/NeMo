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
import itertools

import torch
import nemo
from typing import List, Dict

from nemo.backends.pytorch import TrainableNM, DataLayerNM
from nemo.backends.pytorch.module_wrapper import TrainableNeuralModuleWrapper
from nemo.core import NmTensor
from nemo.core.neural_factory import Actions, OperationMode, Optimization
from nemo.backends.pytorch.optimizers import AdamW, Novograd, master_params
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.optim as optim

# these imports will happen on as-needed basis
amp = None
# convert_syncbn = None
# create_syncbn_process_group = None
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


logging = nemo.logging


def __perform_on_iteration_start(callbacks):
    if callbacks is not None and isinstance(callbacks, List) and len(callbacks) > 0:
        for callback in callbacks:
            callback.on_iteration_start()


def __perform_on_iteration_end(callbacks):
    if callbacks is not None and isinstance(callbacks, List) and len(callbacks) > 0:
        for callback in callbacks:
            callback.on_iteration_end()


def __perform_on_action_start(callbacks):
    if callbacks is not None and isinstance(callbacks, List) and len(callbacks) > 0:
        for callback in callbacks:
            callback.on_action_start()


def __perform_on_action_end(callbacks):
    if callbacks is not None and isinstance(callbacks, List) and len(callbacks) > 0:
        for callback in callbacks:
            callback.on_action_end()


def __perform_on_epoch_start(callbacks):
    if callbacks is not None and isinstance(callbacks, List) and len(callbacks) > 0:
        for callback in callbacks:
            callback.on_epoch_start()


def __perform_on_epoch_end(callbacks):
    if callbacks is not None and isinstance(callbacks, List) and len(callbacks) > 0:
        for callback in callbacks:
            callback.on_epoch_end()


def __update_callbacks(callbacks=None, registered_tensors=None,):
    if callbacks is not None and isinstance(callbacks, List) and len(callbacks) > 0:
        for callback in callbacks:
            callback._registered_tensors = registered_tensors


def __check_all_tensors(list_of_tensors):
    """Method that checks if the passed list contains all NmTensors
    """
    if not isinstance(list_of_tensors, list):
        return False
    for tensor in list_of_tensors:
        if not isinstance(tensor, NmTensor):
            return False
    return True


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
                "Ignoring `optimization_params` parameter for "
                "optimizer because `optimizer_instance` is provided"
            )
        optimizer = optimizer_instance
    return optimizer


def __get_top_sorted_modules_and_dataloader(hook, module_reference_table: Dict, local_rank: int):
    """
    Constructs DAG leading to hook and creates its topological order.
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
            return tuple((producer, tuple([(k, v) for k, v in producer_args.items()]),))

    def is_in_degree_zero(node, processed_nodes):
        """A node has in degree of zero"""
        if node[1] == ():
            return True
        for portname, nmtensor in node[1]:
            nd = create_node(nmtensor.producer, nmtensor.producer_args)
            if nd not in processed_nodes:
                return False
        return True

    hooks = hook if isinstance(hook, list) else [hook]

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
            all_nodes[node] = {k: None for k in nmtensor.producer.output_ports}
        # second, populate output port with current nmtensor
        # where applicable
        all_nodes[node][nmtensor.name] = nmtensor
        processed_nmtensors.add(nmtensor)
        if nmtensor.producer_args is not None and nmtensor.producer_args != {}:
            for _, new_nmtensor in nmtensor.producer_args.items():
                if new_nmtensor not in processed_nmtensors:
                    # put in the start of list
                    hooks_lst.insert(0, new_nmtensor)

    all_node_with_output = []
    # Iterate over all_nodes to create new nodes that include its output
    # now all nodes have (module, input tensors, output tensors)
    for node in all_nodes:
        all_node_with_output.append(tuple((node[0], node[1], all_nodes[node])))

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
            raise ValueError("There were more than one DataLayer NeuralModule inside your DAG.")

    if not isinstance(top_sorted_modules[0][0], DataLayerNM):
        raise ValueError("The first module in your DAG was not a DataLayer NeuralModule.")

    tdataset = top_sorted_modules[0][0].dataset

    # populate self.module_reference_table
    for m in top_sorted_modules:
        if m[0].factory is None and local_rank is not None:
            raise ValueError(
                "Neural module {0} was created without "
                "NeuralModuleFactory, but you are trying to"
                "run in distributed mode. Please instantiate"
                "NeuralModuleFactory first and pass its "
                "instance as `factory` parameter to all your"
                "Neural Module objects."
                "".format(str(m[0]))
            )
        key = m[0].unique_instance_id
        if key not in module_reference_table:
            if isinstance(m[0], TrainableNeuralModuleWrapper):
                module_reference_table[key] = (m[0], m[0]._pt_module)
            else:
                module_reference_table[key] = (m[0], m[0])

    return top_sorted_modules, tdataset


def __check_tuples(list_of_tuples):
    """Method that checks if the passed tuple contains an optimizer in the
    first element, and a list of NmTensors in the second.
    """
    for tup in list_of_tuples:
        if not (isinstance(tup[0], torch.optim.Optimizer) and __check_all_tensors(tup[1])):
            return False
    return True


def pytorch_fit(tensors_to_optimize,
                training_graph,
                optimizer=None,
                optimization_params=None,
                callbacks: list = None,
                lr_policy=None,
                batches_per_step=None,
                stop_on_nan_loss=False,
                steps_per_nan_check=100,
                synced_batchnorm=False,
                synced_batchnorm_groupsize=0,
                gradient_predivide=False,
                amp_max_loss_scale=2.0 ** 24,
                reset=False):
    local_rank = 0 # TODO: Properly set this !!!!
    module_reference_table = {}
    step = 0
    epoch_num = 0
    optimizers = []
    tb_writer = None
    modules = set()
    cache = None
    amp_initialized = False

    # Analyse the arguments passed to train.
    if tensors_to_optimize is not None and training_graph is not None:
        raise ValueError("Cannot pass both `tensors_to_optimize` and `training_graph` to the train() function")
    # if tensors_to_optimize is None and training_graph is None:
    #    raise ValueError(
    #        "One of the `tensors_to_optimize` or `training_graph` values must be passed to the train() function"
    #    )
    # Finally, unify.
    if training_graph is not None:
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
        # Do action start callbacks
        __perform_on_action_end(callbacks=callbacks)
        return
    # Check if tensors_to_optimize is just a list of NmTensors
    elif tensors_to_optimize is not None and (
            isinstance(tensors_to_optimize[0], NmTensor) and __check_all_tensors(tensors_to_optimize)
    ):
        # Parse graph into a topologically sorted sequence of neural
        # modules' calls
        (opt_call_chain, t_dataset,) = __get_top_sorted_modules_and_dataloader(hook=tensors_to_optimize,
                                                                               module_reference_table=module_reference_table,
                                                                               local_rank=local_rank)
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
        optimizer = __setup_optimizer(
            optimizer_instance=optimizer_instance,
            optimizer_class=optimizer_class,
            optimization_params=optimization_params,
            params_to_optimize=params_to_optimize,
        )

        training_loop = [(optimizer, tensors_to_optimize, opt_call_chain)]

        optimizers.append(optimizer)
        assert len(optimizers) == 1, (
            "There was more than one optimizer, was create_optimizer() called before train()? Are you calling "
            "train() twice in one script, If so you need to call NeuralModuleFactory.reset_trainer() first."
        )

    elif __check_tuples(tensors_to_optimize):
        if batches_per_step != 1:
            raise ValueError("Gradient accumlation with multiple optimizers is not supported")
        datasets = []
        training_loop = []
        for step in tensors_to_optimize:
            (step_call_chain, dataset,) = __get_top_sorted_modules_and_dataloader(hook=step[1],
                                                                                  module_reference_table=module_reference_table,
                                                                                  local_rank=local_rank)
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
            if not isinstance(callback, ActionCallback):
                raise ValueError("A callback was received that was not a child of ActionCallback")
            elif isinstance(callback, SimpleLossLoggerCallback):
                if logging_callchain:
                    raise ValueError("We only support one logger callback but more than one were found")
                logger_step_freq = callback._step_freq
                logging_tensors = callback.tensors
                all_tensors = logging_tensors
                for step in training_loop:
                    all_tensors = all_tensors + step[1]
                (logging_callchain, _,) = self.__get_top_sorted_modules_and_dataloader(hook=all_tensors)

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
            amp_max_loss_scale=amp_max_loss_scale,
            amp_min_loss_scale=optimization_params.get('amp_min_loss_scale', 1.0),
        )
        # Use stored mapping to map amp_init opts to training loop
        for i, step in enumerate(training_loop):
            training_loop[i] = (
                self.optimizers[training_loop_opts[i]],
                step[1],
                step[2],
            )

    dataNM = training_loop[0][2][0][0]
    placement_gpu = dataNM.placement == DeviceType.AllGpu
    if placement_gpu:
        # if len(training_loop) > 1:
        #     raise NotImplementedError(
        #         "Distributed training does nor work with multiple "
        #         "optimizers")
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

        for train_iter in training_loop:
            call_chain = train_iter[2]
            for i in range(1, len(call_chain) - 1):
                key = call_chain[i][0].unique_instance_id
                pmodule = module_reference_table[key][1]
                num_trainable_weights = module_reference_table[key][1].num_weights
                if (
                        not isinstance(pmodule, DDP)
                        and isinstance(pmodule, torch.nn.Module)
                        and num_trainable_weights > 0
                ):
                    # gpf = 1
                    # if gradient_predivide:
                    #     gpf = dist.get_world_size()
                    # pmodule = DDP(pmodule, gradient_predivide_factor=gpf)  # Old Apex Method

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

                        pmodule = nn.SyncBatchNorm.convert_sync_batchnorm(
                            pmodule, process_group=sync_batchnorm_group
                        )

                    # By default, disable broadcast_buffers. This disables batch norm synchronization on forward
                    # pass
                    pmodule = DDP(
                        pmodule, device_ids=[self.local_rank], broadcast_buffers=False, find_unused_parameters=True
                    )

                # # Convert batchnorm modules to synced if applicable
                # if synced_batchnorm and isinstance(pmodule, torch.nn.Module):
                #     world_size = dist.get_world_size()
                #     if synced_batchnorm_groupsize > 0 and world_size % synced_batchnorm_groupsize != 0:
                #         raise ValueError(
                #             f"Synchronized batch norm group size"
                #             f" ({synced_batchnorm_groupsize}) must be 0"
                #             f" or divide total number of GPUs"
                #             f" ({world_size})."
                #         )
                #     process_group = create_syncbn_process_group(synced_batchnorm_groupsize)
                #     pmodule = convert_syncbn(pmodule, process_group=process_group)

                self.module_reference_table[key] = (
                    self.module_reference_table[key][0],
                    pmodule,
                )
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
            }
            if hasattr(dataNM, 'collate_fn'):
                dataloader_params['collate_fn'] = dataNM.collate_fn

            train_dataloader = torch.utils.data.DataLoader(**dataloader_params)
        else:
            train_dataloader = dataNM.data_iterator
            train_sampler = None

    self._init_callbacks(callbacks)
    # Do action start callbacks
    self._perform_on_action_start(callbacks=callbacks)

    nan_or_inf = False

    # MAIN TRAINING LOOP
    # iteration over epochs
    while num_epochs is None or self.epoch_num < num_epochs:
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
                curr_optimizer = training_loop[self.step % len(training_loop)][0]
                curr_optimizer.zero_grad()
                # Register iteration start with callbacks
                self._perform_on_iteration_start(callbacks=callbacks)

            # set learning rate policy
            if lr_policy is not None:
                adjusted_lr = lr_policy(optimization_params["lr"], self.step, self.epoch_num)
                for param_group in curr_optimizer.param_groups:
                    param_group["lr"] = adjusted_lr
            if self.tb_writer is not None:
                value = curr_optimizer.param_groups[0]['lr']
                self.tb_writer.add_scalar('param/lr', value, self.step)
            if callbacks is not None:
                for callback in callbacks:
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

            registered_tensors = {
                t.unique_name: d for t, d in zip(curr_call_chain[0][2].values(), tensors) if t is not None
            }
            disable_allreduce = batch_counter < (batches_per_step - 1)
            self.__nm_graph_forward_pass(
                call_chain=curr_call_chain, registered_tensors=registered_tensors,
            )

            curr_tensors_to_optimize = training_loop[self.step % len(training_loop)][1]
            final_loss = 0
            for tensor in curr_tensors_to_optimize:
                if (
                        torch.isnan(registered_tensors[tensor.unique_name]).any()
                        or torch.isinf(registered_tensors[tensor.unique_name]).any()
                ):
                    if (
                            (stop_on_nan_loss)
                            or (self._optim_level not in AmpOptimizations)
                            or (self._optim_level == Optimization.mxprO0)
                    ):
                        # Set flag here and terminate at next all_gather check.
                        nan_or_inf = True
                        logging.warning(
                            'Loss is NaN or inf at step %d, will terminate within the'
                            ' next steps_per_nan_check steps',
                            self.step,
                        )
                    else:
                        logging.warning('Loss is NaN or inf, continuing training')
                final_loss += registered_tensors[tensor.unique_name]

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
                    # Fix (workaround?) enabling to backpropagate gradiens on CPUs.
                    if final_loss.get_device() < 0:
                        final_loss.backward(bps_scale)
                    else:
                        final_loss.backward(bps_scale.to(final_loss.get_device()))

            # Check if we should terminate due to NaN/inf on any workers.
            self._check_nan_or_inf(placement_gpu, nan_or_inf, steps_per_nan_check=steps_per_nan_check)

            batch_counter += 1

            if batch_counter == batches_per_step:
                # Ended step. Do optimizer update
                if grad_norm_clip is not None:
                    torch.nn.utils.clip_grad_norm_(master_params(curr_optimizer), grad_norm_clip)
                curr_optimizer.step()
                batch_counter = 0
                # Register iteration end with callbacks
                self._update_callbacks(
                    callbacks=callbacks, registered_tensors=registered_tensors,
                )
                self._perform_on_iteration_end(callbacks=callbacks)
                self.step += 1
        # End of epoch for loop
        # Register epochs end with callbacks
        self._perform_on_epoch_end(callbacks=callbacks)
        self.epoch_num += 1

    # Check again if we should stop on NaN/inf
    self._check_nan_or_inf(placement_gpu, nan_or_inf)

    self._perform_on_action_end(callbacks=callbacks)
