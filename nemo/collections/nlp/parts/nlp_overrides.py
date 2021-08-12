# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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

import os
from typing import Any, Dict, List, Optional, Union

import torch
from megatron import mpu
from megatron.checkpointing import get_checkpoint_version, set_checkpoint_version
from megatron.initialize import _set_random_seed
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning.overrides import LightningDistributedModule
from pytorch_lightning.plugins.environments.cluster_environment import ClusterEnvironment
from pytorch_lightning.plugins.training_type.ddp import DDPPlugin
from pytorch_lightning.trainer.connectors.checkpoint_connector import CheckpointConnector
from pytorch_lightning.utilities import rank_zero_warn
from pytorch_lightning.utilities.cloud_io import atomic_save
from torch.nn.parallel import DistributedDataParallel

from nemo.collections.nlp.modules.common.megatron.megatron_bert import MegatronBertEncoder
from nemo.collections.nlp.modules.common.megatron.megatron_encoder import MegatronEncoderModule
from nemo.utils import AppState, logging


class NLPDDPPlugin(DDPPlugin):
    """ DDP plugin for Pytorch Lightning. Needed to customize DDP for model parallel models.
    """

    distributed_backend = "ddp"

    def __init__(
        self,
        parallel_devices: Optional[List[torch.device]] = None,
        num_nodes: int = 1,
        cluster_environment: ClusterEnvironment = None,
        sync_batchnorm: bool = False,
        **kwargs: Union[Any, Dict[str, Any]],
    ) -> None:
        super().__init__(parallel_devices, num_nodes, cluster_environment, sync_batchnorm, **kwargs)

    def init_ddp_connection(self, global_rank: int = None, world_size: int = None) -> None:
        # call PTL init ddp
        super().init_ddp_connection()

        # init model parallel if needed
        app_state = AppState()

        if app_state.model_parallel_size is not None:

            if self.lightning_module.has_megatron_encoder and not self.lightning_module.is_model_parallel_initialized:
                self.init_model_parallel(app_state.global_rank, app_state.world_size)

    def start_training(self, trainer: 'Trainer') -> None:
        """ PTL Hook that is called after DPP is initialized. """

        if self.lightning_module.has_megatron_encoder:
            app_state = AppState()
            if app_state.model_parallel_size is not None:
                # mpu grad clipping needs parameters to have the attribute model_parallel
                parameters = self.lightning_module.parameters()
                for p in parameters:
                    if not hasattr(p, 'model_parallel'):
                        p.model_parallel = False

                if get_checkpoint_version() is not None:
                    # megatron checkpoint already restored
                    pass
                elif trainer.checkpoint_connector.resume_checkpoint_path is not None:
                    # PTL auto-resuming, need to update checkpoint name
                    # update path based on model parallel rank
                    filepath = trainer.checkpoint_connector.resume_checkpoint_path
                    dirname = os.path.dirname(os.path.dirname(filepath))
                    basename = os.path.basename(filepath)
                    filepath = f'{dirname}/mp_rank_{app_state.model_parallel_rank:02d}/{basename}'
                    trainer.checkpoint_connector.resume_checkpoint_path = filepath
                    logging.info(
                        f'Resuming training from checkpoint {trainer.checkpoint_connector.resume_checkpoint_path}'
                    )
                    # need to set checkpoint version for megatron-lm
                    checkpoint_version = torch.load(trainer.checkpoint_connector.resume_checkpoint_path).get(
                        'checkpoint_version', None
                    )
                    if checkpoint_version is not None:
                        set_checkpoint_version(checkpoint_version)
                    else:
                        logging.warning('Megatron-lm checkpoint version not found. Setting checkpoint_version to 0.')
                        set_checkpoint_version(0)
                else:
                    self.lightning_module.restore_megatron_encoder_weights()
            else:
                if get_checkpoint_version() is not None:
                    # megatron checkpoint already restored
                    pass
                else:
                    self.lightning_module.restore_megatron_encoder_weights()

            self.lightning_module.register_megatron_checkpoint_version()

        return super().start_training(trainer)

    def start_testing(self, trainer: 'Trainer') -> None:
        """ PTL Hook that is called after DPP is initialized. """
        app_state = AppState()

        if app_state.model_parallel_size is not None:

            if self.has_megatron_encoder:
                # check megatron checkpoint version
                checkpoint_version = get_checkpoint_version()
                if checkpoint_version is None:
                    raise ValueError("Unable to find megatron checkpoint version.")

        return super().start_testing(trainer)

    def configure_ddp(self):
        """ Override LightningModule ddp if using model parallel.
            Sets find_unused_parameters to True.
        """

        app_state = AppState()

        if app_state.model_parallel_size is not None:
            logging.info(f"Configuring DDP for model parallelism.")

            # With model parallelism, multiple GPUs form a large "logical GPU"
            # this means that data parallel groups span multiple GPUs
            # and are non-trivial
            device_ids = self.determine_ddp_device_ids()
            self._model = DistributedDataParallel(
                LightningDistributedModule(self.model),
                device_ids=device_ids,
                output_device=device_ids[0],
                process_group=app_state.data_parallel_group,
                find_unused_parameters=True,
                **self._ddp_kwargs,
            )

        else:
            super().configure_ddp()

    def init_model_parallel(self, global_rank: int, world_size: int) -> None:
        """ Initializes Megatron-LM model parallel if using model parallelism.

        Args:
            global_rank (int): the global process index.
            world_size (int): the total number of GPUs, num_nodes * num_gpus
            is_slurm_managing_tasks (bool, optional): is the cluster managed by SLURM.
        """
        app_state = AppState()

        # we initialize megatron-lm model parallel and data parallel groups
        # after initializing DDP with PTL.
        if app_state.model_parallel_size is not None:
            if torch.distributed.is_initialized():
                mpu.initialize_model_parallel(app_state.model_parallel_size)
                app_state.model_parallel_group = mpu.get_model_parallel_group()
                app_state.data_parallel_group = mpu.get_data_parallel_group()
                app_state.model_parallel_rank = mpu.get_tensor_model_parallel_rank()
                app_state.data_parallel_rank = mpu.get_data_parallel_rank()
                app_state.data_parallel_size = mpu.get_data_parallel_world_size()
                logging.info(f'mp_rank: {app_state.model_parallel_rank}')
                logging.info(f'dp_rank: {app_state.data_parallel_rank}')
                seed = os.environ.get("PL_GLOBAL_SEED", 1234)
                # random seed must be set for megatron model parallel init
                _set_random_seed(seed)

    @property
    def distributed_sampler_kwargs(self):
        app_state = AppState()
        if app_state.model_parallel_size is not None:
            # When using model parallel, data parallel groups are non-trivial and they
            # correspond to the logical GPUs. This means that the GPUs that form a
            # single logical GPU all need to get the same batch of data.
            distributed_sampler_kwargs = dict(
                num_replicas=app_state.data_parallel_size, rank=app_state.data_parallel_rank
            )
            return distributed_sampler_kwargs

        else:
            return super(NLPDDPPlugin, self).distributed_sampler_kwargs


class NLPCheckpointConnector(CheckpointConnector):
    """ Override PTL CheckpointConnector to support model parallel checkpoints from Megatron-LM.
    """

    def __init__(self, trainer):
        super().__init__(trainer)

    def save_checkpoint(self, filepath, weights_only: bool):
        """Slightly modified version of PyTorch Lightning's save_checkpoint.

        Args:
            filepath ([str]): [description]
            weights_only (bool): [description]

        Returns:
            [type]: [description]
        """
        app_state = AppState()
        if app_state.model_parallel_size is not None:
            # filepath needs to be updated to include mp_rank
            dirname = os.path.dirname(filepath)
            basename = os.path.basename(filepath)
            filepath = f'{dirname}/mp_rank_{app_state.model_parallel_rank:02d}/{basename}'

            # dump states as a checkpoint dictionary object
            checkpoint = self.dump_checkpoint(weights_only)

            # each model parallel rank needs to save a copy of its model
            if app_state.data_parallel_rank == 0:
                # write the checkpoint dictionary on the file
                if self.trainer.accelerator_backend:
                    checkpoint = self.trainer.accelerator_backend.on_save(checkpoint)
                try:
                    atomic_save(checkpoint, filepath)
                except AttributeError as err:
                    if LightningModule.CHECKPOINT_HYPER_PARAMS_KEY in checkpoint:
                        del checkpoint[LightningModule.CHECKPOINT_HYPER_PARAMS_KEY]
                    rank_zero_warn(
                        'Warning, `hyper_parameters` dropped from checkpoint.' f' An attribute is not picklable {err}'
                    )
                    atomic_save(checkpoint, filepath)
        return None
