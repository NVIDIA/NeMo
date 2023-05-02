# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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
import torch
import torch.distributed as dist
from pytorch_lightning import Callback


class WDSSampler:
    def __init__(self, mode):
        self.mode = mode
        assert self.mode in ['train', 'val']

    def set_epoch(self, epoch, pseudo_epoch=None, start_index=0):
        if self.mode == 'train':
            world_size = dist.get_world_size()
            num_samples_read_so_far = start_index * world_size
            os.environ["WDS_EPOCH_NUM"] = str(epoch)
            os.environ["WDS_START_INDEX"] = str(num_samples_read_so_far)
            print(f'set WDS_EPOCH_NUM={epoch}; WDS_START_INDEX={num_samples_read_so_far}; start_index={start_index}')
        else:
            pass


class WebDataloaderSamplerCallback(Callback):
    def __init__(self, batch_size, gradient_accumulation=1):
        super().__init__()
        self.batch_size = batch_size
        self.train_sampler = WDSSampler(mode='train')
        self.val_sampler = WDSSampler(mode='val')
        self.resume_flag = False
        self.ga = gradient_accumulation

    def on_train_epoch_start(self, trainer, pl_module):
        # For most cases, epoch should start from 0 (start_index = 0),
        # except for the case when we resume the checkpoint and start the epoch the first time
        if self.resume_flag:
            # We calculate the start_index by estimating the global steps / len(dataloader)
            num_iters = trainer.global_step % trainer.num_training_batches
            self.resume_flag = False
        else:
            num_iters = 0

        # We assume that the batch size, # GPUs between different runs remain the same
        # When ga is larger than 1, num_iters only records steps with back propagation
        # The actual consumed samples needs to multiply with ga batches
        consumed_samples_per_GPU = num_iters * self.batch_size * self.ga
        # This part assume that when we resume, we are using the same num of gpus and also same batchsize as before
        epoch = trainer.global_step * self.ga // trainer.num_training_batches
        print(
            f'WebdataLoaderSampler Calculated epoch={epoch}, num_iters={num_iters}, num_training_batches={trainer.num_training_batches}'
        )
        if pl_module.current_epoch != epoch:
            print(f'Warning: Calculated Epoch={epoch} is not equal to pyt-lightning epoch={pl_module.current_epoch}')

        self.train_sampler.set_epoch(epoch, start_index=consumed_samples_per_GPU)

    def on_validation_epoch_start(self, trainer, pl_module):
        # For validation, we don't care if we finish or not because we never go through a complete epoch of validation set for now
        self.val_sampler.set_epoch(pl_module.current_epoch)
