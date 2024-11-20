# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

from torch.utils.data import DataLoader
from nemo.lightning.pytorch.plugins import MegatronDataSampler
from nemo.utils import logging


class SLMDataSampler(MegatronDataSampler):
    def transform_dataloader(self, dataloader: DataLoader, consumed_samples: int = 0) -> DataLoader:

        sampler = getattr(dataloader, 'sampler', None)

        if sampler is not None and "lhotse." in str(type(sampler)):
            logging.info(f"Using Lhotse sampler for dataloader {dataloader}. Skipping Megatron data sampler.")
            return dataloader

        from megatron.core import parallel_state

        from nemo.lightning.data import add_megatron_sampler

        mode = getattr(dataloader, 'mode', 'train')

        data_parallel_rank = parallel_state.get_data_parallel_rank()
        data_parallel_size = parallel_state.get_data_parallel_world_size()
        return add_megatron_sampler(
            dataloader,
            micro_batch_size=self.micro_batch_size,
            global_batch_size=self.global_batch_size,
            rampup_batch_size=self.rampup_batch_size,
            consumed_samples=self.init_consumed_samples if mode == 'train' else 0,
            dataloader_type=self.dataloader_type,
            drop_last=mode not in ["test", "predict"],  # don't drop the incomplete batch in test and predict methods
            dataloader_mode=mode,  # dataloader wrapped with nemo.lightning.data.WrappedDataLoader has mode attribute
            rank=data_parallel_rank,
            world_size=data_parallel_size,
        )
