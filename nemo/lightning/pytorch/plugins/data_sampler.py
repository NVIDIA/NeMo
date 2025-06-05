# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

import dataclasses
import logging
from typing import List, Literal, Optional, Union

import lightning.pytorch as pl
from torch.utils.data import DataLoader

from nemo.lightning.megatron_parallel import MegatronStep


class DataSampler:
    def connect(self, trainer: pl.Trainer):
        self.trainer = trainer

    def setup(self, global_rank: int) -> None:
        raise NotImplementedError()

    def transform_dataloader(self, dataloader: DataLoader, consumed_samples: int = 0) -> DataLoader:
        raise NotImplementedError()


class MegatronDataSampler(DataSampler):
    def __init__(
        self,
        seq_len: int,
        micro_batch_size: Union[int, List[int]] = 4,
        global_batch_size: int = 8,
        cu_global_batch_splits: Optional[List[int]] = None,
        rampup_batch_size: Optional[List[int]] = None,
        dataloader_type: Literal["single", "cyclic", "batch"] = "single",
        init_consumed_samples: int = 0,
        init_global_step: int = 0,
        output_log: bool = True,
        decoder_seq_len: Optional[int] = None,
    ):
        """
        Args:
            micro_batch_size: int or list of ints, default is 4.
                The size of the micro-batch to use for each GPU. An int config means the same value for all GPUs.
                If a list is provided, the size will be different for each GPU. For example, in multi-dataceter
                running, GPUs of each dataceter can have different micro-batch sizes.
            cu_global_batch_splits: list of ints, default is None.
                Cumulative global batch size splits for multiple GPU world ranges. This is to support cases where
                GPU world size is split into multiple ranges, which are collection of same or different GPUs (e.g.,
                multi-datacenters). Global batch size is split and assigned to each range.
                For example, with 2 GPU ranges/datacenters and global_batch_size=16, cu_global_batch_splits can be
                [0, 4, 16], or [0, 8, 16], etc. [0, 4, 16] means global batch split of 4 and 8 for each range.
        """
        self.seq_len = seq_len
        self.decoder_seq_len = decoder_seq_len
        self.output_log = output_log
        self.micro_batch_size = micro_batch_size
        self.global_batch_size = global_batch_size
        self.cu_global_batch_splits = cu_global_batch_splits
        self.rampup_batch_size = rampup_batch_size
        self.dataloader_type = dataloader_type
        self.init_consumed_samples = init_consumed_samples
        self.prev_consumed_samples = self.init_consumed_samples
        self.if_first_step = 0
        self.prev_global_batch_size = None
        self.init_global_step = init_global_step

    def setup(self, global_rank: int) -> None:
        from megatron.core import parallel_state
        from nemo.lightning.data import setup_microbatch_calculator

        self.data_parallel_rank = parallel_state.get_data_parallel_rank()
        self.data_parallel_size = parallel_state.get_data_parallel_world_size()
        self.global_batch_split_range = None
        assert (
            isinstance(self.micro_batch_size, int) or self.cu_global_batch_splits is not None
        ), f"micro_batch_size: {self.micro_batch_size} can be a list only when cu_global_batch_splits is provided!"

        if self.cu_global_batch_splits is not None:
            # calculate DP info and global batch split range for the rank range where the current GPU belongs to
            # GPU world size consists of (len(cu_global_batch_splits) - 1) rank ranges, assuming all ranges have
            # same number of GPUs and same parallelism config.
            num_rank_ranges = len(self.cu_global_batch_splits) - 1
            self.data_parallel_size = self.data_parallel_size // num_rank_ranges
            rank_range_id = self.data_parallel_rank // self.data_parallel_size
            self.data_parallel_rank = self.data_parallel_rank % self.data_parallel_size

            if isinstance(self.micro_batch_size, list):
                assert (
                    len(self.micro_batch_size) == num_rank_ranges
                ), f"The length of micro_batch_size: {self.micro_batch_size} should be same as the number of \
                    world size split ranges: {num_rank_ranges}!"
                self.micro_batch_size = self.micro_batch_size[rank_range_id]

            assert (
                self.cu_global_batch_splits[0] == 0 and self.cu_global_batch_splits[-1] == self.global_batch_size
            ), f"cu_global_batch_splits: {self.cu_global_batch_splits} should start with 0 and end with \
                {self.global_batch_size}!"
            self.global_batch_split_range = (
                self.cu_global_batch_splits[rank_range_id],
                self.cu_global_batch_splits[rank_range_id + 1],
            )

        setup_microbatch_calculator(
            global_rank,
            self.data_parallel_size,
            self.micro_batch_size,
            self.global_batch_size,
            self.global_batch_split_range,
            self.rampup_batch_size,
        )

    def transform_dataloader(self, dataloader: DataLoader, consumed_samples: int = 0) -> DataLoader:
        from nemo.lightning.data import add_megatron_sampler

        mode = getattr(dataloader, 'mode', 'train')

        return add_megatron_sampler(
            dataloader,
            micro_batch_size=self.micro_batch_size,
            global_batch_size=self.global_batch_size,
            global_batch_split_range=self.global_batch_split_range,
            rampup_batch_size=self.rampup_batch_size,
            consumed_samples=self.init_consumed_samples if mode == 'train' else 0,
            dataloader_type=self.dataloader_type,
            drop_last=mode not in ["test", "predict"],  # don't drop the incomplete batch in test and predict methods
            dataloader_mode=mode,  # dataloader wrapped with nemo.lightning.data.WrappedDataLoader has mode attribute
            rank=self.data_parallel_rank,
            world_size=self.data_parallel_size,
        )

    def compute_consumed_samples(self, steps_since_resume=0) -> int:
        from nemo.lightning.pytorch.strategies import MegatronStrategy

        if not hasattr(self, "trainer") or not isinstance(self.trainer.strategy, MegatronStrategy):
            return 0

        if self.rampup_batch_size is not None:
            consumed_samples = self.prev_consumed_samples + self.if_first_step * self.current_global_batch_size
        else:
            consumed_samples = self.init_consumed_samples + steps_since_resume * self.current_global_batch_size

        return int(consumed_samples)

    # Megatron callbacks

    def on_megatron_step_start(self, step: MegatronStep) -> MegatronStep:
        return dataclasses.replace(
            step,
            seq_length=self.seq_len,
            micro_batch_size=self.micro_batch_size,
            num_microbatches=self.num_microbatches,
            decoder_seq_length=self.decoder_seq_len,
        )

    def on_megatron_microbatches_start(self, step: MegatronStep) -> None:
        if not step.trainer:
            return

        # do validation and save the checkpoint when gbs is changed
        if (
            self.rampup_batch_size is not None
            and self.prev_global_batch_size != self.current_global_batch_size
            and self.prev_global_batch_size
        ):
            step.trainer.should_stop = True

    def on_megatron_step_end(self, step: MegatronStep) -> None:
        trainer = step.trainer
        pl_module = step.pl_module

        try:
            from megatron.core.num_microbatches_calculator import update_num_microbatches

        except (ImportError, ModuleNotFoundError):
            logging.warning("Megatron num_microbatches_calculator not found, using Apex version.")
            from apex.transformer.pipeline_parallel.utils import update_num_microbatches

        self.prev_global_batch_size = self.current_global_batch_size

        if step.step_i:
            consumed_samples = self.compute_consumed_samples(step.step_i + 1 - self.init_global_step)
            if self.output_log and trainer and getattr(trainer, "training", False):
                # You may need to turn off logging, for example when doing trainer.predict(model, data)
                pl_module.log(
                    'consumed_samples',
                    consumed_samples,
                    prog_bar=True,
                    batch_size=1,
                )

            self.prev_consumed_samples = consumed_samples

            update_num_microbatches(
                consumed_samples=consumed_samples,
                consistency_check=False,
            )
        if self.output_log and trainer:
            # You may need to turn off logging, for example when doing trainer.predict(model, data)
            pl_module.log(
                "global_batch_size",
                self.current_global_batch_size,
                prog_bar=True,
                batch_size=1,
            )
        self.if_first_step = 1

    @property
    def num_microbatches(self) -> int:
        try:
            from megatron.core.num_microbatches_calculator import get_num_microbatches

        except (ImportError, ModuleNotFoundError):
            logging.warning("Megatron num_microbatches_calculator not found, using Apex version.")
            from apex.transformer.pipeline_parallel.utils import get_num_microbatches

        return get_num_microbatches()

    @property
    def current_global_batch_size(self) -> int:
        try:
            from megatron.core.num_microbatches_calculator import get_current_global_batch_size

        except (ImportError, ModuleNotFoundError):
            logging.warning("Megatron num_microbatches_calculator not found, using Apex version.")
            from apex.transformer.pipeline_parallel.utils import get_current_global_batch_size

        if get_current_global_batch_size():
            current_global_batch_size = get_current_global_batch_size()
        else:
            current_global_batch_size = 1

        return current_global_batch_size
