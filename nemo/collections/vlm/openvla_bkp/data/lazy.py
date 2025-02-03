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

import json
import logging
import os
import re
import tarfile
from typing import Any, Dict, List, Optional, Sequence

import decord
import lightning.pytorch as pl
import numpy as np
import torch
import torch.nn.functional as F
from lightning.pytorch.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from PIL import Image
from torch.utils import data
from torch.utils.data import DataLoader, Dataset, default_collate
from transformers import CLIPImageProcessor, SiglipImageProcessor

from nemo.collections.vlm.openvla_bkp.data.prismatic.util import *
from nemo.collections.nlp.modules.common.megatron.utils import get_ltor_masks_and_position_ids
from nemo.collections.vlm.neva.data.config import DataConfig, ImageDataConfig
from nemo.collections.vlm.neva.data.conversation import conv_templates as supported_conv_templates
from nemo.collections.vlm.neva.data.multimodal_tokens import IGNORE_INDEX, SPECIAL_TOKEN_MAP
from nemo.collections.vlm.openvla_bkp.data.prismatic.models.materialize import (
    get_llm_backbone_and_tokenizer,
    get_vision_backbone_and_transform,
)
from nemo.collections.vlm.openvla_bkp.data.prismatic.util.data_utils import PaddedCollatorForActionPrediction
from nemo.collections.vlm.openvla_bkp.data.prismatic.vla.action_tokenizer import ActionTokenizer
from nemo.collections.vlm.openvla_bkp.data.prismatic.vla.datasets import RLDSBatchTransform, RLDSDataset
from nemo.lightning.pytorch.plugins import MegatronDataSampler


class OpenVLALazyDataModule(pl.LightningDataModule):
    def __init__(
        self,
        paths: str | List[str],
        # additional params for OpenVLA
        data_mix: str,  # Open-X Embodiment Dataset =>> Unique Mixture ID (e.g., `bridge`)
        shuffle_buffer_size: int,  # Size of Shuffle Buffer (100K for Bridge, 1M for OXE)
        weights: Optional[List[float]] = None,
        data_config: Optional[DataConfig] = ImageDataConfig,
        seq_length: int = 2048,
        decoder_seq_length: Optional[int] = None,
        tokenizer: Optional = None,
        image_processor: Optional = None,
        micro_batch_size: int = 4,
        global_batch_size: int = 8,
        num_train_samples: int = 10_000,
        num_val_samples: int = 10_000,
        num_test_samples: int = 10_000,
        num_workers: int = 8,
        pin_memory: bool = True,
        persistent_workers: bool = False,
        use_packed_sequence: bool = False,
        seed: int = 1234,
        # additional params for OpenVLA
        llm_backbone_id: str = "llama2-7b-pure",
        vision_backbone_id: str = "dinosiglip-vit-so-224px",
        llm_max_length: int = 2048,
        load_for_training: bool = False,
        image_resize_strategy: str = "resize-naive",
        predict_stop_token: bool = True,
        padding_side: str = 'right',
        image_aug: bool = False,
        train: bool = True,
        hf_token: str = None,
    ) -> None:
        super().__init__()
        # if not isinstance(paths, (list, tuple)):
        #     paths = [paths]
        # if weights is not None:
        #     assert len(weights) == len(paths)
        #     if len(weights) == 1:
        #         # weights must be None if there is only one dataset
        #         weights = None

        self.paths = paths
        self.data_mix = data_mix
        self.shuffle_buffer_size = shuffle_buffer_size
        self.weights = weights
        self.data_config = data_config
        self.seq_length = seq_length
        self.decoder_seq_length = decoder_seq_length
        self.micro_batch_size = micro_batch_size
        self.global_batch_size = global_batch_size
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.num_train_samples = num_train_samples
        self.num_val_samples = num_val_samples
        self.num_test_samples = num_test_samples
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        self.seed = seed
        self.use_packed_sequence = use_packed_sequence
        self.init_global_step = 0
        # additional params for OpenVLA
        self.llm_backbone_id = llm_backbone_id
        self.vision_backbone_id = vision_backbone_id
        self.llm_max_length = llm_max_length
        self.load_for_training = load_for_training
        self.image_resize_strategy = image_resize_strategy
        self.predict_stop_token = predict_stop_token
        self.padding_side = padding_side
        self.image_aug = image_aug
        self.train = train
        self.hf_token = hf_token

        # if tokenizer is None or image_processor is None:
        #     logging.warning(f"Processor and tokenizer are not provided! Fall back to 'meta-llama/Llama-2-7b-hf'.")
        #     from transformers import AutoProcessor
        #     from nemo.collections.common.tokenizers.huggingface.auto_tokenizer import AutoTokenizer

        #     processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")
        #     self.tokenizer = tokenizer or AutoTokenizer("llava-hf/llava-1.5-7b-hf")
        #     self.image_processor = image_processor or processor.image_processor

        self.llm_backbone, self.tokenizer = get_llm_backbone_and_tokenizer(
            self.llm_backbone_id,
            llm_max_length=self.llm_max_length,
            hf_token=self.hf_token,
            inference_mode=not self.load_for_training,
        )
        self.vision_backbone, self.image_transform = get_vision_backbone_and_transform(
            self.vision_backbone_id,
            self.image_resize_strategy,
        )

        self.action_tokenizer = ActionTokenizer(self.tokenizer)

        self.batch_transform = RLDSBatchTransform(
            self.action_tokenizer,
            self.tokenizer,
            self.image_transform,
            prompt_builder_fn=self.llm_backbone.prompt_builder_fn,
            predict_stop_token=self.predict_stop_token,
        )

        self.collator = PaddedCollatorForActionPrediction(
            self.tokenizer.model_max_length,
            self.tokenizer.pad_token_id,
            padding_side=self.padding_side,
        )

        # DEBUGGING
        # we still set it here because we need self.data_sampler in multiple places (e.g. setup_microbatch_calculator, on_megatron_step_start, compute_consumed_samples),
        # but we disable to process_dataloader() method in megatron_strategy.py

        # TODO(abhinavg): WHy are we using decoder_seq_length here?
        self.data_sampler = MegatronDataSampler(
            seq_len=self.seq_length,
            decoder_seq_len=self.decoder_seq_length,
            micro_batch_size=micro_batch_size,
            global_batch_size=global_batch_size,
            dataloader_type="cyclic",
        )

    def setup(self, stage: str = "") -> None:
        # assert len(self.paths) == 1, "not yet support blend dataset in Neva 2.0!"
        # if self.use_packed_sequence:
        #     pass  # TODO
        # else:
        #     # TODO:
        #     # rng = torch.Generator().manual_seed(self.seed)
        #     # train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size], generator=rng)
        #     self._train_ds = NevaDataset(self.paths[0], self.data_config, self.tokenizer, self.image_processor)
        #     self._validation_ds = NevaDataset(self.paths[0], self.data_config, self.tokenizer, self.image_processor)

        if self.use_packed_sequence:
            pass  # TODO
        else:
            self._train_ds = RLDSDataset(
                self.paths,
                self.data_mix,
                self.batch_transform,
                resize_resolution=self.vision_backbone.default_image_resolution[1:],
                shuffle_buffer_size=self.shuffle_buffer_size,
                train=self.train,
                image_aug=self.image_aug,
            )
            self._validation_ds = self._train_ds
            # self._validation_ds = RLDSDataset(
            #     self.paths,
            #     self.data_mix,
            #     self.batch_transform,
            #     resize_resolution=self.vision_backbone.default_image_resolution[1:],
            #     shuffle_buffer_size=self.shuffle_buffer_size,
            #     train=self.train,
            #     image_aug=self.image_aug,
            # )

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return self._create_dataloader(self._train_ds)

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return self._create_dataloader(self._validation_ds)

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return self._create_dataloader(self._test_ds)

    def _create_dataloader(self, dataset, **kwargs) -> DataLoader:
        self.init_global_step = self.trainer.global_step
        self.data_sampler.init_global_step = self.init_global_step
        return DataLoader(
            dataset,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            collate_fn=self.collator,
            # DEBUGGING
            batch_size=self.micro_batch_size,
            **kwargs,
        )

    def state_dict(self) -> Dict[str, Any]:
        """Called when saving a checkpoint, implement to generate and save datamodule state.

        Returns:
            A dictionary containing datamodule state.

        """
        consumed_samples = self.data_sampler.compute_consumed_samples(self.trainer.global_step - self.init_global_step)
        return {'consumed_samples': consumed_samples}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Called when loading a checkpoint, implement to reload datamodule state given datamodule stat

        Args:
            state_dict: the datamodule state returned by ``state_dict``.

        """
        try:
            from apex.transformer.pipeline_parallel.utils import _GLOBAL_NUM_MICROBATCHES_CALCULATOR
        except ModuleNotFoundError:
            from nemo.lightning.apex_utils import _GLOBAL_NUM_MICROBATCHES_CALCULATOR
        consumed_samples = state_dict['consumed_samples']
        self.data_sampler.init_consumed_samples = consumed_samples
        self.data_sampler.prev_consumed_samples = consumed_samples
        self.if_first_step = 1

        if _GLOBAL_NUM_MICROBATCHES_CALCULATOR is not None:
            num_microbatch_calculator = _GLOBAL_NUM_MICROBATCHES_CALCULATOR  # noqa: SLF001

            num_microbatch_calculator.update(
                consumed_samples=consumed_samples,
                consistency_check=False,
            )


# NeVa input:
#         batch = {
#             'tokens': tokens, # <image> speicial token (-200) - can just prepare labels and prepend -200 in the beginning
#             'labels': labels, # <image> speicial token (-200)
#             'attention_mask': attention_mask, # NeVa and Llama2 : causaul mask no matter - None
#             'loss_mask': loss_mask,
#             'position_ids': position_ids, # NeVa and Llama2: causaul mask no matter - None, only matters if used older GPT model, cannot be None => just generate reasonable
#             'media': media, # [batch_size, 2 x 3, width, height] => make sure concat the rightway, concat and split on dim1
#         }
