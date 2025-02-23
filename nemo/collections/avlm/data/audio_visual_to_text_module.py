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

from functools import lru_cache
from typing import Any, Dict, List, Union

from megatron.core import parallel_state
from omegaconf.omegaconf import DictConfig

from nemo.collections.asr.parts.preprocessing.perturb import process_augmentations as audio_process_augmentations
from nemo.collections.common.tokenizers import TokenizerSpec
from nemo.collections.speechlm.data.audio_to_text_module import AudioToTextDataModule
from nemo.collections.speechlm.data.dataset.audio_text_dataset import (
    get_audio_visual_text_webdataset_from_config,
)

from nemo.utils import logging


class AudioVisualToTextDataModule(AudioToTextDataModule):
    """
    Data module for speech-image/video-to-text LLM.
    """

    def __init__(self, config: Union[DictConfig, Dict], tokenizer: TokenizerSpec, visual_processor = None):
        super().__init__(config, tokenizer)
        # image/video processor
        self.visual_processor = visual_processor
        
    @lru_cache
    def _create_dataset(self, mode: str):
        """
        Create the datasets for each of train/validation/test/predict mode
        """
        data_cfg = self.cfg.get(f"{mode}_ds", None)
        if data_cfg is None:
            logging.info(f"Skipping {mode} dataset creation as it is not specified in the config: {self.cfg}")
            return None

        audio_augmentor = None
        if 'audio' in data_cfg and 'augmentor' in data_cfg.get('audio'):
            audio_augmentor = audio_process_augmentations(
                data_cfg['audio']['augmentor'],
                global_rank=parallel_state.get_data_parallel_rank(),
                world_size=parallel_state.get_data_parallel_world_size(),
            )

        # TODO: implement the image/video augmentor
        visual_augmentor = None            

        # Notably, the data weights are controlled by either bucketing_weights
        # or concat_sampling_probabilities depending on the dataset type.
        if data_cfg.get("use_lhotse"):
            logging.info(f"lhotse dataset is not yet supported. Skipping {mode} dataset creation .")
            return None

        setattr(self, f"_{mode}_names", data_cfg.get('name', None))

        # Notably, the data weights are controlled by either bucketing_weights
        # or concat_sampling_probabilities depending on the dataset type.
        if data_cfg.get('is_wds', False):
            dataset = get_audio_visual_text_webdataset_from_config(
                config=data_cfg,
                text_processor=self.text_processor,
                visual_processor=self.visual_processor,
                audio_augmentor=audio_augmentor,
                visual_augmentor=visual_augmentor,
                global_rank=parallel_state.get_data_parallel_rank(),
                world_size=parallel_state.get_data_parallel_world_size(),
            )
        else:
            # TODO: implement get_audio_text_dataset_from_config for non wds compliant dataset
            logging.info(f"None webdataset compliant data is not yet supported. Skipping {mode} dataset creation.")
            return None

        if mode != 'train':
            num_ds = len(dataset) if isinstance(dataset, list) else 1
            setattr(self, f"_num_{mode}_dl", num_ds)
        return dataset
