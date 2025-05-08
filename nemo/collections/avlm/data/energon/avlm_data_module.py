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

from typing import Optional, Literal

from megatron.energon import get_train_dataset

from nemo.collections.multimodal.data.energon import EnergonMultiModalDataModule
from nemo.collections.avlm.data.energon.avlm_sample_config import AVLMSampleConfig
from nemo.collections.avlm.data.energon.avlm_task_encoder import AVLMTaskEncoder


class AVLMDataModule(EnergonMultiModalDataModule):
    """
    Energon Data module for media(audio and/or image and/or video and text)-to-text LLM.
    
    It looks for a path leads to path/.nv-meta/dataset.yaml 
    which looks like the follows:
    sample_type:
      __module__: nemo.collections.avlm.data.energon.media_to_text_config
      __class__: MediaToMediaSample
    field_map:
      image: image_file_extension
      video: video_file_extension
      audio: audio_file_extension
      offset: ...

    More data preparation details can be found at:
    https://github.com/NVIDIA/Megatron-Energon/blob/24123d4b63a451980eec6ee0ddaa4f007ef562e7/docs/source/basic/data_prep.md?plain=1#L85
    """
        
    def __init__(
        self,
        path: str,
        tokenizer=None,
        audio_processor=None,
        image_processor=None,
        seq_length: int = 2048,
        micro_batch_size: int = 1,
        global_batch_size: int = 1,
        num_workers: int = 1,
        num_val_workers: int | None = None,
        pin_memory: bool = True,
        shuffle_buffer_size: int = 100,
        max_samples_per_sequence: int | None = None,
        multimodal_sample_config: Optional[AVLMSampleConfig] = AVLMSampleConfig(),
        task_encoder: Optional[AVLMTaskEncoder] = None,
        decoder_seq_length: Optional[int] = None,
        packing_buffer_size: Optional[int] = None,
        validation_task_encoder: Optional[AVLMTaskEncoder] = None,
        **kwargs,
    ) -> None:
        super().__init__(
            path,
            tokenizer,
            image_processor,
            seq_length,
            micro_batch_size,
            global_batch_size,
            num_workers,
            num_val_workers,
            pin_memory,
            shuffle_buffer_size,
            max_samples_per_sequence,
            multimodal_sample_config,
            task_encoder,
            decoder_seq_length,
            packing_buffer_size,
            validation_task_encoder,
            **kwargs,
        )
        self.audio_processor = audio_processor

    def datasets_provider(self, worker_config, split: Literal['train', 'val'] = 'val'):
        """
        Provide the dataset for training or validation.

        This method retrieves the dataset for the specified split (either 'train' or 'val') and configures
        it according to the worker configuration.

        audio, video and image files wil be passed to the task encoder as raw bytes which will be processed in a 
        more fine-grained way.

        Parameters:
        worker_config: Configuration for the data loader workers.
        split (Literal['train', 'val'], optional): The data split to retrieve ('train' or 'val'). Defaults to 'val'.

        Returns:
        Dataset: The dataset configured for the specified split.
        """

        if split not in {'train', 'val'}:
            raise ValueError("Invalid value for split. Allowed values are 'train' or 'val'.")

        if split == "train":
            task_encoder = self.task_encoder
        else:
            task_encoder = self.validation_task_encoder

        _dataset = get_train_dataset(
            self.path,
            batch_size=self.micro_batch_size,
            task_encoder=task_encoder,
            worker_config=worker_config,
            packing_buffer_size=self.packing_buffer_size,
            split_part=split,
            shuffle_buffer_size=self.shuffle_buffer_size,
            max_samples_per_sequence=self.max_samples_per_sequence,
            auto_decode=False,
            **self.kwargs,
        )

        return _dataset
