# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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
from abc import ABC, abstractmethod
from typing import Dict, List

import torch

from nemo.core.classes import ModelPT

__all__ = ['ASRModel']


class ASRModel(ModelPT, ABC):
    @abstractmethod
    def transcribe(self, paths2audio_files: List[str], batch_size: int = 4) -> List[str]:
        """
        Takes paths to audio files and returns text transcription
        Args:
            paths2audio_files: paths to audio fragment to be transcribed

        Returns:
            transcription texts
        """
        pass

    def multi_validation_epoch_end(self, outputs, dataloader_idx: int = 0):
        val_loss_mean = torch.stack([x['val_loss'] for x in outputs]).mean()
        wer_num = torch.stack([x['val_wer_num'] for x in outputs]).sum()
        wer_denom = torch.stack([x['val_wer_denom'] for x in outputs]).sum()
        tensorboard_logs = {'validation_loss': val_loss_mean, 'validation_wer': wer_num / wer_denom}
        return {'val_loss': val_loss_mean, 'log': tensorboard_logs}
