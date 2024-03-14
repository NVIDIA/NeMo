import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

from nemo.collections.asr.parts.preprocessing.segment import AudioSegment
from nemo.collections.asr.parts.utils.manifest_utils import read_manifest
from nemo.collections.common.parts.preprocessing.manifest import get_full_path
from nemo.core.classes import Exportable, NeuralModule, typecheck


class WavLMAugmentation(object):
    def __init__(
        self,
        prob: float = 0.0,
        noise_ratio: float = 0.0,
        noise_manifest: Optional[str] = None,
        target_sr: int = 16000,
    ):
        super().__init__()
        self.prob = prob
        self.noise_ratio = noise_ratio
        self.noise_manifest = noise_manifest
        self.target_sr = target_sr
        self.noise_data = self.load_noise_manifest(noise_manifest)
        if not (0 <= self.prob <= 1):
            raise ValueError(f"prob must be in [0, 1], got: {self.prob}")
        if not (0 <= self.noise_ratio <= 1):
            raise ValueError(f"noise_ratio must be in [0, 1], got: {self.noise_ratio}")

    def load_noise_manifest(self, noise_manifest: str):
        noise_manifest_list = noise_manifest.split(',')
        noise_data = []
        for manifest in noise_manifest_list:
            curr_data = read_manifest(manifest)
            for i in range(len(curr_data)):
                curr_data[i]['audio_filepath'] = get_full_path(curr_data[i]['audio_filepath'], manifest)
            noise_data.extend(curr_data)
        return noise_data

    def load_noise_audio(self, sample: Dict[str, Any]):
        audio_segment = AudioSegment.from_file(
            audio_file=sample['audio_filepath'],
            offset=sample.get('offset', 0.0),
            duration=sample.get("duration", 0.0),
            target_sr=self.target_sr,
        )
        return torch.tensor(audio_segment.samples, dtype=torch.float)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        B, D, T = inputs.shape
        if self.prob == 0 or np.random.uniform() > self.prob:
            return inputs
