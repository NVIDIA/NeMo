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


from typing import Any

import torch
from omegaconf.dictconfig import DictConfig

from nemo.collections.asr.inference.factory.buffered_recognizer_builder import BufferedSpeechRecognizerBuilder
from nemo.collections.asr.inference.factory.cache_aware_recognizer_builder import CacheAwareSpeechRecognizerBuilder
from nemo.collections.asr.inference.utils.enums import RecognizerType
from nemo.utils import logging


class RecognizerBuilder:

    @staticmethod
    def set_matmul_precision(matmul_precision: str) -> None:
        """
        Set the matmul precision.
        Args:
            matmul_precision: (str) Matmul precision: highest, high, medium
        """
        choices = ["highest", "high", "medium"]
        matmul_precision = matmul_precision.lower()
        if matmul_precision not in choices:
            raise ValueError(f"Invalid matmul precision: {matmul_precision}. Need to be one of {choices}")
        torch.set_float32_matmul_precision(matmul_precision)
        logging.info(f"Using matmul precision: {matmul_precision}")

    @staticmethod
    def build_recognizer(cfg: DictConfig) -> Any:
        """
        Build the recognizer based on the config.
        Args:
            cfg: (DictConfig) Config
        Returns:
            Returns Recognizer object
        """
        RecognizerBuilder.set_matmul_precision(cfg.matmul_precision)
        recognizer_type = RecognizerType.from_str(cfg.recognizer_type)
        if recognizer_type is RecognizerType.BUFFERED_STREAMING:
            builder = BufferedSpeechRecognizerBuilder
        elif recognizer_type is RecognizerType.CACHE_AWARE_STREAMING:
            builder = CacheAwareSpeechRecognizerBuilder
        else:
            raise ValueError(f"Invalid recognizer type: {cfg.recognizer_type}")

        return builder.build(cfg)
