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

from typing import Union

from omegaconf.dictconfig import DictConfig

from nemo.collections.asr.inference.factory.base_builder import BaseBuilder
from nemo.collections.asr.inference.stream.recognizers.cache_aware_ctc_recognizer import CacheAwareCTCSpeechRecognizer
from nemo.collections.asr.inference.stream.recognizers.cache_aware_rnnt_recognizer import (
    CacheAwareRNNTSpeechRecognizer,
)
from nemo.collections.asr.inference.utils.enums import ASRDecodingType
from nemo.collections.asr.parts.submodules.ctc_decoding import CTCDecodingConfig
from nemo.collections.asr.parts.submodules.rnnt_decoding import RNNTDecodingConfig
from nemo.utils import logging


class CacheAwareSpeechRecognizerBuilder(BaseBuilder):

    @classmethod
    def build(cls, cfg: DictConfig) -> Union[CacheAwareCTCSpeechRecognizer, CacheAwareRNNTSpeechRecognizer]:
        """
        Build the cache aware streaming recognizer based on the config.
        Args:
            cfg: (DictConfig) Config
        Returns:
            Returns CacheAwareCTCSpeechRecognizer or CacheAwareRNNTSpeechRecognizer object
        """
        asr_decoding_type = ASRDecodingType.from_str(cfg.asr_decoding_type)

        if asr_decoding_type is ASRDecodingType.RNNT:
            return cls.build_cache_aware_rnnt_speech_recognizer(cfg)
        elif asr_decoding_type is ASRDecodingType.CTC:
            return cls.build_cache_aware_ctc_speech_recognizer(cfg)

        raise ValueError(f"Invalid asr decoding type for cache aware streaming. Need to be one of ['CTC', 'RNNT']")

    @classmethod
    def get_rnnt_decoding_cfg(cls) -> RNNTDecodingConfig:
        """
        Get the decoding config for the RNNT recognizer.
        Returns:
            (RNNTDecodingConfig) Decoding config
        """
        decoding_cfg = RNNTDecodingConfig()
        decoding_cfg.strategy = "greedy_batch"
        decoding_cfg.preserve_alignments = False
        decoding_cfg.greedy.use_cuda_graph_decoder = False
        decoding_cfg.greedy.max_symbols = 10
        decoding_cfg.fused_batch_size = -1
        return decoding_cfg

    @classmethod
    def get_ctc_decoding_cfg(cls) -> CTCDecodingConfig:
        """
        Get the decoding config for the CTC recognizer.
        Returns:
            (CTCDecodingConfig) Decoding config
        """
        decoding_cfg = CTCDecodingConfig()
        decoding_cfg.strategy = "greedy"
        decoding_cfg.preserve_alignments = False
        return decoding_cfg

    @classmethod
    def build_cache_aware_rnnt_speech_recognizer(cls, cfg: DictConfig) -> CacheAwareRNNTSpeechRecognizer:
        """
        Build the cache aware RNNT streaming recognizer based on the config.
        Args:
            cfg: (DictConfig) Config
        Returns:
            Returns CacheAwareRNNTSpeechRecognizer object
        """
        # building ASR model
        decoding_cfg = cls.get_rnnt_decoding_cfg()
        asr_model = cls._build_asr(cfg, decoding_cfg)
        logging.info(f"ASR model `{cfg.asr.model_name}` loaded")

        # building PnC model
        asr_supports_pnc = asr_model.supports_punctuation()
        pnc_model = cls._build_pnc(
            cfg,
            asr_supports_pnc=asr_supports_pnc,
            force_to_use_pnc_model=cfg.text_postprocessor.force_to_use_pnc_model,
        )
        if pnc_model is not None:
            logging.info(f"PnC model `{cfg.pnc.model_name}` loaded")

        # building ITN model
        itn_model = cls._build_itn(cfg, input_is_lower_cased=True)
        if itn_model is not None:
            logging.info(f"ITN model loaded")

        ca_rnnt_recognizer = CacheAwareRNNTSpeechRecognizer(cfg, asr_model, pnc_model=pnc_model, itn_model=itn_model)
        logging.info(f"`{type(ca_rnnt_recognizer).__name__}` recognizer loaded")
        return ca_rnnt_recognizer

    @classmethod
    def build_cache_aware_ctc_speech_recognizer(cls, cfg: DictConfig) -> CacheAwareCTCSpeechRecognizer:
        """
        Build the cache aware CTC streaming recognizer based on the config.
        Args:
            cfg: (DictConfig) Config
        Returns:
            Returns CacheAwareCTCSpeechRecognizer object
        """
        # building ASR model
        decoding_cfg = cls.get_ctc_decoding_cfg()
        asr_model = cls._build_asr(cfg, decoding_cfg)
        logging.info(f"ASR model `{cfg.asr.model_name}` loaded")

        # building PnC model
        asr_supports_pnc = asr_model.supports_punctuation()
        pnc_model = cls._build_pnc(
            cfg,
            asr_supports_pnc=asr_supports_pnc,
            force_to_use_pnc_model=cfg.text_postprocessor.force_to_use_pnc_model,
        )
        if pnc_model is not None:
            logging.info(f"PnC model `{cfg.pnc.model_name}` loaded")

        # building ITN model
        itn_model = cls._build_itn(cfg, input_is_lower_cased=True)
        if itn_model is not None:
            logging.info(f"ITN model loaded")

        ca_ctc_recognizer = CacheAwareCTCSpeechRecognizer(cfg, asr_model, pnc_model=pnc_model, itn_model=itn_model)
        logging.info(f"`{type(ca_ctc_recognizer).__name__}` recognizer loaded")
        return ca_ctc_recognizer
