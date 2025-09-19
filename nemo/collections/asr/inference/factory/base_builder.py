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

from typing import TYPE_CHECKING, Any, Optional, Union

from omegaconf.dictconfig import DictConfig

from nemo.collections.asr.inference.asr.asr_inference import ASRInference
from nemo.collections.asr.inference.asr.cache_aware_ctc_inference import CacheAwareCTCInference
from nemo.collections.asr.inference.asr.cache_aware_rnnt_inference import CacheAwareRNNTInference
from nemo.collections.asr.inference.asr.ctc_inference import CTCInference
from nemo.collections.asr.inference.asr.rnnt_inference import RNNTInference
from nemo.collections.asr.inference.utils.enums import ASRDecodingType, RecognizerType
from nemo.collections.asr.parts.submodules.ctc_decoding import CTCDecodingConfig
from nemo.collections.asr.parts.submodules.rnnt_decoding import RNNTDecodingConfig
from nemo.utils import logging

if TYPE_CHECKING:
    from nemo.collections.asr.inference.itn.batch_inverse_normalizer import BatchAlignmentPreservingInverseNormalizer
    from nemo.collections.asr.inference.pnc.punctuation_capitalizer import PunctuationCapitalizer


class BaseBuilder:

    @classmethod
    def _build_asr(cls, cfg: DictConfig, decoding_cfg: Union[CTCDecodingConfig, RNNTDecodingConfig]) -> ASRInference:
        """
        Build the ASR model based on the config.
        Args:
            cfg: (DictConfig) Config
            decoding_cfg: (Union[CTCDecodingConfig, RNNTDecodingConfig]) Decoding config
        Returns:
            (ASRInference) ASR inference model
        """

        asr_decoding_type = ASRDecodingType.from_str(cfg.asr_decoding_type)
        recognizer_type = RecognizerType.from_str(cfg.recognizer_type)
        comb = (asr_decoding_type, recognizer_type)
        if comb == (ASRDecodingType.CTC, RecognizerType.BUFFERED):
            asr_class = CTCInference
        elif comb == (ASRDecodingType.RNNT, RecognizerType.BUFFERED):
            asr_class = RNNTInference
        elif comb == (ASRDecodingType.CTC, RecognizerType.CACHE_AWARE):
            asr_class = CacheAwareCTCInference
        elif comb == (ASRDecodingType.RNNT, RecognizerType.CACHE_AWARE):
            asr_class = CacheAwareRNNTInference
        else:
            raise ValueError(f"Wrong combination of ASR decoding type and recognizer type: {comb}")

        asr_model = asr_class(
            model_name=cfg.asr.model_name,
            decoding_cfg=decoding_cfg,
            device=cfg.asr.device,
            device_id=cfg.asr.device_id,
            compute_dtype=cfg.asr.compute_dtype,
            use_amp=cfg.asr.use_amp,
        )
        return asr_model

    @classmethod
    def _build_pnc(
        cls, cfg: DictConfig, asr_supports_pnc: bool, force_to_use_pnc_model: bool
    ) -> Optional[PunctuationCapitalizer]:
        """
        Build the PNC model based on the config.
        Args:
            cfg: (DictConfig) Config
            asr_supports_pnc: (bool) Whether the ASR model supports PNC
            force_to_use_pnc_model: (bool) Whether to force the use of the PNC model
        Returns:
            (Optional[PunctuationCapitalizer]) PNC model
        """

        pnc_model = None

        if cfg.automatic_punctuation and asr_supports_pnc and not force_to_use_pnc_model:
            # no need to load the PNC model if the ASR model already supports it
            logging.info("Automatic punctuation is already supported by the ASR model.")
            return pnc_model

        if cfg.automatic_punctuation:

            target_lang = getattr(cfg, "lang", getattr(cfg, "target_lang", None))
            if target_lang is None:
                raise ValueError("Language is not specified. Cannot load external PnC model.")

            if target_lang == "en":
                # Do not remove this import. It is used to avoid megatron import when automatic punctuation is disabled.
                from niva.core.pnc import NivaPunctuationCapitalizer

                pnc_model = NivaPunctuationCapitalizer(
                    model_name=cfg.pnc.model_name,
                    device=cfg.pnc.device,
                    device_id=cfg.pnc.device_id,
                    compute_dtype=cfg.pnc.compute_dtype,
                    use_amp=cfg.pnc.use_amp,
                )
            else:
                logging.info(f"External automatic punctuation is not supported for language {target_lang}.")
        return pnc_model

    @classmethod
    def _build_itn(
        cls, cfg: DictConfig, input_is_lower_cased: bool
    ) -> Optional[BatchAlignmentPreservingInverseNormalizer]:
        """
        Build the ITN model based on the config.
        Args:
            cfg: (DictConfig) Config
            input_is_lower_cased: (bool) Whether the input is lower cased
        Returns:
            (Optional[BatchAlignmentPreservingInverseNormalizer]) ITN model
        """
        itn_model = None
        if not cfg.verbatim_transcripts:
            # Do not remove this import. It is used to avoid nemo_text_processing import when verbatim transcripts is enabled.
            from nemo.collections.asr.inference.itn.batch_inverse_normalizer import (
                BatchAlignmentPreservingInverseNormalizer,
            )

            input_case = (
                BatchAlignmentPreservingInverseNormalizer.LOWER_CASED
                if input_is_lower_cased
                else BatchAlignmentPreservingInverseNormalizer.UPPER_CASED
            )

            target_lang = getattr(cfg, "lang", getattr(cfg, "target_lang", None))
            if target_lang is None:
                raise ValueError("Language is not specified. Cannot load PnC model.")

            itn_cfg = cfg.itn
            itn_cfg.lang = target_lang
            itn_cfg.input_case = input_case
            itn_cfg.cache_dir = cfg.cache_dir

            itn_model = BatchAlignmentPreservingInverseNormalizer(
                lang=itn_cfg.lang,
                input_case=itn_cfg.input_case,
                whitelist=itn_cfg.whitelist,
                cache_dir=itn_cfg.cache_dir,
                overwrite_cache=itn_cfg.overwrite_cache,
                max_number_of_permutations_per_split=itn_cfg.max_number_of_permutations_per_split,
            )
            logging.info(f"Built inverse text normalizer with the input case: `{input_case}`.")
        return itn_model

    @classmethod
    def build(cls, cfg: DictConfig) -> Any:
        """
        Build the recognizer based on the config.
        Args:
            cfg: (DictConfig) Config
        Returns:
            Returns object responsible for the inference
        """
        raise NotImplementedError("This method should be implemented in subclasses.")
