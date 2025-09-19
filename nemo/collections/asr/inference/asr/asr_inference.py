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


import copy
from functools import cached_property
from typing import List, Set, Union

import torch
from omegaconf import open_dict

from nemo.collections.asr.inference.utils.constants import SENTENCEPIECE_UNDERSCORE
from nemo.collections.asr.inference.utils.device_utils import setup_device
from nemo.collections.asr.models import ASRModel, EncDecHybridRNNTCTCModel
from nemo.collections.asr.parts.submodules.ctc_decoding import CTCDecodingConfig
from nemo.collections.asr.parts.submodules.multitask_decoding import MultiTaskDecodingConfig
from nemo.collections.asr.parts.submodules.rnnt_decoding import RNNTDecodingConfig
from nemo.collections.asr.parts.utils.asr_confidence_utils import get_confidence_aggregation_bank

SUPPORTED_CONFIDENCE_AGGREGATORS = get_confidence_aggregation_bank()


class ASRInference:

    def __init__(
        self,
        model_name: str,
        decoding_cfg: Union[CTCDecodingConfig, RNNTDecodingConfig, MultiTaskDecodingConfig],
        device: str = 'cuda',
        device_id: int = 0,
        compute_dtype: str = 'bfloat16',
        use_amp: bool = True,
    ):
        """
        Args:
            model_name: (str) path to the model checkpoint or a model name from the NGC cloud.
            decoding_cfg: (Union[CTCDecodingConfig, RNNTDecodingConfig]) decoding configuration.
            device: (str) device to run the model on.
            device_id: (int) device ID to run the model on.
            compute_dtype: (str) compute dtype to run the model on.
            use_amp: (bool) Use Automatic Mixed Precision
        """

        self.decoding_cfg = decoding_cfg
        self.device_str, self.device_id, self.compute_dtype = setup_device(device.strip(), device_id, compute_dtype)
        self.device = torch.device(self.device_str)
        self.use_amp = use_amp
        self.asr_model = self.load_model(model_name, self.device)
        self.asr_model_cfg = self.asr_model._cfg
        self.set_dither_to_zero()
        self.tokenizer = self.asr_model.tokenizer

        # post initialization steps that must be implemented in the derived classes
        self.__post_init__()

    @staticmethod
    def load_model(model_name: str, map_location: torch.device) -> ASRModel:
        """
        Load the ASR model.
        Args:
            model_name: (str) path to the model checkpoint or a model name from the NGC cloud.
            map_location: (torch.device) device to load the model on.
        Returns:
            (ASRModel) loaded ASR model.
        """
        try:
            if model_name.endswith('.nemo'):
                asr_model = ASRModel.restore_from(model_name, map_location=map_location)
            else:
                asr_model = ASRModel.from_pretrained(model_name, map_location=map_location)
            asr_model.eval()
            return asr_model
        except Exception as e:
            raise RuntimeError(f"Failed to load model {model_name}: {str(e)}")

    @property
    def segment_separators(self) -> List[str]:
        """
        Returns:
            (List[str]) list of segment separators.
        """
        return self.decoding_cfg.segment_seperators

    @property
    def word_separator(self) -> str:
        """
        Returns:
            (str) word separator.
        """
        return self.decoding_cfg.word_seperator

    @property
    def confidence_aggregator(self):
        """
        Returns:
            (function) confidence aggregator function.
        """
        return SUPPORTED_CONFIDENCE_AGGREGATORS[self.decoding_cfg.confidence_cfg.aggregation]

    def copy_asr_config(self):
        """
        Returns:
            (dict) copy of the ASR model configuration.
        """
        return copy.deepcopy(self.asr_model_cfg)

    def supports_capitalization(self) -> bool:
        """
        Returns:
            (bool) True if the ASR model supports capitalization, False otherwise.
        """
        if not hasattr(self, "asr_model") or self.asr_model is None:
            raise ValueError("ASR model is not initialized.")
        return self.tokenizer.supports_capitalization

    def supports_punctuation(self) -> bool:
        """
        Returns:
            (bool) True if the ASR model supports punctuation, False otherwise.
        """
        if not hasattr(self, "asr_model") or self.asr_model is None:
            raise ValueError("ASR model is not initialized.")
        return self.supported_punctuation() != set()

    def supported_punctuation(self) -> Set:
        """
        Returns:
            (set) Set of supported punctuation symbols.
        """
        if not hasattr(self, "asr_model") or self.asr_model is None:
            raise ValueError("ASR model is not initialized.")
        return self.tokenizer.supported_punctuation - set("'")

    def get_punctuation_ids(self) -> Set:
        """
        Returns:
            (set) Set of punctuation ids.
        """
        punctuation_ids = set()
        if self.supports_punctuation():
            for punctuation in self.supported_punctuation():
                punctuation_ids.add(self.tokenizer.tokens_to_ids(punctuation)[0])
        return punctuation_ids

    @cached_property
    def get_underscore_id(self) -> int:
        """
        Returns:
            (int) underscore id for the model.
        """
        if getattr(self.asr_model.tokenizer, "spm_separator_id", None) is not None:
            return self.asr_model.tokenizer.spm_separator_id
        else:
            return self.asr_model.tokenizer.tokens_to_ids(SENTENCEPIECE_UNDERSCORE)

    @cached_property
    def get_language_token_ids(self) -> Set:
        """
        This property is used for some Riva models that have language tokens included in their vocabulary.
        Returns:
            (set) Set of language token ids.
        """
        vocab = self.get_vocabulary()
        language_token_ids = set()
        for token in vocab:
            if token.startswith("<") and token.endswith(">") and token != "<unk>":
                language_token_ids.add(self.asr_model.tokenizer.tokens_to_ids(token)[0])
        return language_token_ids

    def reset_decoding_strategy(self, decoder_type: str) -> None:
        """
        Reset the decoding strategy for the model.
        Args:
            decoder_type: (str) decoding type either 'ctc', 'rnnt' or 'aed'.
        """
        if isinstance(self.asr_model, EncDecHybridRNNTCTCModel):
            self.asr_model.change_decoding_strategy(decoding_cfg=None, decoder_type=decoder_type)
        else:
            self.asr_model.change_decoding_strategy(None)

    def set_decoding_strategy(self, decoder_type: str) -> None:
        """
        Set the decoding strategy for the model.
        Args:
            decoder_type: (str) decoding type either 'ctc', 'rnnt' or 'aed'.
        """
        if isinstance(self.asr_model, EncDecHybridRNNTCTCModel):
            self.asr_model.change_decoding_strategy(decoding_cfg=self.decoding_cfg, decoder_type=decoder_type)
        else:
            self.asr_model.change_decoding_strategy(self.decoding_cfg)

    def set_dither_to_zero(self) -> None:
        """
        To remove randomness from preprocessor set the dither value to zero.
        """
        self.asr_model.preprocessor.featurizer.dither = 0.0
        with open_dict(self.asr_model_cfg):
            self.asr_model_cfg.preprocessor.dither = 0.0

    # Methods that must be implemented in the derived classes.
    def __post_init__(self):
        """
        Additional post initialization steps that must be implemented in the derived classes.
        """
        raise NotImplementedError()

    def get_blank_id(self) -> int:
        """
        Returns:
            (int) blank id for the model.
        """
        raise NotImplementedError()

    def get_vocabulary(self) -> List[str]:
        """
        Returns:
            (List[str]) list of vocabulary tokens.
        """
        raise NotImplementedError()

    def get_subsampling_factor(self) -> int:
        """
        Returns:
            (int) subsampling factor for the model.
        """
        raise NotImplementedError()
