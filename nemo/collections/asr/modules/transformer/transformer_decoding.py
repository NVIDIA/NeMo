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

import re
from abc import abstractmethod
from dataclasses import dataclass, field, is_dataclass
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch import nn

from nemo.collections.asr.modules.transformer import BeamSearchSequenceGenerator
from nemo.collections.asr.parts.submodules import ctc_beam_decoding, ctc_greedy_decoding
from nemo.collections.asr.parts.utils.asr_confidence_utils import ConfidenceConfig, ConfidenceMixin
from nemo.collections.asr.parts.utils.rnnt_utils import Hypothesis, NBestHypotheses
from nemo.collections.common.tokenizers.aggregate_tokenizer import DummyTokenizer
from nemo.collections.common.tokenizers.tokenizer_spec import TokenizerSpec
from nemo.utils import logging, logging_mode


def lens_to_mask(lens, max_length):
    batch_size = lens.shape[0]
    mask = torch.arange(max_length).repeat(batch_size, 1).to(lens.device) < lens[:, None]
    return mask


class TransformerDecoding:
    """
    Used for performing Transformer auto-regressive of the logprobs.

    """

    def __init__(
        self, decoding_cfg, transformer_decoder: torch.Tensor, classifier: torch.Tensor, tokenizer: TokenizerSpec
    ):
        # super().__init__()

        # Convert dataclas to config
        if is_dataclass(decoding_cfg):
            decoding_cfg = OmegaConf.structured(decoding_cfg)

        if not isinstance(decoding_cfg, DictConfig):
            decoding_cfg = OmegaConf.create(decoding_cfg)

        OmegaConf.set_struct(decoding_cfg, False)

        self.cfg = decoding_cfg
        self.tokenizer = tokenizer

        possible_strategies = ['beam']
        if self.cfg.strategy not in possible_strategies:
            raise ValueError(f"Decoding strategy must be one of {possible_strategies}. Given {self.cfg.strategy}")

        if self.cfg.strategy == 'beam':

            self.decoding = BeamSearchSequenceGenerator(
                embedding=transformer_decoder.embedding,
                decoder=transformer_decoder.decoder,
                log_softmax=classifier,
                max_sequence_length=transformer_decoder.max_sequence_length,
                bos=self.tokenizer.bos_id,
                pad=self.tokenizer.pad_id,
                eos=self.tokenizer.eos_id,
                beam_size=self.cfg.beam_size,
                len_pen=self.cfg.len_pen,
                max_delta_length=self.cfg.max_delta_length,
            )

        else:
            raise ValueError(
                f"Incorrect decoding strategy supplied. Must be one of {possible_strategies}\n"
                f"but was provided {self.cfg.strategy}"
            )

    def transformer_decoder_predictions_tensor(
        self, encoder_output: torch.Tensor, encoder_lengths: torch.Tensor = None, encoder_mask: torch.Tensor = None
    ) -> Tuple[List[str]]:
        """
        """
        if encoder_mask is None:
            encoder_mask = lens_to_mask(encoder_lengths, encoder_output.shape[1]).to(encoder_output.dtype)

        with torch.inference_mode():
            beam_hypotheses = (
                self.decoding(
                    encoder_hidden_states=encoder_output, encoder_input_mask=encoder_mask, return_beam_scores=False
                )
                .detach()
                .cpu()
                .numpy()
            )
            beam_hypotheses = [self.decode_tokens_to_str(hyp) for hyp in beam_hypotheses]
            return beam_hypotheses, None

    def decode_tokens_to_str(self, tokens: List[int]) -> str:
        """
        Implemented by subclass in order to decoder a token list into a string.

        Args:
            tokens: List of int representing the token ids.

        Returns:
            A decoded string.
        """
        hypothesis = self.tokenizer.ids_to_text(tokens)
        return hypothesis

    def decode_ids_to_tokens(self, tokens: List[int]) -> List[str]:
        """
        Implemented by subclass in order to decode a token id list into a token list.
        A token list is the string representation of each token id.

        Args:
            tokens: List of int representing the token ids.

        Returns:
            A list of decoded tokens.
        """
        token_list = self.tokenizer.ids_to_tokens(tokens)
        return token_list


@dataclass
class TransformerBPEConfig:
    strategy: str = "beam"

    beam_size: int = 4
    len_pen: float = 0.0
    max_delta_length: int = 50
