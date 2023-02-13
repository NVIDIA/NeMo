# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import partial
from typing import List, Optional

import torch
from omegaconf import DictConfig, OmegaConf

from nemo.collections.asr.parts.utils.rnnt_utils import Hypothesis


class ConfidenceMeasureConstants:
    NAMES = ("max_prob", "entropy")
    ENTROPY_TYPES = ("gibbs", "tsallis", "renyi")
    ENTROPY_NORMS = ("lin", "exp")


class ConfidenceConstants:
    AGGREGATIONS = ("mean", "min", "max", "prod")


@dataclass
class ConfidenceMeasureConfig:
    name: str = "entropy"
    entropy_type: str = "tsallis"
    temperature: float = 0.33
    entropy_norm: str = "exp"

    def __post_init__(self):
        if self.name not in ConfidenceMeasureConstants.NAMES:
            raise ValueError(
                f"`name` has to be one of the following: "
                f"{'`' + '`, `'.join(ConfidenceMeasureConstants.NAMES) + '`'}. Provided: `{self.name}`"
            )
        if self.entropy_type not in ConfidenceMeasureConstants.ENTROPY_TYPES:
            raise ValueError(
                f"`entropy_type` has to be one of the following: "
                f"{'`' + '`, `'.join(ConfidenceMeasureConstants.ENTROPY_TYPES) + '`'}. Provided: `{self.entropy_type}`"
            )
        if self.temperature <= 0.0:
            raise ValueError(f"`temperature` has to be > 0. Provided: {self.temperature}")
        if self.entropy_norm not in ConfidenceMeasureConstants.ENTROPY_NORMS:
            raise ValueError(
                f"`entropy_norm` has to be one of the following: "
                f"{'`' + '`, `'.join(ConfidenceMeasureConstants.ENTROPY_NORMS) + '`'}. Provided: `{self.entropy_norm}`"
            )


@dataclass
class ConfidenceConfig:
    preserve_frame_confidence: bool = False
    preserve_token_confidence: bool = False
    preserve_word_confidence: bool = False
    exclude_blank: bool = True
    aggregation: str = "min"
    measure_cfg: ConfidenceMeasureConfig = ConfidenceMeasureConfig()

    def __post_init__(self):
        if self.aggregation not in ConfidenceConstants.AGGREGATIONS:
            raise ValueError(
                f"`aggregation` has to be one of the following: "
                f"{'`' + '`, `'.join(ConfidenceMeasureConstants.AGGREGATIONS) + '`'}. Provided: `{self.aggregation}`"
            )


def get_confidence_measure_bank():
    """Generate a dictionary with confidence measure functionals.

    Supported confidence measures:
        max_prob: normalized maximum probability
        entropy_gibbs_lin: Gibbs entropy with linear normalization
        entropy_gibbs_exp: Gibbs entropy with exponential normalization
        entropy_tsallis_lin: Tsallis entropy with linear normalization
        entropy_tsallis_exp: Tsallis entropy with exponential normalization
        entropy_renyi_lin: Rényi entropy with linear normalization
        entropy_renyi_exp: Rényi entropy with exponential normalization

    Returns:
        dictionary with lambda functions.
    """
    # helper functions
    # Gibbs entropy is implemented without temperature
    neg_entropy_gibbs = lambda x: (x.exp() * x).sum(-1)
    neg_entropy_temperature = lambda x, t: (x * t).exp().sum(-1)
    neg_entropy_temperature_gibbs = lambda x, t: ((x * t).exp() * x).sum(-1)
    # too big for a lambda
    def entropy_tsallis_exp(x, v, t):
        exp_neg_max_ent = math.exp((1 - math.pow(v, 1 - t)) / (1 - t))
        return (((1 - neg_entropy_temperature(x, t)) / (1 - t)).exp() - exp_neg_max_ent) / (1 - exp_neg_max_ent)

    def entropy_gibbs_exp(x, v, t):
        exp_neg_max_ent = math.pow(v, -t * math.pow(v, 1 - t))
        return ((neg_entropy_temperature_gibbs(x, t) * t).exp() - exp_neg_max_ent) / (1 - exp_neg_max_ent)

    # use Gibbs entropies for Tsallis and Rényi with t == 1.0
    entropy_gibbs_lin_baseline = lambda x, v: 1 + neg_entropy_gibbs(x) / math.log(v)
    entropy_gibbs_exp_baseline = lambda x, v: (neg_entropy_gibbs(x).exp() * v - 1) / (v - 1)
    # fill the measure bank
    confidence_measure_bank = {}
    # Maximum probability measure is implemented without temperature
    confidence_measure_bank["max_prob"] = (
        lambda x, v, t: (x.max(dim=-1)[0].exp() * v - 1) / (v - 1)
        if t == 1.0
        else ((x.max(dim=-1)[0] * t).exp() * math.pow(v, t) - 1) / (math.pow(v, t) - 1)
    )
    confidence_measure_bank["entropy_gibbs_lin"] = (
        lambda x, v, t: entropy_gibbs_lin_baseline(x, v)
        if t == 1.0
        else 1 + neg_entropy_temperature_gibbs(x, t) / math.log(v) / math.pow(v, 1 - t)
    )
    confidence_measure_bank["entropy_gibbs_exp"] = (
        lambda x, v, t: entropy_gibbs_exp_baseline(x, v) if t == 1.0 else entropy_gibbs_exp(x, v, t)
    )
    confidence_measure_bank["entropy_tsallis_lin"] = (
        lambda x, v, t: entropy_gibbs_lin_baseline(x, v)
        if t == 1.0
        else 1 + (1 - neg_entropy_temperature(x, t)) / (math.pow(v, 1 - t) - 1)
    )
    confidence_measure_bank["entropy_tsallis_exp"] = (
        lambda x, v, t: entropy_gibbs_exp_baseline(x, v) if t == 1.0 else entropy_tsallis_exp(x, v, t)
    )
    confidence_measure_bank["entropy_renyi_lin"] = (
        lambda x, v, t: entropy_gibbs_lin_baseline(x, v)
        if t == 1.0
        else 1 + neg_entropy_temperature(x, t).log2() / (t - 1) / math.log(v, 2)
    )
    confidence_measure_bank["entropy_renyi_exp"] = (
        lambda x, v, t: entropy_gibbs_exp_baseline(x, v)
        if t == 1.0
        else (neg_entropy_temperature(x, t).pow(1 / (t - 1)) * v - 1) / (v - 1)
    )
    return confidence_measure_bank


def get_confidence_aggregation_bank():
    """Generate a dictionary with confidence aggregation functions.

    Supported confidence measures:
        min: minimum
        max: maximum
        mean: arithmetic mean
        prod: product

    Returns:
        dictionary with functions.
    """
    confidence_aggregation_bank = {"mean": lambda x: sum(x) / len(x), "min": min, "max": max}
    # python 3.7 and earlier do not have math.prod
    if hasattr(math, "prod"):
        confidence_aggregation_bank["prod"] = math.prod
    else:
        import operator
        from functools import reduce

        confidence_aggregation_bank["prod"] = lambda x: reduce(operator.mul, x, 1)
    return confidence_aggregation_bank


class ConfidenceMeasureMixin(ABC):
    """Confidence Measure Mixin class.

    It initializes per-frame confidence measure.
    """

    def _init_confidence_measure(self, confidence_measure_cfg: Optional[DictConfig] = None):
        """Initialize per-frame confidence measure from config.
        """
        # this ensures that post_init check is always executed
        confidence_measure_cfg = OmegaConf.structured(
            ConfidenceMeasureConfig()
            if confidence_measure_cfg is None
            else ConfidenceMeasureConfig(**confidence_measure_cfg)
        )

        # set confidence calculation measure
        # we suppose that self.blank_id == len(vocabulary)
        self.num_tokens = (self.blank_id if hasattr(self, "blank_id") else self._blank_index) + 1
        self.temperature = confidence_measure_cfg.temperature

        # init confidence measure bank
        self.confidence_measure_bank = get_confidence_measure_bank()

        measure = None
        # construct measure_name
        measure_name = ""
        if confidence_measure_cfg.name == "max_prob":
            measure_name = "max_prob"
        elif confidence_measure_cfg.name == "entropy":
            measure_name = '_'.join(
                [confidence_measure_cfg.name, confidence_measure_cfg.entropy_type, confidence_measure_cfg.entropy_norm]
            )
        else:
            raise ValueError(f"Unsupported `confidence_measure_cfg.name`: `{confidence_measure_cfg.name}`")
        if measure_name not in self.confidence_measure_bank:
            raise ValueError(f"Unsupported measure setup: `{measure_name}`")
        measure = partial(self.confidence_measure_bank[measure_name], v=self.num_tokens, t=self.temperature)
        self._get_confidence = lambda x: measure(torch.nan_to_num(x)).tolist()


class ConfidenceMixin(ABC):
    """Confidence Mixin class.

    It is responsible for confidence estimation method initialization and high-level confidence score calculation.
    """

    def _init_confidence(self, confidence_cfg: Optional[DictConfig] = None):
        """Initialize confidence-related fields and confidence aggregation function from config.
        """
        # this ensures that post_init check is always executed
        confidence_cfg = OmegaConf.structured(
            ConfidenceConfig() if confidence_cfg is None else ConfidenceConfig(**confidence_cfg)
        )

        # extract the config
        self.preserve_word_confidence = confidence_cfg.get('preserve_word_confidence', False)
        # set preserve_frame_confidence and preserve_token_confidence to True
        # if preserve_word_confidence is True
        self.preserve_token_confidence = (
            confidence_cfg.get('preserve_token_confidence', False) | self.preserve_word_confidence
        )
        # set preserve_frame_confidence to True if preserve_token_confidence is True
        self.preserve_frame_confidence = (
            confidence_cfg.get('preserve_frame_confidence', False) | self.preserve_token_confidence
        )
        self.exclude_blank_from_confidence = confidence_cfg.get('exclude_blank', True)
        self.word_confidence_aggregation = confidence_cfg.get('aggregation', "min")
        self.confidence_measure_cfg = confidence_cfg.get('measure_cfg', None)

        # define aggregation functions
        self.confidence_aggregation_bank = get_confidence_aggregation_bank()
        self._aggregate_confidence = self.confidence_aggregation_bank[self.word_confidence_aggregation]

        # Update preserve frame confidence
        if self.preserve_frame_confidence is False:
            if self.cfg.strategy in ['greedy', 'greedy_batch']:
                self.preserve_frame_confidence = self.cfg.greedy.get('preserve_frame_confidence', False)
                self.confidence_measure_cfg = self.cfg.greedy.get('confidence_measure_cfg', None)

    @abstractmethod
    def compute_confidence(self, hypotheses_list: List[Hypothesis]) -> List[Hypothesis]:
        """Computes high-level (per-token and/or per-word) confidence scores for a list of hypotheses.
        Assumes that `frame_confidence` is present in the hypotheses.

        Args:
            hypotheses_list: List of Hypothesis.

        Returns:
            A list of hypotheses with high-level confidence scores.
        """
        raise NotImplementedError()

    @abstractmethod
    def _aggregate_token_confidence(self, hypothesis: Hypothesis) -> List[float]:
        """Implemented by subclass in order to aggregate token confidence to a word-level confidence.

        Args:
            hypothesis: Hypothesis

        Returns:
            A list of word-level confidence scores.
        """
        raise NotImplementedError()

    def _aggregate_token_confidence_chars(self, words: List[str], token_confidence: List[float]) -> List[float]:
        """Implementation of token confidence aggregation for character-based models.

        Args:
            words: List of words of a hypothesis.
            token_confidence: List of token-level confidence scores of a hypothesis.

        Returns:
            A list of word-level confidence scores.
        """
        word_confidence = []
        i = 0
        for word in words:
            word_len = len(word)
            word_confidence.append(self._aggregate_confidence(token_confidence[i : i + word_len]))
            # we assume that there is exactly one space token between words and exclude it from word confidence
            i += word_len + 1
        return word_confidence

    def _aggregate_token_confidence_subwords_sentencepiece(
        self, words: List[str], token_confidence: List[float], token_ids: List[int]
    ) -> List[float]:
        """Implementation of token confidence aggregation for subword-based models.

        **Note**: Only supports Sentencepiece based tokenizers !

        Args:
            words: List of words of a hypothesis.
            token_confidence: List of token-level confidence scores of a hypothesis.
            token_ids: List of token ids of a hypothesis.

        Returns:
            A list of word-level confidence scores.
        """
        word_confidence = []
        # run only if there are final words
        if len(words) > 0:
            j = 0
            prev_unk = False
            prev_underline = False
            for i, token_id in enumerate(token_ids):
                token = self.decode_ids_to_tokens([int(token_id)])[0]
                token_text = self.decode_tokens_to_str([int(token_id)])
                # treat `<unk>` as a separate word regardless of the next token
                # to match the result of `tokenizer.ids_to_text`
                if (token != token_text or prev_unk) and i > j:
                    # do not add confidence for `▁` if the current token starts with `▁`
                    # to match the result of `tokenizer.ids_to_text`
                    if not prev_underline:
                        word_confidence.append(self._aggregate_confidence(token_confidence[j:i]))
                    j = i
                prev_unk = token == '<unk>'
                prev_underline = token == '▁'
            if not prev_underline:
                word_confidence.append(self._aggregate_confidence(token_confidence[j : len(token_ids)]))
        if len(words) != len(word_confidence):
            raise RuntimeError(
                f"""Something went wrong with word-level confidence aggregation.\n
            Please check these values for debugging:\n
            len(words): {len(words)},\n
            len(word_confidence): {len(word_confidence)},\n
            recognized text: `{' '.join(words)}`"""
            )
        return word_confidence
