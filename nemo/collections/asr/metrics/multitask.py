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

import operator
from collections import defaultdict
from functools import partial
from typing import Optional

import regex as re
import torch
from lhotse import CutSet
from lhotse.cut import MixedCut
from omegaconf import DictConfig, OmegaConf
from torch import nn

from nemo.collections.asr.data.audio_to_text_lhotse_prompted import PromptedAudioToTextMiniBatch
from nemo.collections.asr.metrics.wer import WER
from nemo.core.classes import Serialization

__all__ = ['MultiTaskMetric']


# Helper functions for managing constraint criteria on metrics.
def _logical_not(expr, properties):
    return not expr(properties)


def _logical_and(l_expr, r_expr, properties):
    return l_expr(properties) and r_expr(properties)


def _logical_or(l_expr, r_expr, properties):
    return l_expr(properties) or r_expr(properties)

def _no_constraint(properties):
    return True

def _static_constraint(fnc, key, val, properties):
    return fnc(val, properties.get(key))


def _compare_constraint(fnc, key1, key2, properties):
    return (
        (prop_val1 := properties.get(key1)) is not None
        and (prop_val2 := properties.get(key2)) is not None
        and fnc(prop_val1, prop_val2)
    )


# Basic operators for comparison. Add more as necessary.
operators = {
    "==": operator.eq,
    "!=": operator.ne,
}


def _build_constraint_fn(constraint: str):
    """
    Parse a constraint string and build a callable constraint function.

    Supports the following constraint syntax:
    - Simple comparisons: ".task == transcribe", ".lang != en"
    - Property comparisons: ".source_lang == .target_lang"
    - Logical operations: "not .task == translate", ".task == transcribe and .lang == en"
    - Complex expressions: ".task == transcribe or .task == translate and .source_lang != .target_lang"
    
    Args:
        constraint (str): Constraint expression string

    Returns:
        callable: Function that takes properties dict and returns bool

    Raises:
        AssertionError: If constraint syntax is invalid

    Examples:
        >>> fn = _build_constraint_fn(".task == transcribe")
        >>> fn({"task": "transcribe"})  # True
        >>> fn({"task": "translate"})   # False

        >>> fn = _build_constraint_fn(".task == transcribe and .lang == en")
        >>> fn({"task": "transcribe", "lang": "en"})  # True
        >>> fn({"task": "transcribe", "lang": "de"})  # False
    """
    c = constraint.strip()

    # Dummy function to return no constraint (i.e. True).
    if not constraint:
        return _no_constraint

    # Basic boolean recursion precedence.
    pattern = r'not\s+(.+)'  # not
    match = re.match(pattern, c)
    if match:
        expr = match.group(1).strip()
        return partial(_logical_not, _build_constraint_fn(expr))

    pattern = fr'(.+?)\s+and\s+(.+)'  # and
    match = re.match(pattern, c)
    if match:
        left_expr, right_expr = match.groups()
        return partial(_logical_and, _build_constraint_fn(left_expr), _build_constraint_fn(right_expr))

    pattern = fr'(.+?)\s+or\s+(.+)'  # or
    match = re.match(pattern, c)
    if match:
        left_expr, right_expr = match.groups()
        return partial(_logical_or, _build_constraint_fn(left_expr), _build_constraint_fn(right_expr))

    # Check for compare custom against defined value.
    for n, o in operators.items():
        pattern = fr'\.(\S+)\s*{n}\s*(\S+)'
        match = re.match(pattern, c)
        if match:
            key1, val = match.groups()
            return partial(_static_constraint, o, key1, val)

    # Check for compare custom against custom
    for n, o in operators.items():
        pattern = fr'\.(\S+)\s*{n}\s*\.(\S+)'
        match = re.match(pattern, c)
        if match:
            key1, key2 = match.groups()
            return partial(_compare_constraint, o, key1, key2)

    assert False, f"Constraint {c} cannot be resolved by `MultiTaskMetric`."


class MultiTaskMetric(Serialization):
    """
    Wrapper class for managing multiple metrics in multitask ASR/NLP models.

    This class enables conditional metric computation based on sample properties stored in Lhotse cuts.
    It's primarily designed for `EncDecMultiTaskModel` but can support any model with a prompt schema.

    Key Features:
        1. **Automatic Model Integration**: Instantiated metrics are automatically added as attributes
           to the parent model, enabling seamless integration with existing logging infrastructure.

        2. **Conditional Metric Updates**: Only samples meeting specific constraints are passed to
           each metric, avoiding inappropriate metric calculations (e.g., WER for translation tasks).

        3. **Flexible Constraint System**: Supports complex logical expressions for determining
           when metrics should be applied to samples.

        4. **Configuration Inheritance**: Global configuration parameters are automatically
           inherited by all metrics unless explicitly overridden.

    Args:
        model (nn.Module): Parent model that will receive metric instances as attributes.
                          Must have a `decoding` attribute for metrics that require decoding.
        cfg (DictConfig): Configuration dictionary containing metric definitions and constraints.

    Configuration Format:
        The configuration should follow this structure:

        ``'
        # Global parameters (inherited by all metrics unless overridden)
        log_predictions: true
        batch_dim_index: 0

        # Metric definitions
        metrics:
          - name: wer                                    # Metric name (becomes model attribute)
            _target_: nemo.collections.asr.metrics.WER  # Metric class to instantiate
            constraint: ".task == transcribe"           # When to apply this metric
            use_cer: false                              # Metric-specific parameters

          - name: bleu
            _target_: nemo.collections.asr.metrics.BLEU
            constraint: ".task == translate"
            bleu_tokenizer: "13a"
            n_gram: 4

          - name: multilingual_wer
            _target_: nemo.collections.asr.metrics.WER
            constraint: ".task == transcribe and .lang != en"
            use_cer: true
        ```

    Constraint Syntax:
        Constraints are evaluated against the `custom` dictionary of Lhotse cuts:

        - **Custom attribute Access**: `.task`, `.lang`, `.domain`
        - **Comparisons**: `==`, `!=`
        - **Logical Operations**: `and`, `or`, `not`
        - **Property Comparisons**: `.source_lang == .target_lang`

        Examples:
        - `".task == transcribe"` - Apply to transcription tasks
        - `".task == translate and .source_lang != .target_lang"` - Cross-lingual translation
        - `"not .task == other"` - Apply to all tasks except 'other'
        - `".domain == medical or .domain == legal"` - Specific domains

    Usage Example:
        ```python
        # In model initialization
        if hasattr(cfg, 'multitask_metrics'):
            self.multitask_metrics = MultiTaskMetric(self, cfg.multitask_metrics)

        # During training/validation
        if hasattr(self, 'multitask_metrics'):
            metrics = self.multitask_metrics.eval(
                batch=batch,
                predictions=predictions,
                predictions_lengths=pred_lengths,
                predictions_mask=pred_mask,
                prefix="val",
                return_all_metrics=True
            )
            self.log_dict(metrics)
        ```

    Note:
        - Each metric receives the model's `decoding` instance for text decoding operations
        - Metrics are automatically added to the model as attributes (e.g., `model.wer`, `model.bleu`)
        - Global configuration parameters are inherited unless explicitly overridden per metric
        - Metrics defined without 'constraint' keyword are called on every prediction sample
        - Empty batches (no samples matching constraints) are handled by children metrics.
    """

    def __init__(self, model: nn.Module, cfg: DictConfig):
        """
        Initialize MultiTaskMetric with model and configuration.

        Args:
            model (nn.Module): Parent model that will contain metric instances
            cfg (DictConfig): Configuration containing metric definitions
        """
        super().__init__()

        # Setup tracking dictionaries
        self._metric_dict, self._constr_dict = {}, {}
        cfg = OmegaConf.to_container(cfg)
<<<<<<< HEAD
        
        # Process each metric instance.
        seen_types = set()
=======

        # Process each metric definition
>>>>>>> 5da09f8a74bd6c6f77f4fa339e2cb52e5b6f84ab
        for metric in cfg.pop("metrics"):
            name, constraint = metric.pop("name"), metric.pop("constraint", "")  # Empty string for no constraint value. Metric always calculated.

            # Inherit global configuration parameters
            for k, v in cfg.items():
                if k not in metric:  # do not override explicit metric values
                    metric[k] = v

            # Instantiates as instance of `model`. Avoids breaking behavior when other modules call specific metrics. (See `asr_model` for example.)
            metric["decoding"] = model.decoding  # For decoding reliant metrics like 'WER' or 'BLEU'
            metric = MultiTaskMetric.from_config_dict(metric)
            setattr(model, name, metric)

            # TODO: This is a limitation brought upon by `asr_model` aggregation. To fix, update metric classes to support custom naming
            # and update `asr_model` `multi_{validation,test}_epoch_end` to support metric aggregation with custom names.
            metric_type = type(metric)
            if metric_type in seen_types:
                raise TypeError("MultiTaskMetric currently only supports one instance of each metric class. Please check your configs for duplicates values of `_target_` entry.")
            seen_types.add(metric_type)

            # Store metric and its constraint function
            self._metric_dict[name] = metric
            self._constr_dict[name] = _build_constraint_fn(constraint)

    # Performs full PyMetrics validation loop for all metrics.
    def eval(
        self,
        batch: PromptedAudioToTextMiniBatch,
        predictions: torch.Tensor,
        predictions_lengths: torch.Tensor,
        predictions_mask: torch.Tensor,
        return_all_metrics: Optional[bool] = False,
        prefix: Optional[str] = None,
    ):
        metric_dict = {}
        self.update(
            predictions=predictions,
            predictions_lengths=predictions_lengths,
            targets=batch.transcript,
            targets_lengths=batch.transcript_lens,
            predictions_mask=predictions_mask,
            input_ids=getattr(batch, "prompt", None),  # Allows for CTC and RNN-T support.
            cuts=batch.cuts,
        )
        metric_dict.update(
            self.compute(
                prefix=f"{prefix}_" if prefix else "",
                return_all_metrics=return_all_metrics,
            )
        )
        self.reset()
        return metric_dict

    def update(
        self,
        predictions: torch.Tensor,
        predictions_lengths: torch.Tensor,
        predictions_mask: torch.Tensor,
        targets: torch.Tensor,
        targets_lengths: torch.Tensor,
        input_ids: torch.Tensor,
        cuts: CutSet,
    ):
<<<<<<< HEAD
        
=======
        cuts_split, idx_split = self._split_cuts(cuts)

>>>>>>> 5da09f8a74bd6c6f77f4fa339e2cb52e5b6f84ab
        # Update each metric with its filtered data
        cuts_split, idx_split = self._split_cuts(cuts)
        for name, metric in self._metric_dict.items():
            cuts_subset, indices = cuts_split[name], idx_split[name]
<<<<<<< HEAD
=======

>>>>>>> 5da09f8a74bd6c6f77f4fa339e2cb52e5b6f84ab
            # Update metric with filtered tensors
            metric.update(
                predictions=predictions[indices],
                predictions_lengths=predictions_lengths[indices],
                predictions_mask=predictions_mask[indices],
                targets=targets[indices],
                targets_lengths=targets_lengths[indices],
                input_ids=input_ids[indices],
                cuts=cuts_subset,
            )

    def compute(self, return_all_metrics=False, prefix=""):
        output_dict = {}

        for name, metric in self._metric_dict.items():
            # Handle WER metric's special return format
            # Custom name of metric used as suffix to allow custom naming.
            # TODO: Standardize WER to return dict like other metrics
            if type(metric) is WER:
                wer, wer_num, wer_denom = metric.compute()
                if return_all_metrics:
<<<<<<< HEAD
                    output_dict.update({
                        f"{prefix}wer": wer,
                        f"{prefix}wer_num": wer_num,
                        f"{prefix}wer_denom": wer_denom,
                    })
                else:
                    output_dict.update({
                        f"{prefix}wer": wer,
                    })
=======
                    output_dict.update(
                        {
                            f"{prefix}wer{suffix}": wer,
                            f"{prefix}wer_num{suffix}": wer_num,
                            f"{prefix}wer_denom{suffix}": wer_denom,
                        }
                    )
                else:
                    output_dict.update(
                        {
                            f"{prefix}wer{suffix}": wer,
                        }
                    )
>>>>>>> 5da09f8a74bd6c6f77f4fa339e2cb52e5b6f84ab
            else:
                # Standard metric compute (returns dict)
                output_dict.update(
                    metric.compute(
                        return_all_metrics=return_all_metrics,
                        prefix=prefix,
                    )
                )
        return output_dict

    def reset(self):
        {metric.reset() for name, metric in self._metric_dict.items()}

    def _split_cuts(self, cuts):
        """
        Split cuts based on metric constraints and return filtered subsets.

        This method evaluates each cut against all metric constraints and creates
        separate lists of cuts and indices for each metric.

        Args:
            cuts (CutSet): Input cuts containing sample metadata

        Returns:
            tuple: (cuts_splits, idx_splits) where:
                - cuts_splits (dict): Maps metric names to lists of matching cuts
                - idx_splits (dict): Maps metric names to lists of matching indices

        Note:
            - Handles both regular cuts and MixedCuts (uses first_non_padding_cut)
            - A single cut may match multiple metrics
            - Cuts not matching any constraints are ignored
        """
        cuts_splits, idx_splits = defaultdict(list), defaultdict(list)
        for idx, c in enumerate(cuts):
            c = c.first_non_padding_cut if isinstance(c, MixedCut) else c
            for metric in self._metric_dict:
                if self._constr_dict[metric](c.custom):
                    cuts_splits[metric].append(c)
                    idx_splits[metric].append(idx)
        return cuts_splits, idx_splits
