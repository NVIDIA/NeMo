# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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

from typing import Dict, List, Tuple

import torch

from nemo.core.classes.mixins import AccessMixin


class InterCTCMixin:
    """Adds utilities for computing interCTC loss from https://arxiv.org/abs/2102.03216.

    To use, make sure encoder accesses ``interctc['capture_layers']``
    property in the AccessMixin and registers ``interctc/layer_output_X`` and
    ``interctc/layer_length_X`` for all layers that we want to get loss from.
    Additionally, specify the following config parameters to set up loss::

        interctc:
            # can use different values
            loss_weights: [0.3]
            apply_at_layers: [8]

    Then call

        * ``self.setup_interctc`` in the init method
        * ``self.add_interctc_losses`` after computing regular loss.
        * ``self.finalize_interctc_metrics(metrics, outputs, prefix="val_")``
          in the `multi_validation_epoch_end` method.
        * ``self.finalize_interctc_metrics(metrics, outputs, prefix="test_")``
          in the `multi_test_epoch_end` method.
    """

    def _process_config_values(
        self, 
        a_loss_weights: List[float], 
        a_apply_at_layers: List[int],
        v_loss_weights: List[float], 
        v_apply_at_layers: List[int],
        av_loss_weights: List[float], 
        av_apply_at_layers: List[int],
        main_loss_weight=None,
    ):
        self._a_intermediate_loss_weights = a_loss_weights
        self._a_apply_at_layers = a_apply_at_layers

        self._v_intermediate_loss_weights = v_loss_weights
        self._v_apply_at_layers = v_apply_at_layers

        self._av_intermediate_loss_weights = av_loss_weights
        self._av_apply_at_layers = av_apply_at_layers

        if main_loss_weight is None:
            self._main_loss_weight = 1.0 - (0.5 * sum(self._a_intermediate_loss_weights) + 0.5 * sum(self._v_intermediate_loss_weights) + sum(self._av_intermediate_loss_weights))
            if self._main_loss_weight <= 0.0:
                raise ValueError(
                    "Make sure that sum of intermediate loss weights is < 1.0. "
                    "Note that we don't do any normalization and assign "
                    "remaining weight to the regular model loss. "
                    "E.g., if interctc.loss_weights = [0.1, 0.3], regular "
                    "loss will have weight of 0.6"
                )
        else:
            self._main_loss_weight = main_loss_weight
        self._interctc_enabled = len(self._a_intermediate_loss_weights) > 0 or len(self._v_intermediate_loss_weights) > 0 or len(self._av_intermediate_loss_weights) > 0

        if len(self._a_apply_at_layers) != len(self._a_intermediate_loss_weights) or len(self._v_apply_at_layers) != len(self._v_intermediate_loss_weights) or len(self._av_apply_at_layers) != len(self._av_intermediate_loss_weights):
            raise ValueError('Length of interctc.apply_at_layers has to match interctc.loss_weights')

        # setting up config for AccessMixin that will be checked in encoders to
        # log the layers we need
        AccessMixin.update_access_cfg({'interctc': {'capture_layers': list(set(self._a_apply_at_layers  + self._v_apply_at_layers + self._av_apply_at_layers))}})

    def setup_interctc(self):
        """Sets up all interctc-specific parameters and checks config consistency."""
        interctc_config = self.cfg.get("interctc")
        if interctc_config is not None:
            # if interctc is in the config, we want to check that it indeed defines
            # the required keys and nothing else - that's automatically done by
            # matching with keyword arguments in self._process_config_values
            self._process_config_values(**interctc_config)
        else:
            self._interctc_enabled = False

    def _verify_setup_was_called(self):
        """Can be used to verify if setup_interctc was called."""
        if not hasattr(self, '_interctc_enabled'):
            raise RuntimeError('self.setup_interctc() has to be called before InterCTC loss can be used!')

    def is_interctc_enabled(self) -> bool:
        """Returns whether interCTC loss is enabled."""
        self._verify_setup_was_called()
        return self._interctc_enabled

    def set_interctc_enabled(self, enabled: bool):
        """Can be used to enable/disable InterCTC manually."""
        self._verify_setup_was_called()
        if enabled:  # checking if proper config parameters were specified
            if len(self._a_intermediate_loss_weights) == 0 and len(self._v_intermediate_loss_weights) == 0 and len(self._av_intermediate_loss_weights) == 0:
                raise RuntimeError(
                    'InterCTC cannot be enabled since interctc.loss_weights was not specified in the config.'
                )
            if len(self._a_apply_at_layers) != len(self._a_intermediate_loss_weights) or len(self._v_apply_at_layers) != len(self._v_intermediate_loss_weights) or len(self._av_apply_at_layers) != len(self._av_intermediate_loss_weights):
                raise RuntimeError(
                    'InterCTC cannot be enabled, since length of "loss_weights" does not match "apply_at_layers".'
                )
        self._interctc_enabled = enabled

    def finalize_interctc_metrics(self, metrics: Dict, outputs: List[Dict], prefix: str):
        """Finalizes InterCTC WER and loss metrics for logging purposes.

        Should be called inside ``multi_validation_epoch_end`` (with ``prefix="val_"``) or
        ``multi_test_epoch_end`` (with ``prefix="test_"``).

        Note that ``metrics`` argument is going to be updated in-place.
        """

        
        if self.is_interctc_enabled():

            # Audio
            for layer_idx in self._a_apply_at_layers:
                loss = torch.stack([x[f"{prefix}inter_ctc_loss_l{layer_idx}_a"] for x in outputs]).mean()
                wer_num = torch.stack([x[f"{prefix}inter_wer_num_l{layer_idx}_a"] for x in outputs]).sum()
                wer_denom = torch.stack([x[f"{prefix}inter_wer_denom_l{layer_idx}_a"] for x in outputs]).sum()
                metrics["log"].update(
                    {
                        f"{prefix}inter_ctc_loss_l{layer_idx}_a": loss,
                        f"{prefix}inter_wer_l{layer_idx}_a": wer_num / wer_denom,
                    }
                )

            # Visual
            for layer_idx in self._v_apply_at_layers:
                loss = torch.stack([x[f"{prefix}inter_ctc_loss_l{layer_idx}_v"] for x in outputs]).mean()
                wer_num = torch.stack([x[f"{prefix}inter_wer_num_l{layer_idx}_v"] for x in outputs]).sum()
                wer_denom = torch.stack([x[f"{prefix}inter_wer_denom_l{layer_idx}_v"] for x in outputs]).sum()
                metrics["log"].update(
                    {
                        f"{prefix}inter_ctc_loss_l{layer_idx}_v": loss,
                        f"{prefix}inter_wer_l{layer_idx}_v": wer_num / wer_denom,
                    }
                )

            # Audio Visual
            for layer_idx in self._av_apply_at_layers:
                loss = torch.stack([x[f"{prefix}inter_ctc_loss_l{layer_idx}_av"] for x in outputs]).mean()
                wer_num = torch.stack([x[f"{prefix}inter_wer_num_l{layer_idx}_av"] for x in outputs]).sum()
                wer_denom = torch.stack([x[f"{prefix}inter_wer_denom_l{layer_idx}_av"] for x in outputs]).sum()
                metrics["log"].update(
                    {
                        f"{prefix}inter_ctc_loss_l{layer_idx}_av": loss,
                        f"{prefix}inter_wer_l{layer_idx}_av": wer_num / wer_denom,
                    }
                )

            # Final
            metrics["log"][f"{prefix}final_ctc_loss"] = torch.stack(
                [x[f"{prefix}final_ctc_loss"] for x in outputs]
            ).mean()

    def get_captured_interctc_tensors(self) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """Returns a list of captured tensors from encoder: tuples of (output, length).

        Will additionally apply ``self.decoder`` to the outputs.
        """
        if not self.is_interctc_enabled():
            return []

        # Select ctc decocer
        if hasattr(self, "ctc_decoder"):
            decoder = self.ctc_decoder
        else:
            decoder = self.decoder

        # Audio

        # note that we have a loop here, because tensors can be defined from
        # submodules of encoder (e.g., that's the case in Jasper)
        total_registry = {}
        for module_registry in AccessMixin.get_module_registry(self.audio_back_end).values():
            for key, value in module_registry.items():
                if key.startswith("interctc/") and key in total_registry:
                    raise RuntimeError(f"layer {key} has been logged multiple times!")
            total_registry.update(module_registry)
        # if intermediate_loss_weights was set, the encoder has to register
        # interctc/layer_output_X and interctc/layer_length_X tensors.
        # We need to apply decoder to each of them and compute CTC loss.
        a_captured_tensors = []
        for layer_idx in self._a_apply_at_layers:
            try:
                layer_outputs = total_registry[f"interctc/layer_output_{layer_idx}"]
                layer_lengths = total_registry[f"interctc/layer_length_{layer_idx}"]
            except KeyError:
                raise RuntimeError(
                    f"Intermediate layer {layer_idx} was not captured! "
                    "Check if length of model.encoder.captured_layer_outputs matches "
                    "length of model.intermediate_loss_weights properties."
                )
            if len(layer_outputs) > 1 or len(layer_lengths) > 1:
                raise RuntimeError(
                    "Make sure encoder.forward is called exactly one time before interCTC loss is computed."
                )
            a_captured_tensors.append((decoder(encoder_output=layer_outputs[0]), layer_lengths[0]))

        # Visual

        # note that we have a loop here, because tensors can be defined from
        # submodules of encoder (e.g., that's the case in Jasper)
        total_registry = {}
        for module_registry in AccessMixin.get_module_registry(self.video_back_end).values():
            for key, value in module_registry.items():
                if key.startswith("interctc/") and key in total_registry:
                    raise RuntimeError(f"layer {key} has been logged multiple times!")
            total_registry.update(module_registry)
        # if intermediate_loss_weights was set, the encoder has to register
        # interctc/layer_output_X and interctc/layer_length_X tensors.
        # We need to apply decoder to each of them and compute CTC loss.
        v_captured_tensors = []
        for layer_idx in self._v_apply_at_layers:
            try:
                layer_outputs = total_registry[f"interctc/layer_output_{layer_idx}"]
                layer_lengths = total_registry[f"interctc/layer_length_{layer_idx}"]
            except KeyError:
                raise RuntimeError(
                    f"Intermediate layer {layer_idx} was not captured! "
                    "Check if length of model.encoder.captured_layer_outputs matches "
                    "length of model.intermediate_loss_weights properties."
                )
            if len(layer_outputs) > 1 or len(layer_lengths) > 1:
                raise RuntimeError(
                    "Make sure encoder.forward is called exactly one time before interCTC loss is computed."
                )
            v_captured_tensors.append((decoder(encoder_output=layer_outputs[0]), layer_lengths[0]))

        # Audio Visual

        # note that we have a loop here, because tensors can be defined from
        # submodules of encoder (e.g., that's the case in Jasper)
        total_registry = {}
        for module_registry in AccessMixin.get_module_registry(self.audio_visual_encoder).values():
            for key, value in module_registry.items():
                if key.startswith("interctc/") and key in total_registry:
                    raise RuntimeError(f"layer {key} has been logged multiple times!")
            total_registry.update(module_registry)
        # if intermediate_loss_weights was set, the encoder has to register
        # interctc/layer_output_X and interctc/layer_length_X tensors.
        # We need to apply decoder to each of them and compute CTC loss.
        av_captured_tensors = []
        for layer_idx in self._av_apply_at_layers:
            try:
                layer_outputs = total_registry[f"interctc/layer_output_{layer_idx}"]
                layer_lengths = total_registry[f"interctc/layer_length_{layer_idx}"]
            except KeyError:
                raise RuntimeError(
                    f"Intermediate layer {layer_idx} was not captured! "
                    "Check if length of model.encoder.captured_layer_outputs matches "
                    "length of model.intermediate_loss_weights properties."
                )
            if len(layer_outputs) > 1 or len(layer_lengths) > 1:
                raise RuntimeError(
                    "Make sure encoder.forward is called exactly one time before interCTC loss is computed."
                )
            av_captured_tensors.append((decoder(encoder_output=layer_outputs[0]), layer_lengths[0]))      

        return a_captured_tensors, v_captured_tensors, av_captured_tensors

    def add_interctc_losses(
        self,
        loss_value: torch.Tensor,
        transcript: torch.Tensor,
        transcript_len: torch.Tensor,
        compute_wer: bool,
        log_wer_num_denom: bool = False,
        log_prefix: str = "",
    ) -> Tuple[torch.Tensor, Dict]:
        """Adding interCTC losses if required.

        Will also register loss/wer metrics in the returned dictionary.

        Args:
            loss_value (torch.Tensor): regular loss tensor (will add interCTC loss to it).
            transcript (torch.Tensor): current utterance transcript.
            transcript_len (torch.Tensor): current utterance transcript length.
            compute_wer (bool): whether to compute WER for the current utterance.
                Should typically be True for validation/test and only True for
                training if current batch WER should be logged.
            log_wer_num_denom (bool): if True, will additionally log WER num/denom
                in the returned metrics dictionary. Should always be True for
                validation/test to allow correct metrics aggregation. Should
                always be False for training. Defaults to False.
            log_prefix (str): prefix added to all log values. Should be ``""`` for
                training and ``"val_"`` for validation.

        Returns:
            tuple[torch.Tensor, Dict]: tuple of new loss tensor and dictionary with logged metrics.
        """
        if not self.is_interctc_enabled() or not AccessMixin.is_access_enabled():
            return loss_value, {}
        metrics = {f"{log_prefix}final_ctc_loss": loss_value}
        a_captured_tensors, v_captured_tensors, av_captured_tensors = self.get_captured_interctc_tensors()

        # Weight main loss
        loss_value *= self._main_loss_weight

        # Select ctc loss
        if hasattr(self, "ctc_loss"):
            loss = self.ctc_loss
        else:
            loss = self.loss

        # Select ctc wer
        if hasattr(self, "ctc_wer"):
            _wer = self.ctc_wer
        else:
            _wer = self._wer

        # Audio
        for layer_idx, intermediate_result, loss_weight in zip(
            self._a_apply_at_layers, a_captured_tensors, self._a_intermediate_loss_weights
        ):
            if self.audio_dropped is None:
                inter_loss_value = loss(
                    log_probs=intermediate_result[0],
                    targets=transcript,
                    target_lengths=transcript_len,
                    input_lengths=intermediate_result[1],
                )
            else:
                loss_reduction = loss.reduction
                loss_apply_reduction = loss._apply_reduction
                loss.reduction = "none"
                loss._apply_reduction = False

                # Compute loss without reduction
                inter_loss_value = loss(
                    log_probs=intermediate_result[0],
                    targets=transcript,
                    target_lengths=transcript_len,
                    input_lengths=intermediate_result[1],
                )

                # Remove masked samples and apply Reduction
                inter_loss_value = inter_loss_value[self.audio_dropped.logical_not()]
                inter_loss_value = loss.reduce(inter_loss_value, transcript_len)

                loss.reduction = loss_reduction
                loss._apply_reduction = loss_apply_reduction

            
            metrics[f"{log_prefix}inter_ctc_loss_l{layer_idx}_a"] = inter_loss_value.detach()
            loss_value += inter_loss_value * loss_weight
            if compute_wer:
                _wer.update(
                    predictions=intermediate_result[0],
                    targets=transcript,
                    target_lengths=transcript_len,
                    predictions_lengths=intermediate_result[1],
                )
                wer, wer_num, wer_denom = _wer.compute()
                _wer.reset()
                metrics.update({f'{log_prefix}inter_wer_l{layer_idx}_a': wer})
                if log_wer_num_denom:
                    metrics.update(
                        {
                            f'{log_prefix}inter_wer_num_l{layer_idx}_a': wer_num,
                            f'{log_prefix}inter_wer_denom_l{layer_idx}_a': wer_denom,
                        }
                    )
            
        # Visual
        for layer_idx, intermediate_result, loss_weight in zip(
            self._v_apply_at_layers, v_captured_tensors, self._v_intermediate_loss_weights
        ):
            if self.video_dropped is None:
                inter_loss_value = loss(
                    log_probs=intermediate_result[0],
                    targets=transcript,
                    target_lengths=transcript_len,
                    input_lengths=intermediate_result[1],
                )
            else:
                loss_reduction = loss.reduction
                loss_apply_reduction = loss._apply_reduction
                loss.reduction = "none"
                loss._apply_reduction = False

                # Compute loss without reduction
                inter_loss_value = loss(
                    log_probs=intermediate_result[0],
                    targets=transcript,
                    target_lengths=transcript_len,
                    input_lengths=intermediate_result[1],
                )

                # Remove masked samples and apply Reduction
                inter_loss_value = inter_loss_value[self.video_dropped.logical_not()]
                inter_loss_value = loss.reduce(inter_loss_value, transcript_len)

                loss.reduction = loss_reduction
                loss._apply_reduction = loss_apply_reduction

            metrics[f"{log_prefix}inter_ctc_loss_l{layer_idx}_v"] = inter_loss_value.detach()
            loss_value += inter_loss_value * loss_weight
            if compute_wer:
                _wer.update(
                    predictions=intermediate_result[0],
                    targets=transcript,
                    target_lengths=transcript_len,
                    predictions_lengths=intermediate_result[1],
                )
                wer, wer_num, wer_denom = _wer.compute()
                _wer.reset()
                metrics.update({f'{log_prefix}inter_wer_l{layer_idx}_v': wer})
                if log_wer_num_denom:
                    metrics.update(
                        {
                            f'{log_prefix}inter_wer_num_l{layer_idx}_v': wer_num,
                            f'{log_prefix}inter_wer_denom_l{layer_idx}_v': wer_denom,
                        }
                    )

        # Audio Visual
        for layer_idx, intermediate_result, loss_weight in zip(
            self._av_apply_at_layers, av_captured_tensors, self._av_intermediate_loss_weights
        ):
            inter_loss_value = loss(
                log_probs=intermediate_result[0],
                targets=transcript,
                target_lengths=transcript_len,
                input_lengths=intermediate_result[1],
            )
            metrics[f"{log_prefix}inter_ctc_loss_l{layer_idx}_av"] = inter_loss_value.detach()
            loss_value += inter_loss_value * loss_weight
            if compute_wer:
                _wer.update(
                    predictions=intermediate_result[0],
                    targets=transcript,
                    target_lengths=transcript_len,
                    predictions_lengths=intermediate_result[1],
                )
                wer, wer_num, wer_denom = _wer.compute()
                _wer.reset()
                metrics.update({f'{log_prefix}inter_wer_l{layer_idx}_av': wer})
                if log_wer_num_denom:
                    metrics.update(
                        {
                            f'{log_prefix}inter_wer_num_l{layer_idx}_av': wer_num,
                            f'{log_prefix}inter_wer_denom_l{layer_idx}_av': wer_denom,
                        }
                    )

        # return total loss
        return loss_value, metrics
