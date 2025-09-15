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


from typing import List, Tuple

import torch
from torch import Tensor

from nemo.collections.asr.inference.asr.asr_inference import ASRInference
from nemo.collections.asr.models import EncDecHybridRNNTCTCModel, EncDecRNNTModel


class RNNTInference(ASRInference):

    def __post_init__(self) -> None:
        """
        Additional post initialization step
        Checks if the model is a rnnt model and sets the decoding strategy to rnnt.
        """
        if not isinstance(self.asr_model, (EncDecRNNTModel, EncDecHybridRNNTCTCModel)):
            raise ValueError(
                "Provided model is not a RNNT type. You are trying to use a RNNT transcriber with a non-RNNT model."
            )

        decoder_type = 'rnnt'
        if isinstance(self.asr_model, EncDecHybridRNNTCTCModel):
            self.asr_model.cur_decoder = decoder_type

        # reset the decoding strategy
        self.reset_decoding_strategy(decoder_type)
        self.set_decoding_strategy(decoder_type)

    def get_blank_id(self) -> int:
        """
        Returns:
            (int) blank id for the model.
        """
        blank_id = len(self.asr_model.joint.vocabulary)
        return blank_id

    def get_vocabulary(self) -> List[str]:
        """
        Returns:
            (List[str]) list of vocabulary tokens.
        """
        return self.asr_model.joint.vocabulary

    def get_subsampling_factor(self) -> int:
        """
        Returns:
            (int) subsampling factor for the ASR encoder model.
        """
        return self.asr_model.encoder.subsampling_factor

    def encode(self, processed_signal: Tensor, processed_signal_length: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Get encoder output from the model. It is used for streaming inference.
        Args:
            processed_signal: (Tensor) processed signal. Shape is torch.Size([B, C, T]).
            processed_signal_length: (Tensor) processed signal length. Shape is torch.Size([B]).
        Returns:
            encoder_output: (Tensor) encoder output. Shape is torch.Size([B, T, D]).
        """
        if processed_signal.device != self.device:
            processed_signal = processed_signal.to(self.device)

        if processed_signal_length.device != self.device:
            processed_signal_length = processed_signal_length.to(self.device)

        with (
            torch.amp.autocast(device_type=self.device_str, dtype=self.compute_dtype, enabled=self.use_amp),
            torch.inference_mode(),
            torch.no_grad(),
        ):

            forward_outs = self.asr_model(
                processed_signal=processed_signal, processed_signal_length=processed_signal_length
            )

        encoded, encoded_len = forward_outs
        return encoded, encoded_len

    def decode(self, encoded: Tensor, encoded_len: Tensor, partial_hypotheses: List) -> List:
        """
        RNNT decoding function
        Args:
            encoded: (Tensor) encoder output.
            encoded_len: (Tensor) encoder output length.
            partial_hypotheses: (List) list of partial hypotheses for stateful decoding.
        Returns:
            (List) list of best hypotheses.
        """
        best_hyp = self.asr_model.decoding.rnnt_decoder_predictions_tensor(
            encoded, encoded_len, return_hypotheses=True, partial_hypotheses=partial_hypotheses
        )
        return best_hyp
