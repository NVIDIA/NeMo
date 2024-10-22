# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

from argparse import Namespace
from typing import Dict, List

import torch
from megatron.core import tensor_parallel
from megatron.core.inference.model_inference_wrappers.abstract_model_inference_wrapper import (
    AbstractModelInferenceWrapper,
)


class VLMInferenceWrapper(AbstractModelInferenceWrapper):
    """Constructor for the model inference wrapper

    The wrapper prepares the model for inference, provides the required input
    data, and runs the forward pass

    Args:
        model (T5Model): The T5 model (MCore or legacy)
        args (Namespace): The command line arguments that were passed
    """

    def __init__(self, model, args: Namespace):
        super().__init__(model, args)

    def prep_model_for_inference(self, prompts_tokens: torch.Tensor, image_dict: List[Dict] = None):

        super().prep_model_for_inference(prompts_tokens=prompts_tokens)
        self.pixel_values = image_dict[0]["pixel_values"]
        self.aspect_ratio_ids = image_dict[0]["aspect_ratio_ids"]
        self.num_tiles = image_dict[0]["num_tiles"]
        seq_length = prompts_tokens.size(1)
        self.position_ids = (
            torch.arange(seq_length, dtype=torch.long, device=prompts_tokens.device)
            .unsqueeze(0)
            .expand_as(prompts_tokens)
        )

    def get_batch_for_context_window(self, context_start_position: int, context_end_position: int) -> List:
        tokens2use = self.prompts_tokens[:, :context_end_position]
        positions2use = self.position_ids[:, :context_end_position]
        data_at_step_idx = [tokens2use, positions2use]

        return data_at_step_idx

    def forward_pass_without_pipeline_parallel(self, inference_input: List) -> torch.Tensor:
        """Utility to carry out simple forward pass for TP or no model parallel models

        Runs a very simple forward pass for model. Used  in the case of models without
        any parallelism or only tensor parallelism.

        Args:
            inference_input (List): A list containg the inputs for the gpt
                model [tokens, position ids, attention mask]

        Returns:
            torch.Tensor: The output logits of shape [batch_size, seq_len, padded_vocab_size]
        """
        tokens2use, positions2use = inference_input
        logits = self.model(
            batch_images=self.pixel_values,
            batch_masks=[[[5, 512]]] if self.num_tiles is not None else None,
            num_chunks=self.num_tiles,
            aspect_ratio_ids=self.aspect_ratio_ids,
            tokens=tokens2use,
            position_ids=positions2use,
        )
        logits = tensor_parallel.gather_from_tensor_model_parallel_region(logits)

        return logits
