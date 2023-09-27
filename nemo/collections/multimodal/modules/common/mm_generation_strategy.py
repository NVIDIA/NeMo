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

import abc
import warnings
from typing import List, Set, Tuple

import numpy as np
import torch

import nemo.collections.nlp.modules.common.text_generation_strategy as text_generation_strategy
from nemo.collections.nlp.modules.common.lm_utils import pad_batch
from nemo.collections.nlp.modules.common.megatron.utils import get_ltor_masks_and_position_ids

try:
    from apex.transformer.pipeline_parallel.utils import get_num_microbatches

    HAVE_APEX = True

except (ImportError, ModuleNotFoundError):
    HAVE_APEX = False

try:
    from megatron.core.pipeline_parallel.schedules import get_forward_backward_func

    HAVE_MEGATRON_CORE = True

except (ImportError, ModuleNotFoundError):

    HAVE_MEGATRON_CORE = False


# the text representation of eos_id, it applies for all tokenizers
END_OF_SEQ = '<|endoftext|>'


class MMGPTModelGenerationStrategy(text_generation_strategy.GPTModelTextGenerationStrategy):
    def __init__(self, model, **kwargs):
        super().__init__(model)

    def prepare_batch_at_step(
        self,
        tokens: torch.Tensor,
        maxlen: int,
        micro_batch_size: int,
        step: int,
        context_length: int,
        compute_attention_mask: bool = True,
        additional_tokens: torch.Tensor = torch.empty(0),
        additional_tokens_mask: torch.Tensor = torch.empty(0),
    ) -> Tuple[List[torch.Tensor], List[int]]:
        """
        generate the batch used in inference for each of the steps

        """
        if step == 0:
            # Allocate memory for the entire context.
            set_inference_key_value_memory = True
            tokens2use = tokens[:, :context_length]
            positions2use = self.position_ids[:, :context_length]
            additional_tokens2use = additional_tokens[:, :context_length, :]
            additional_tokens_mask2use = additional_tokens_mask[:, :context_length, :]

            # not using type2use. uncomment it if it is used
            # if type_ids is not None:
            #     types2use = type_ids[:, :context_length]
        else:
            # Set this to false so the memory is not reallocated.
            set_inference_key_value_memory = False
            tokens2use = tokens[:, context_length - 1].view(micro_batch_size, -1)
            positions2use = self.position_ids[:, context_length - 1].view(micro_batch_size, -1)
            additional_tokens2use = additional_tokens[:, context_length - 1, :].view(micro_batch_size, -1, self.model.num_audio_codebooks-1)
            additional_tokens_mask2use = additional_tokens_mask[:, context_length - 1, :].view(micro_batch_size, -1, self.model.num_audio_codebooks-1)
            # not using type2use. uncomment it if it is used
            # if type_ids is not None:
            #     types2use = type_ids[:, context_length - 1].view(batch_size, -1)

        """Prepare batch for each of the inference steps"""
        attention_mask_repeat = None
        if compute_attention_mask:
            attention_mask_repeat = torch.concat([self.attention_mask for _ in range(micro_batch_size)])

        setkey_value_array = torch.tensor(
            [set_inference_key_value_memory] * micro_batch_size, device=torch.cuda.current_device()
        )
        len_array = torch.tensor([maxlen] * micro_batch_size, device=torch.cuda.current_device())

        batch = [tokens2use, attention_mask_repeat, positions2use, setkey_value_array, len_array, additional_tokens2use, additional_tokens_mask2use]
        tensor_shape = [tokens2use.shape[1], micro_batch_size, self.model.cfg.hidden_size]
        return batch, tensor_shape




def model_inference_strategy_dispatcher(model, **args):

    from nemo.collections.multimodal.models.megatron_mm_gpt_sft_model import MegatronMMGPTSFTModel

    if isinstance(model, MegatronMMGPTSFTModel):
        return MMGPTModelGenerationStrategy(model, **args)
    else:
        return text_generation_strategy.model_inference_strategy_dispatcher(model, **args)


