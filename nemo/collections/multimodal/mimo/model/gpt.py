from typing import Callable, Optional, Tuple, Union

import torch
from megatron.core.inference_params import InferenceParams
from megatron.core.models.gpt.gpt_model import GPTModel as MCoreGPTModel
from torch import Tensor


class MimoGPTModel(MCoreGPTModel):
    from megatron.core.packed_seq_params import PackedSeqParams

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(
        self,
        input_ids: Tensor,
        position_ids: Tensor,
        attention_mask: Tensor,
        decoder_input: Tensor = None,
        labels: Tensor = None,
        inference_params: InferenceParams = None,
        packed_seq_params: PackedSeqParams = None,
        extra_block_kwargs: dict = None,
        runtime_gather_output: Optional[bool] = None,
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:

        original_post_process = self.post_process
        self.post_process = False

        try:
            hidden_states = super().forward(
                input_ids=input_ids,
                position_ids=position_ids,
                attention_mask=attention_mask,
                decoder_input=decoder_input,
                inference_params=inference_params,
                packed_seq_params=packed_seq_params,
                extra_block_kwargs=extra_block_kwargs,
                runtime_gather_output=runtime_gather_output,
            )
        finally:
            self.post_process = original_post_process

        output_weight = None
        if self.share_embeddings_and_output_weights:
            output_weight = self.shared_embedding_or_output_weight()

        logits, _ = self.output_layer(hidden_states, weight=output_weight, runtime_gather_output=runtime_gather_output)

        if labels is None:
            return logits.transpose(0, 1).contiguous(), hidden_states
        loss = self.compute_language_model_loss(labels, logits)
        return loss, hidden_states
