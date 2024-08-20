import torch
from dataclasses import dataclass
from typing import Callable, Literal, Optional
from nemo.collections.llm.gpt.model.base import GPTModel, gpt_forward_step, gpt_data_step
from megatron.core.transformer.transformer_config import TransformerConfig
from nemo.lightning import get_vocab_size, io
from megatron.core import parallel_state
from megatron.core.models.mamba.mamba_layer_specs import mamba_stack_spec
from megatron.core.models.mamba import MambaModel as MCoreMambaModel

@dataclass
class SSMConfig(TransformerConfig, io.IOMixin):
    # From megatron.core.models.mamba.mamba_model.MambaModel
    fp16_lm_cross_entropy: bool = False,
    parallel_output: bool = True,
    share_embeddings_and_output_weights: bool = False,
    num_layers: int = 2,
    mamba_ssm_ngroups: int = 8
    hybrid_attention_ratio: float = 0.0
    hybrid_mlp_ratio: float = 0.0
    hybrid_override_pattern: str = None
    post_process: bool = True,
    pre_process: bool = True
    seq_length: int = 2048
    params_dtype: torch.dtype = torch.bfloat16
    # Mamba with no attention has no need for position embeddings, so none is default
    position_embedding_type: Literal['learned_absolute', 'rope', 'none'] = 'none'
    rotary_percent: float = 1.0
    rotary_base: int = 10000
    seq_len_interpolation_factor: Optional[float] = None
    apply_rope_fusion: bool = True
    make_vocab_size_divisible_by: int = 128
    # fp32_residual_connections: bool = False

    # TODO: Move this to better places?
    get_attention_mask_from_fusion: bool = False
    
    forward_step_fn: Callable = gpt_forward_step
    data_step_fn: Callable = gpt_data_step

    def configure_model(self, tokenizer) -> "MCoreMambaModel":

        return MCoreMambaModel(
            self,
            mamba_stack_spec=mamba_stack_spec,
            vocab_size=get_vocab_size(self, tokenizer.vocab_size, self.make_vocab_size_divisible_by),
            max_sequence_length=self.seq_length,
            mamba_ssm_ngroups=self.mamba_ssm_ngroups,
            hybrid_attention_ratio=self.hybrid_attention_ratio,
            hybrid_mlp_ratio=self.hybrid_mlp_ratio,
            hybrid_override_pattern=self.hybrid_override_pattern,
            position_embedding_type=self.position_embedding_type,
            rotary_percent=self.rotary_percent,
            rotary_base=self.rotary_base,
            seq_len_interpolation_factor=self.seq_len_interpolation_factor,
            pre_process=parallel_state.is_pipeline_first_stage(),
            post_process=parallel_state.is_pipeline_last_stage(),
        )

class SSMModel(GPTModel):

    def forward(self, input_ids, position_ids=None, attention_mask=None, labels=None):
        attention_mask = None
        output_tensor = self.module(
            input_ids=input_ids, 
            position_ids=position_ids, 
            attention_mask=attention_mask, 
            labels=labels
        )
        return output_tensor


__all__ = [
    "SSMModel",
    "SSMConfig",
]
