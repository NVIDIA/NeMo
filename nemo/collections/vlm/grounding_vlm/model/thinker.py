from dataclasses import dataclass
from typing import Optional, Union, Callable
import io

import torch
from torch import Tensor

from megatron.core.models.vision.multimodal_projector import MultimodalProjector
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.mlp import MLPSubmodules

from megatron.core.transformer.transformer_block import TransformerBlock
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_block import TransformerBlockSubmodules
from megatron.core.inference.contexts import BaseInferenceContext
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.utils import make_viewless_tensor
from megatron.core import tensor_parallel
from contextlib import nullcontext
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.fusions.fused_bias_dropout import get_bias_dropout_add
from megatron.core.fp8_utils import get_fp8_context
from megatron.core.transformer.transformer_layer import TransformerLayer, TransformerLayerSubmodules
from megatron.core.transformer.attention import CrossAttention, CrossAttentionSubmodules
from megatron.core.extensions.transformer_engine import (
    TELayerNormColumnParallelLinear,
    TEDotProductAttention,
    TERowParallelLinear,
)
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.wrapped_tensor import WrappedTensor


from nemo.collections.llm.fn.activation import quick_gelu
from nemo.collections.vlm.layer_specs import get_norm_mlp_module_spec_te

# define a dummy class for the thinker config (it is redefined later)
class ThinkingAttnRefineModule:
    pass

def get_layer_spec_thinker() -> ModuleSpec:
    """
    Cross-attention Transformer Layer Spec w/ TE Modules

    this forms the base transformer layer spec that is copied as many layers as needed by the thinker module
    """
    attn_mask_type = AttnMaskType.no_mask

    mlp = get_norm_mlp_module_spec_te()
    return ModuleSpec(
        module=TransformerLayer,
        submodules=TransformerLayerSubmodules(
            cross_attention=ModuleSpec(
                module=CrossAttention,
                params={"attn_mask_type": attn_mask_type},
                submodules=CrossAttentionSubmodules(
                    linear_q=TELayerNormColumnParallelLinear,
                    linear_kv=TELayerNormColumnParallelLinear,
                    core_attention=TEDotProductAttention,
                    linear_proj=TERowParallelLinear,
                ),
            ),
            cross_attn_bda=get_bias_dropout_add,
            pre_mlp_layernorm=IdentityOp,
            mlp=mlp,
            mlp_bda=get_bias_dropout_add,
        ),
    )

@dataclass
class ThinkingAttnRefineModuleConfig(TransformerConfig, io.IOMixin):
    """Configuration for the Thinking Attention Refinement Module.
    
    This module implements a cross attention transformer that allows the model to 
    refine its understanding by attending between vision features (from encoder) and 
    language features (from decoder's penultimate layer).
    
    The cross attention mechanism enables:
    1. Vision features to attend to relevant language tokens
    2. Language features to attend to relevant image patches
    """
    
    # Cross attention specific
    num_layers: int = 4
    hidden_size: int = 1280
    num_attention_heads: int = 16
    attention_dropout: float = 0.0
    ffn_hidden_size: int = 5120
    activation_func: Callable = quick_gelu
    kv_channels: int = 80
    num_query_groups: int = 16
    layernorm_zero_centered_gamma: bool = False
    apply_query_key_layer_scaling: bool = False

    # classification head config
    cls_head: ModuleSpec = None

    def configure_model(self) -> ThinkingAttnRefineModule:
        """Configure and return a ThinkingAttnRefineModule instance.
        
        This creates a cross-attention transformer model that refines the understanding
        between vision encoder features and <|img_think|> features.
        
        Returns:
            ThinkingAttnRefineModule: The configured cross-attention transformer model
        """
        transformer_layer_spec = self.transformer_layer_spec
        if not isinstance(transformer_layer_spec, ModuleSpec):
            from nemo.collections.vlm.grounding_vlm.model.thinker import get_layer_spec_thinker
            transformer_layer_spec = get_layer_spec_thinker()
            
        model = ThinkingAttnRefineModule(
            config=self,
            spec=transformer_layer_spec,
            pre_process=False,
            post_process=False,
        )
        
        return model

@dataclass
class ClassifierHeadModuleConfig(TransformerConfig, io.IOMixin):
    """
    Configuration for the ClassifierHeadModule.
    """
    hidden_size: int = 1280
    projector_config: TransformerConfig = None
    projector_submodules: MLPSubmodules = None

    def __post_init__(self):
        """Initialize default configurations if not provided."""
        super().__post_init__()
        
        if self.projector_config is None:
            # Default projector config for classification MLP
            self.projector_config = TransformerConfig(
                hidden_size=self.hidden_size,
                num_layers=2,  # Two layer MLP
                ffn_hidden_size=self.hidden_size * 4,  # Standard 4x expansion
                num_attention_heads=1,  # Not used for MLP
                attention_dropout=0.1,  # Some dropout for regularization
                hidden_dropout=0.1,
                layernorm_epsilon=1e-5,
                init_method=lambda x: torch.nn.init.trunc_normal_(x, mean=0.0, std=0.02),
                output_layer_init_method=lambda x: torch.nn.init.trunc_normal_(x, mean=0.0, std=0.02),
            )
            
        if self.projector_submodules is None:
            # Default to standard MLP modules
            self.projector_submodules = MLPSubmodules(
                linear_fc1=TELayerNormColumnParallelLinear,
                linear_fc2=TERowParallelLinear,
                dropout=torch.nn.Dropout,
            )

class ClassifierHeadModule(MegatronModule):
    """
    Simple MLP classifier head
    """
    def __init__(self, config: ClassifierHeadModuleConfig):
        super().__init__(config=config)
        self.cls_mlp = MultimodalProjector(config=config.projector_config, submodules=config.projector_submodules,
                                           projector_type='mlp', input_size=config.hidden_size)
        
    def forward(self, cls_embeddings: Tensor, think_states: Tensor, hidden_states_indices: Tensor):
        '''
        Given classification embeddings and think states, run classification head

        cls_embeddings: [b, c, h]
        think_states: [cu_num_thinking_tokens, h]
        hidden_states_indices: [cu_num_thinking_tokens, 2]  # [seq_idx, b_idx]

        '''
        cls_embeddings = self.cls_mlp(cls_embeddings)  # [b, c, h]
        # aggregate think_states by b_idx
        seq_idx, b_idx = hidden_states_indices
        batch_size = cls_embeddings.shape[0]
        ret = []
        for b in range(batch_size):
            b_idx_mask = b_idx == b
            # if there is no thinking tokens for this batch, skip
            if b_idx_mask.sum() == 0:
                ret.append(torch.zeros(cls_embeddings.shape[1]))
                continue
            # aggregate think states by b_idx
            b_think_states = think_states[b_idx_mask].mean(dim=0, keepdim=True) # [1, h]
            cls_logits = torch.matmul(cls_embeddings[b], b_think_states.T) # [c, 1]
            ret.append(cls_logits.squeeze(dim=-1))
        ret = torch.stack(ret, dim=0) # [b, c]
        return ret

@dataclass
class DetectionHeadModuleConfig(TransformerConfig, io.IOMixin):
    """
    Configuration for the DetectionHeadModule.
    
    This module implements a detection head that uses cross-attention to refine
    detection features using thinking tokens.
    """
    # Base config
    hidden_size: int = 1280
    
    # Projector config for detection MLP
    projector_config: TransformerConfig = None
    projector_submodules: MLPSubmodules = None
    
    # Cross attention transformer config
    transformer_config: TransformerConfig = None
    transformer_spec: ModuleSpec = None
    
    def __post_init__(self):
        """Initialize default configurations if not provided."""
        super().__post_init__()
        
        if self.projector_config is None:
            # Default projector config
            self.projector_config = TransformerConfig(
                hidden_size=self.hidden_size,
                num_layers=2,
                ffn_hidden_size=self.hidden_size * 4,
                num_attention_heads=1,  # Not used for MLP
                attention_dropout=0.0,
                hidden_dropout=0.0,
                layernorm_epsilon=1e-5,
            )
            
        if self.transformer_config is None:
            # Default transformer config with cross attention only
            self.transformer_config = TransformerConfig(
                num_layers=2,
                hidden_size=self.hidden_size,
                num_attention_heads=16,
                attention_dropout=0.0,
                hidden_dropout=0.0,
                ffn_hidden_size=self.hidden_size * 4,
                kv_channels=80,
                apply_query_key_layer_scaling=False,
                layernorm_epsilon=1e-5,
            )
            
        if self.transformer_spec is None:
            # Create transformer spec with cross attention only
            self.transformer_spec = ModuleSpec(
                module=TransformerBlock,
                submodules=TransformerBlockSubmodules(
                    cross_attention=ModuleSpec(
                        module=CrossAttention,
                        params={"attn_mask_type": AttnMaskType.no_mask},
                        submodules=CrossAttentionSubmodules(
                            linear_q=TELayerNormColumnParallelLinear,
                            linear_kv=TELayerNormColumnParallelLinear,
                            core_attention=TEDotProductAttention,
                            linear_proj=TERowParallelLinear,
                        ),
                    ),
                    cross_attn_bda=get_bias_dropout_add,
                    pre_mlp_layernorm=IdentityOp,
                    mlp=get_norm_mlp_module_spec_te(),
                    mlp_bda=get_bias_dropout_add,
                ),
            )

class DetectionHeadModule(MegatronModule):
    """
    Simple MLP detection head
    """
    def __init__(self, config: DetectionHeadModuleConfig):
        super().__init__(config=config)
        self.det_mlp = MultimodalProjector(config=config.projector_config, submodules=config.projector_submodules,
                                           projector_type='mlp', input_size=config.hidden_size)
        self.transformer = TransformerBlock(config=config.transformer_config, spec=config.transformer_spec)
        # detection head
        self.det_head = MultimodalProjector(config=config.projector_config, submodules=config.projector_submodules,
                                           projector_type='mlp', input_size=config.hidden_size)
        
    def forward(self, det_embeddings: Tensor, think_states: Tensor, instance_det_indices: Tensor, think_indices: Tensor, max_instances_per_batch: int):
        '''
        Given detection embeddings and think states, run detection head
        '''
        det_embeddings = self.det_mlp(det_embeddings)  # [cu_num_instances, h]
        # aggregate think_states by b_idx
        seq_idx, b_idx = think_indices.T
        batch_size = b_idx.max() + 1

        ret = torch.zeros(batch_size, max_instances_per_batch, 8)

        for b in range(batch_size):
            b_think_mask = b_idx == b
            b_det_mask = instance_det_indices[:, 1] == b
            # if there is no thinking tokens for this batch, skip
            if b_think_mask.sum() == 0 or b_det_mask.sum() == 0:
                continue
            # update ret
            b_think_states = think_states[b_think_mask].unsqueeze(1)   # [s, 1, h]
            b_det_states = det_embeddings[b_det_mask].unsqueeze(0)   # [si, 1, h]
            b_ret = self.transformer(hidden_states=b_det_states, attention_mask=None, context=b_think_states, context_mask=None) # [si, 1, h]
            b_ret = self.det_head(b_ret).squeeze(1) # [si, h]
            ret[b, :b_det_states.shape[0], :] = b_ret
        
        return ret
            

class ThinkingAttnRefineModule(TransformerBlock):
    """
    A specialized transformer block that alternates between processing hidden states 
    and context through cross attention layers.

    For reference: 
    Qwen2VL uses a simple TransformerBlock as a decoder
    Qwen25VL uses a modified TransformerBlock as a decoder where the sequence packing is done cleverly for alternating between chunked attention and full attention

    in our case, we want to alternate cross attention between hidden states and context
    
    This module implements a "thinking" mechanism where each layer alternates between:
    1. Using hidden states as query and context as key/value
    2. Using context as query and hidden states as key/value
    
    This allows bidirectional refinement between the two sequences.
    """

    def __init__(
        self,
        config: ThinkingAttnRefineModuleConfig,
        spec: Union[TransformerBlockSubmodules, ModuleSpec],
        post_layer_norm: bool = True,
        pre_process: bool = True,
        post_process: bool = True,
        model_comm_pgs = None,
        vp_stage: Optional[int] = None,
    ):
        super().__init__(
            config=config,
            spec=spec,
            post_layer_norm=post_layer_norm,
            pre_process=pre_process,
            post_process=post_process,
            model_comm_pgs=model_comm_pgs,
            vp_stage=vp_stage,
        )

        # classification head
        if config.cls_head is not None:
            self.cls_head = config.cls_head.configure_model()

    def cls_head_forward(self, cls_embeddings: Tensor, think_states: Tensor, hidden_states_indices: Tensor):
        '''
        Given classification embeddings and think states, run classification head
        '''
        return self.cls_head(cls_embeddings, think_states, hidden_states_indices)

    def forward(
        self,
        image_embeddings: Union[Tensor, WrappedTensor],
        think_states: Union[Tensor, WrappedTensor],
        image_grid_thw: Tensor,
        hidden_states_indices: Tensor,  # [len, idx] tensor where idx = (seq_index, batch_index)
        # attention_mask: Optional[Tensor],
        # context: Optional[Tensor] = None,
        # context_mask: Optional[Tensor] = None,
        rotary_pos_emb: Optional[Tensor] = None,
        rotary_pos_cos: Optional[Tensor] = None,
        rotary_pos_sin: Optional[Tensor] = None,
        attention_bias: Optional[Tensor] = None,
        inference_context: Optional[BaseInferenceContext] = None,
        packed_seq_params: Optional[PackedSeqParams] = None,
        sequence_len_offset: Optional[Tensor] = None,
    ):
        """
        Forward pass that alternates between processing hidden states and context.
        
        On even layers: hidden_states = layer(hidden_states, context=context)
        On odd layers: context = layer(context, context=hidden_states)
        
        Args:
            hidden_states: Primary sequence tensor [seq_len, batch, hidden_size]
            attention_mask: Attention mask for hidden states
            context: Secondary sequence tensor [context_len, batch, hidden_size] 
            context_mask: Attention mask for context
            rotary_pos_emb: Optional rotary positional embeddings
            attention_bias: Optional attention bias tensor
            inference_context: Optional inference optimization context
            packed_seq_params: Optional packed sequence parameters
            
        Returns:
            tuple: (hidden_states, context) - The refined sequences after alternating processing
        """
        # Handle wrapped tensor case
        if isinstance(image_embeddings, WrappedTensor):
            image_embeddings = image_embeddings.unwrap()
        if isinstance(think_states, WrappedTensor):
            think_states = think_states.unwrap()

        # if not self.pre_process:
        #     hidden_states = self.input_tensor

        # Make tensors viewless for gradient computation
        image_embeddings = make_viewless_tensor(inp=image_embeddings, requires_grad=True, keep_graph=True)
        think_states = make_viewless_tensor(inp=think_states, requires_grad=True, keep_graph=True)

        # Setup RNG context for sequence parallelism
        if self.config.sequence_parallel:
            rng_context = tensor_parallel.get_cuda_rng_tracker().fork()
        else:
            rng_context = nullcontext()

        # Determine if we need outer or inner FP8 contexts
        use_outer_fp8_context = self.config.fp8 and self.config.fp8_recipe == "delayed"
        use_inner_fp8_context = self.config.fp8 and self.config.fp8_recipe != "delayed"
        outer_fp8_context = get_fp8_context(self.config) if use_outer_fp8_context else nullcontext()

        # generate packed sequence params
        assert len(image_embeddings.shape) == 2, "image embeddings should be [packed seq * batches, hidden_size]"
        assert len(think_states.shape) == 2, "think states should be [num_thinking_tokens across batches, hidden_size]"

        # generate attention masks   # [s', 1, h]
        image_embeddings = image_embeddings.unsqueeze(1)
        think_states = think_states.unsqueeze(1)

        img_cu_len = image_embeddings.shape[0]
        think_cu_len = think_states.shape[0]

        # batch size
        batch_size = image_grid_thw.shape[0]   # [B, 3]
        start_idx = 0

        # generate attention mask (think query mask is just a transpose of img query mask)
        img_query_mask = torch.zeros((1, 1, img_cu_len, think_cu_len))

        # create attention mask
        for b in range(batch_size):
            think_idx = torch.nonzero(hidden_states_indices[:, 1] == b)
            img_len = image_grid_thw[b, 0] * image_grid_thw[b, 1] * image_grid_thw[b, 2]
            img_query_mask[0, 0, start_idx:start_idx+img_len, think_idx] = 1
            start_idx += img_len

        think_query_mask = img_query_mask.transpose(2, 3)

        with rng_context, outer_fp8_context:
            # Forward pass through layers with alternating processing
            for i, layer in enumerate(self.layers):
                inner_fp8_context = (
                    get_fp8_context(self.config, layer.layer_number - 1)
                    if use_inner_fp8_context
                    else nullcontext()
                )
                
                with self.offload_context, inner_fp8_context:
                    if i % 2 == 0:
                        # Even layers: Process hidden states with context attention
                        image_embeddings, _ = layer(
                            hidden_states=image_embeddings,
                            attention_mask=None,
                            context=think_states,
                            context_mask=img_query_mask,
                            rotary_pos_emb=rotary_pos_emb,
                            rotary_pos_cos=rotary_pos_cos,
                            rotary_pos_sin=rotary_pos_sin,
                            attention_bias=attention_bias,
                            inference_context=inference_context,
                            packed_seq_params=packed_seq_params,
                            sequence_len_offset=sequence_len_offset,
                        )
                    else:
                        # Odd layers: Process context with hidden states attention
                        think_states, _ = layer(
                            hidden_states=think_states,
                            attention_mask=None,
                            context=image_embeddings,
                            context_mask=think_query_mask,
                            rotary_pos_emb=rotary_pos_emb,
                            rotary_pos_cos=rotary_pos_cos,
                            rotary_pos_sin=rotary_pos_sin,
                            attention_bias=attention_bias,
                            inference_context=inference_context,
                            packed_seq_params=packed_seq_params,
                            sequence_len_offset=sequence_len_offset,
                        )

                # Handle CPU offloading if enabled
                if (
                    torch.is_grad_enabled()
                    and self.config.cpu_offloading
                    and self.group_prefetch_offload_commit_async is not None
                ):
                    image_embeddings = self.group_prefetch_offload_commit_async(image_embeddings)
                    think_states = self.group_prefetch_offload_commit_async(think_states)

        # Apply final layer norm if present
        if self.final_layernorm is not None:
            image_embeddings = self.final_layernorm(image_embeddings)
            think_states = self.final_layernorm(think_states)
            
            # Make output tensors viewless
            image_embeddings = make_viewless_tensor(inp=image_embeddings, requires_grad=True, keep_graph=True)
            think_states = make_viewless_tensor(inp=think_states, requires_grad=True, keep_graph=True)

        # Handle empty transformer block case
        if not self.pre_process and len(self.layers) == 0 and not self.final_layernorm:
            image_embeddings = image_embeddings.clone()
            think_states = think_states.clone()

        return image_embeddings, think_states

