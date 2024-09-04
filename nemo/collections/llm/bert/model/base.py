
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Dict, Literal, Optional, Union

import pytorch_lightning as L
import torch
import torch.distributed
from megatron.core.optimizer import OptimizerConfig
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_config import TransformerConfig
from torch import nn
from megatron.core.models.bert import bert_layer_specs
from nemo.collections.llm.bert.model.bert_spec import bert_layer_with_transformer_engine_spec_postln, bert_layer_local_spec_postln
from nemo.collections.llm import fn
from nemo.lightning import get_vocab_size, io
from nemo.lightning.megatron_parallel import MaskedTokenLossReduction
from nemo.lightning.pytorch.optim import MegatronOptimizerModule, OptimizerModule
from megatron.core.models.bert.bert_model import BertModel as MCoreBert
from megatron.core.transformer.transformer_layer import TransformerLayer, TransformerLayerSubmodules
from megatron.core.models.bert.bert_lm_head import BertLMHead as MCoreBertLMHead
from megatron.core.models.bert.pooler import Pooler
from megatron.core.transformer.utils import get_linear_layer as mcore_get_linear_layer
from megatron.core.utils import make_viewless_tensor
from megatron.core import InferenceParams, ModelParallelConfig, parallel_state, tensor_parallel
from torch import Tensor
from megatron.core.transformer.spec_utils import build_module
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer.transformer_block import TransformerBlock
from megatron.core.fusions.fused_layer_norm import FusedLayerNorm

HAVE_TE = True
try:
    import transformer_engine
except (ImportError, ModuleNotFoundError):
    HAVE_TE = False

if TYPE_CHECKING:
    from nemo.collections.common.tokenizers.tokenizer_spec import TokenizerSpec


def bert_data_step(dataloder_iter) -> Dict[str, torch.Tensor]:
    batch = next(dataloder_iter)
    
    _batch: dict
    if isinstance(batch, tuple) and len(batch) == 3:
        _batch = batch[0]
    else:
        _batch = batch

    required_keys = set()
    required_keys.add("attention_mask")
    if parallel_state.is_pipeline_first_stage():
        required_keys.update(("tokens", "position_ids"))
    if parallel_state.is_pipeline_last_stage():
        required_keys.update(("labels", "loss_mask"))

    _batch = {key: val.cuda(non_blocking=True) if key in required_keys else None for key, val in _batch.items()}
    # slice batch along sequence dimension for context parallelism
    output = get_batch_on_this_context_parallel_rank(_batch)

    return output

def bert_forward_step(model: L.LightningModule, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
    """
    This subsets the batch keys to the ones actually used by forward pass of the model, and then calls the model's forward pass.
    if "cu_seqsens" are defined in the batch, then the packed sequence parameters are also passed to the model for forward pass efficiency.
    """
    forward_args = {
        "input_ids": batch["text"],
        "attention_mask": batch["attention_mask"],
    }

    if "cu_seqlens" in batch:
        forward_args["packed_seq_params"] = get_packed_seq_params(batch)

    forward_results = model(**forward_args)
    return forward_results  # TODO support losses that also include the binary head, this means doing something more fancy than the one default GPT reduction function above MaskedTokenLossReduction()


    
def default_layer_spec(config: "BertConfig") -> ModuleSpec:
    transformer_block_type = config.get('transformer_block_type', 'pre_ln')
    assert transformer_block_type == 'pre_ln' or transformer_block_type == 'post_ln', f'Unknown transformer block type {transformer_block_type}, supported type for bert model is: pre_ln, post_ln'
    if HAVE_TE:
        if transformer_block_type == 'pre_ln':
            return bert_layer_specs.bert_layer_with_transformer_engine_spec
        else:
            return bert_layer_with_transformer_engine_spec_postln
  
    if transformer_block_type == 'pre_ln':      
        return bert_layer_specs.bert_layer_local_spec
    else:
        return bert_layer_local_spec_postln


@dataclass
class BertConfig(TransformerConfig, io.IOMixin):
    # From megatron.core.models.bert.bert_model.BertModel
    fp16_lm_cross_entropy: bool = False
    parallel_output: bool = True
    share_embeddings_and_output_weights: bool = False
    position_embedding_type: Literal["learned_absolute", "rope"] = "learned_absolute"
    rotary_base: int = 10000
    rotary_percent: float = 1.0
    seq_len_interpolation_factor: Optional[float] = None
    seq_length: int = 1024
    attention_softmax_in_fp32: bool = False
    masked_softmax_fusion: bool = True
    deallocate_pipeline_outputs = True

    transformer_layer_spec: Union[ModuleSpec, Callable[["BertConfig"], ModuleSpec]] = default_layer_spec
    forward_step_fn: Callable = bert_forward_step
    data_step_fn: Callable = bert_data_step
    
    transformer_block_type: Literal["pre-ln", "post-ln"] = "pre-ln"
    add_pooler: bool = True
    bert_binary_head: bool = True

    def configure_model(self, tokenizer) -> "MCoreBertModelWrapperWithPostLNSupport":
        breakpoint()
        vp_size = self.virtual_pipeline_model_parallel_size
        if vp_size:
            p_size = self.pipeline_model_parallel_size
            assert (
                self.num_layers // p_size
            ) % vp_size == 0, "Make sure the number of model chunks is the same across all pipeline stages."
        from megatron.core import parallel_state

        transformer_layer_spec = self.transformer_layer_spec
        if not isinstance(transformer_layer_spec, ModuleSpec):
            transformer_layer_spec = transformer_layer_spec(self)
        return MCoreBertModelWrapperWithPostLNSupport(
            transformer_block_type=self.transformer_block_type,
            add_pooler=self.add_pooler,
            config=self,
            num_tokentypes=2 if self.bert_binary_head else 0,
            transformer_layer_spec=transformer_layer_spec,
            vocab_size=get_vocab_size(self, tokenizer.vocab_size, self.make_vocab_size_divisible_by),
            max_sequence_length=self.seq_length,
            pre_process=parallel_state.is_pipeline_first_stage(),
            post_process=parallel_state.is_pipeline_last_stage(),
            fp16_lm_cross_entropy=self.fp16_lm_cross_entropy,
            parallel_output=self.parallel_output,
            share_embeddings_and_output_weights=self.share_embeddings_and_output_weights,
            position_embedding_type=self.position_embedding_type,
            rotary_percent=self.rotary_percent,
            # TODO: MCore bert not have rotary base
            seq_len_interpolation_factor=self.seq_len_interpolation_factor,
            add_binary_head=self.bert_binary_head,
            return_embeddings=False, #TODO
        )


'''
This class is used for working with HF Bert Checkpoints. These checkpoints
by default have post layer norm, while the vanilla mcore bert model does not support it.
'''
class MCoreBertModelWrapperWithPostLNSupport(MCoreBert):
    def __init__(self, transformer_block_type='pre-ln', add_pooler=True, *args, **kwargs):

        super(MCoreBertModelWrapperWithPostLNSupport, self).__init__(*args, **kwargs)
        breakpoint()
        self.add_pooler = add_pooler
        self.transformer_block_type = transformer_block_type

        # Transformer.
        self.encoder = TransformerBlockWithPostLNSupport(
            config=self.config,
            spec=self.transformer_layer_spec,
            pre_process=self.pre_process,
            post_process=self.post_process,
            transformer_block_type=self.transformer_block_type,
        )

        if self.add_pooler:
            self.pooler = Pooler(
                self.config.hidden_size, self.config.init_method, self.config, self.config.sequence_parallel
            )

        # Output
        if self.post_process:
            # TODO: Make sure you are passing in the mpu_vocab_size properly

            self.lm_head = MCoreBertLMHead(
                self.config.hidden_size,
                self.config,
            )

            self.output_layer = tensor_parallel.ColumnParallelLinear(
                self.config.hidden_size,
                self.vocab_size,
                config=self.config,
                init_method=self.config.init_method,
                bias=True,
                skip_bias_add=False,
                gather_output=not self.parallel_output,
                skip_weight_param_allocation=self.pre_process and self.share_embeddings_and_output_weights,
            )

            self.binary_head = None
            if self.add_binary_head:
                # TODO: Shoudl switch this to TE ?
                self.binary_head = mcore_get_linear_layer(
                    self.config.hidden_size, 2, self.config.init_method, self.config.perform_initialization
                )

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        tokentype_ids: Tensor = None,
        lm_labels: Tensor = None,
        inference_params=None,
    ):
        """Forward function of BERT model

        Forward function of the BERT Model This function passes the input tensors
        through the embedding layer, and then the encoder and finally into the post
        processing layer (optional).

        It either returns the Loss values if labels are given  or the final hidden units
        """
        original_post_process = self.post_process

        # We set this to false since we just want to get the hidden states from the encoder
        self.post_process = False
        hidden_states = super().forward(input_ids, attention_mask, tokentype_ids, lm_labels, inference_params)
        self.post_process = original_post_process

        if not self.post_process:
            return hidden_states

        if self.add_pooler:
            pooled_output = self.pooler(hidden_states, 0)

        if self.return_embeddings:
            embeddings = torch.transpose(hidden_states, 0, 1)
            masks = torch.sum(attention_mask, dim=1)
            # Collect masked embeddings.
            output = torch.zeros(
                size=(embeddings.shape[0], embeddings.shape[2]),
                dtype=torch.float32,
                device=torch.cuda.current_device(),
            )
            for i, (embedding, mask) in enumerate(zip(embeddings, masks)):
                output[i, :] = torch.mean(embedding[1 : mask - 1], dim=0)
            return output

        # logits and loss
        output_weight = None
        if self.share_embeddings_and_output_weights:
            output_weight = self.shared_embedding_or_output_weight()

        hidden_states_after_lm_head = self.lm_head(hidden_states=hidden_states)
        logits, _ = self.output_layer(hidden_states_after_lm_head, weight=output_weight)

        binary_logits = None
        if self.binary_head is not None and self.add_pooler:
            binary_logits = self.binary_head(pooled_output)

        if lm_labels is None:
            # [s b h] => [b s h]
            return logits.transpose(0, 1).contiguous(), binary_logits

        loss = self.compute_language_model_loss(lm_labels, logits)

        return loss, binary_logits

@dataclass
class TransformerLayerSubmodulesWithPostLNSupport(TransformerLayerSubmodules):
    def __init__(self, post_att_layernorm, post_mlp_layernorm, **kwargs):
        super(TransformerLayerSubmodulesWithPostLNSupport, self).__init__(**kwargs)
        self.post_att_layernorm = post_att_layernorm
        self.post_mlp_layernorm = post_mlp_layernorm


class TransformerLayerWithPostLNSupport(TransformerLayer):
    def __init__(self, *args, **kwargs):
        super(TransformerLayerWithPostLNSupport, self).__init__(*args, **kwargs)
        ## [Module add: Post attention LN]
        self.post_att_layernorm = build_module(
            self.submodules_config.post_att_layernorm,
            config=self.config,
            hidden_size=self.config.hidden_size,
            eps=self.config.layernorm_epsilon,
        )
        ## [Module add: Post MLP LN]
        self.post_mlp_layernorm = build_module(
            self.submodules_config.post_mlp_layernorm,
            config=self.config,
            hidden_size=self.config.hidden_size,
            eps=self.config.layernorm_epsilon,
        )

    def forward(
        self,
        hidden_states,
        attention_mask,
        context=None,
        context_mask=None,
        rotary_pos_emb=None,
        inference_params=None,
        packed_seq_params=None,
    ):
        # hidden_states: [s, b, h]

        # Residual connection.
        residual = hidden_states

        # Optional Input Layer norm
        input_layernorm_output = self.input_layernorm(hidden_states)

        # Self attention.
        attention_output_with_bias = self.self_attention(
            input_layernorm_output,
            attention_mask=attention_mask,
            inference_params=inference_params,
            rotary_pos_emb=rotary_pos_emb,
            packed_seq_params=packed_seq_params,
        )

        # TODO: could we move `bias_dropout_add_exec_handler` itself
        # inside the module provided in the `bias_dropout_add_spec` module?
        with self.bias_dropout_add_exec_handler():
            hidden_states = self.self_attn_bda(self.training, self.config.bias_dropout_fusion)(
                attention_output_with_bias, residual, self.hidden_dropout
            )

        # Residual connection.
        residual = hidden_states

        # Post-LN after Self Attention
        hidden_states = self.post_att_layernorm(hidden_states)

        # Optional Layer norm after self-attention
        pre_cross_attn_layernorm_output = self.pre_cross_attn_layernorm(hidden_states)

        # Cross attention.
        attention_output_with_bias = self.cross_attention(
            pre_cross_attn_layernorm_output,
            attention_mask=context_mask,
            key_value_states=context,
            inference_params=inference_params,
        )

        if isinstance(attention_output_with_bias, dict) and "context" in attention_output_with_bias:
            context = attention_output_with_bias["context"]

        # TODO: could we move `bias_dropout_add_exec_handler` itself
        # inside the module provided in the `bias_dropout_add_spec` module?
        with self.bias_dropout_add_exec_handler():
            hidden_states = self.cross_attn_bda(self.training, self.config.bias_dropout_fusion)(
                attention_output_with_bias, residual, self.hidden_dropout
            )

        # Residual connection.
        residual = hidden_states

        # Optional Layer norm post the cross-attention.
        pre_mlp_layernorm_output = self.pre_mlp_layernorm(hidden_states)

        # MLP.
        mlp_output_with_bias = self.mlp(pre_mlp_layernorm_output)

        # TODO: could we move `bias_dropout_add_exec_handler` itself
        # inside the module provided in the `bias_dropout_add_spec` module?
        with self.bias_dropout_add_exec_handler():
            hidden_states = self.mlp_bda(self.training, self.config.bias_dropout_fusion)(
                mlp_output_with_bias, residual, self.hidden_dropout
            )

        # Post-LN after MLP
        hidden_states = self.post_mlp_layernorm(hidden_states)

        # Jit compiled function creates 'view' tensor. This tensor
        # potentially gets saved in the MPU checkpoint function context,
        # which rejects view tensors. While making a viewless tensor here
        # won't result in memory savings (like the data loader, or
        # p2p_communication), it serves to document the origin of this
        # 'view' tensor.
        output = make_viewless_tensor(inp=hidden_states, requires_grad=hidden_states.requires_grad, keep_graph=True)

        return output, context


class TransformerBlockWithPostLNSupport(TransformerBlock):
    def __init__(self, transformer_block_type='post_ln', *args, **kwargs):

        super(TransformerBlockWithPostLNSupport, self).__init__(*args, **kwargs)
        self.transformer_block_type = transformer_block_type
        if self.transformer_block_type == 'post_ln':
            self.initial_layernorm = FusedLayerNorm(
                config=self.config, hidden_size=self.config.hidden_size, eps=self.config.layernorm_epsilon
            )

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor,
        context: Tensor = None,
        context_mask: Tensor = None,
        rotary_pos_emb: Tensor = None,
        inference_params: InferenceParams = None,
        packed_seq_params: PackedSeqParams = None,
    ):
        # hidden_states (float): [s, b, h]
        # attention_mask (bool): [1, 1, s, s]

        if not self.pre_process:
            # See set_input_tensor()
            hidden_states = self.input_tensor
        if self.transformer_block_type == 'post_ln':
            hidden_states = self.initial_layernorm(hidden_states)
        return super(TransformerBlockWithPostLNSupport, self).forward(
            hidden_states, attention_mask, context, context_mask, rotary_pos_emb, inference_params, packed_seq_params
        )

class BertModel(L.LightningModule, io.IOMixin, io.ConnectorMixin, fn.FNMixin):
    def __init__(
        self,
        config: BertConfig,
        # TODO: Add transformer_layer_spec when we update mcore
        optim: Optional[OptimizerModule] = None,
        tokenizer: Optional["TokenizerSpec"] = None,
        model_transform: Optional[Callable[[nn.Module], nn.Module]] = None,
    ):
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer
        self.optim = optim or MegatronOptimizerModule(config=OptimizerConfig(lr=1e-4, use_distributed_optimizer=True))
        self.optim.connect(self) # This will bind the `configure_optimizers` method
        self.model_transform = model_transform
        self._training_loss_reduction = None
        self._validation_loss_reduction = None

    def configure_model(self) -> None:
        if not hasattr(self, "module"):
            self.module = self.config.configure_model(self.tokenizer)

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        decoder_input: Optional[torch.Tensor] = None,
        inference_params=None,
    ) -> torch.Tensor:
        output_tensor = self.module(
            input_ids,
            position_ids,
            attention_mask,
            decoder_input=decoder_input,
            labels=labels,
            inference_params=inference_params,
        )

        return output_tensor

    def data_step(self, dataloader_iter) -> Dict[str, torch.Tensor]:
        return self.config.data_step_fn(dataloader_iter)

    def forward_step(self, batch) -> torch.Tensor:
        return self.config.forward_step_fn(self, batch)

    def training_step(self, batch, batch_idx=None) -> torch.Tensor:
        # In mcore the loss-function is part of the forward-pass (when labels are provided)
        return self.forward_step(batch)

    def validation_step(self, batch, batch_idx=None) -> torch.Tensor:
        # In mcore the loss-function is part of the forward-pass (when labels are provided)

        return self.forward_step(batch)

    @property
    def training_loss_reduction(self) -> MaskedTokenLossReduction:
        if not self._training_loss_reduction:
            self._training_loss_reduction = MaskedTokenLossReduction()

        return self._training_loss_reduction

    @property
    def validation_loss_reduction(self) -> MaskedTokenLossReduction:
        if not self._validation_loss_reduction:
            self._validation_loss_reduction = MaskedTokenLossReduction(validation_step=True)

        return self._validation_loss_reduction


def get_batch_on_this_context_parallel_rank(batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Modifies the batch data based on the context parallel rank, if the context parallel world size is greater than 1. Otherwise the batch is returned as-is.

    Args:
        batch (dict): The input batch data.

    Returns:
        dict: The modified batch data based on the context parallel rank.
    """
    if cp_size := parallel_state.get_context_parallel_world_size() > 1:
        num_valid_tokens_in_ub = None
        if "loss_mask" in batch and batch["loss_mask"] is not None:
            num_valid_tokens_in_ub = batch["loss_mask"].sum()

        cp_rank = parallel_state.get_context_parallel_rank()
        for key, val in batch.items():
            if val is not None:
                seq_dim = 1 if key != "attention_mask" else 2
                _val = val.view(
                    *val.shape[0:seq_dim],
                    2 * cp_size,
                    val.shape[seq_dim] // (2 * cp_size),
                    *val.shape[(seq_dim + 1) :],
                )
                index = torch.tensor([cp_rank, (2 * cp_size - cp_rank - 1)], device="cpu", pin_memory=True).cuda(
                    non_blocking=True
                )
                _val = _val.index_select(seq_dim, index)
                _val = _val.view(*val.shape[0:seq_dim], -1, *_val.shape[(seq_dim + 2) :])
                batch[key] = _val
        batch["num_valid_tokens_in_ub"] = num_valid_tokens_in_ub
    return batch


def get_packed_seq_params(batch: Dict[str, torch.Tensor]) -> PackedSeqParams:
    """Get the packed sequence parameters for the given batch. This function should only be called if `cu_seqlens` is defined in the batch.

    Args:
        batch (dict): The input batch containing the following keys:
            - cu_seqlens (torch.Tensor): The sequence lengths of the input batch.
            - cu_seqlens_argmin (torch.Tensor, optional): The minimum sequence length index.
            - max_seqlen (torch.Tensor, optional): The maximum sequence length.

    Returns:
        PackedSeqParams: The packed sequence parameters containing the following attributes:
            - cu_seqlens_q (torch.Tensor): The sequence lengths for query.
            - cu_seqlens_kv (torch.Tensor): The sequence lengths for key and value.
            - max_seqlen_q (torch.Tensor, optional): The maximum sequence length for query.
            - max_seqlen_kv (torch.Tensor, optional): The maximum sequence length for key and value.
            - qkv_format (str): The format of query, key, and value tensors.

    """
    cu_seqlens = batch["cu_seqlens"].squeeze()  # remove batch size dimension (mbs=1)
    # remove -1 "paddings" added in collate_fn
    if cu_seqlens_argmin := batch.get("cu_seqlens_argmin", None) is not None:
        # pre-compute cu_seqlens_argmin in dataset class for perf
        cu_seqlens = cu_seqlens[: cu_seqlens_argmin.item()]
    else:
        cu_seqlens = cu_seqlens[: torch.argmin(cu_seqlens)]

    # pre-compute max_seqlens in dataset class for perf
    max_seqlen = batch["max_seqlen"].squeeze() if "max_seqlen" in batch else None

    # these args are passed eventually into TEDotProductAttention.forward()
    return PackedSeqParams(
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_kv=cu_seqlens,
        max_seqlen_q=max_seqlen,
        max_seqlen_kv=max_seqlen,
        qkv_format="thd",
    )