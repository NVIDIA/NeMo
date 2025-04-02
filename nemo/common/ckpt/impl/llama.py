from pathlib import Path

import torch
from torch import nn
from transformers import AutoModelForCausalLM, LlamaForCausalLM, LlamaConfig as HFLlamaConfig


from nemo.collections.llm.gpt.model.llama import (
    LlamaConfig, 
    LlamaModel, 
    _import_qkv,
    _import_linear_fc1,
    _export_qkv, 
    _export_linear_fc1, 
    _export_embedding,
    _export_head
)
from nemo.lightning.pytorch.utils import dtype_from_hf
from nemo.collections.common.tokenizers.huggingface.auto_tokenizer import (
    AutoTokenizer,
)
from auto_model.auto import AutoModel, register_context_convert, register_state_convert
from auto_model.plan import Plan
from auto_model.checkpoint.impl.hf import HFPreTrained
from auto_model.convert.impl_model.megatron import MegatronSaveModel, MegatronStateConvertPlan
from auto_model.convert.api import StateConverter


@register_context_convert("megatron", source=LlamaForCausalLM, target=LlamaModel)
def llama_hf_to_megatron(context: HFPreTrained[LlamaForCausalLM]) -> LlamaModel:
    source_config = context.config

    def make_vocab_size_divisible_by(vocab_size):
        base = 128
        while vocab_size % base != 0:
            base //= 2
        return base

    config = LlamaConfig(
        num_layers=source_config.num_hidden_layers,
        hidden_size=source_config.hidden_size,
        ffn_hidden_size=source_config.intermediate_size,
        num_attention_heads=source_config.num_attention_heads,
        init_method_std=source_config.initializer_range,
        layernorm_epsilon=source_config.rms_norm_eps,
        num_query_groups=source_config.num_key_value_heads,
        rotary_base=source_config.rope_theta,
        gated_linear_unit=True,
        make_vocab_size_divisible_by=make_vocab_size_divisible_by(source_config.vocab_size),
        share_embeddings_and_output_weights=getattr(
            source_config, "tie_word_embeddings", False
        ),
        fp16=(dtype_from_hf(source_config) == torch.float16),
        bf16=(dtype_from_hf(source_config) == torch.bfloat16),
        params_dtype=dtype_from_hf(source_config),
    )
    tokenizer = AutoTokenizer(context.path)

    return LlamaModel(config, tokenizer=tokenizer)


@register_state_convert("megatron", source=LlamaForCausalLM, target=LlamaModel)
class HFLlamaToMegatron(MegatronStateConvertPlan):
    def __init__(self):
        super().__init__(
            {
                "model.embed_tokens.weight": "embedding.word_embeddings.weight",
                "model.layers.*.self_attn.o_proj.weight": "decoder.layers.*.self_attention.linear_proj.weight",
                "model.layers.*.mlp.down_proj.weight": "decoder.layers.*.mlp.linear_fc2.weight",
                "model.layers.*.input_layernorm.weight": "decoder.layers.*.self_attention.linear_qkv.layer_norm_weight",
                "model.layers.*.post_attention_layernorm.weight": "decoder.layers.*.mlp.linear_fc1.layer_norm_weight",
                "model.norm.weight": "decoder.final_layernorm.weight",
                "lm_head.weight": "output_layer.weight",
            },
            transforms=[_import_qkv, _import_linear_fc1]
        )

    def create_source_module(self, input_path) -> nn.Module:
        from transformers import LlamaForCausalLM

        source_module = LlamaForCausalLM.from_pretrained(input_path, torch_dtype="auto")

        if getattr(source_module.config, "tie_word_embeddings", False):
            # llama 3.2 1B and 3B models have no shared input output embeddings
            del self.converter.mapping["lm_head.weight"]

        return source_module


@register_context_convert("huggingface", source=LlamaModel, target=LlamaForCausalLM)
def llama_megatron_to_hf(
    context: HFPreTrained[LlamaForCausalLM],
    dtype: torch.dtype = torch.bfloat16,
) -> LlamaForCausalLM:
    source = context.config
    config = HFLlamaConfig(
        num_hidden_layers=source.num_layers,
        hidden_size=source.hidden_size,
        intermediate_size=source.ffn_hidden_size,
        num_attention_heads=source.num_attention_heads,
        max_position_embeddings=source.seq_length,
        initializer_range=source.init_method_std,
        rms_norm_eps=source.layernorm_epsilon,
        num_key_value_heads=source.num_query_groups,
        rope_theta=source.rotary_base,
        vocab_size=context.tokenizer.vocab_size,
        tie_word_embeddings=source.share_embeddings_and_output_weights,
    )

    return AutoModelForCausalLM.from_config(config, torch_dtype=dtype)


@register_state_convert("huggingface", source=LlamaModel, target=LlamaForCausalLM)
class MegatronToHFLlama(Plan):
    def __init__(self):
        self.converter = StateConverter(
            {
                "decoder.layers.*.self_attention.linear_proj.weight": "model.layers.*.self_attn.o_proj.weight",
                "decoder.layers.*.mlp.linear_fc2.weight": "model.layers.*.mlp.down_proj.weight",
                "decoder.layers.*.self_attention.linear_qkv.layer_norm_weight": "model.layers.*.input_layernorm.weight",
                "decoder.layers.*.mlp.linear_fc1.layer_norm_weight": "model.layers.*.post_attention_layernorm.weight",
                "decoder.final_layernorm.weight": "model.norm.weight",
            },
            transforms=[_export_qkv, _export_linear_fc1, _export_embedding]
        )
        self.saver = ...
    
    def execute(self, source: Path | str, target: Path) -> Path:
        from transformers import LlamaForCausalLM

        source_module = AutoModel(source, importer="megatron", setup="megatron_meta")
        target_module = AutoModel(source, importer="huggingface", setup="meta")
        

        if not target_module.config.tie_word_embeddings:
            self.converter.transforms.append(_export_head)

        target_module = self.converter(source_module, target_module)
        # We have model.fabric, so we could do the saving with just the model
        self.saver(target_module, target)
