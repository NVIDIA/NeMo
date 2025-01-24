from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.distributed

from nemo.collections.llm.fn.activation import quick_gelu
from nemo.collections.nlp.modules.common.megatron.utils import ApproxGELUActivation
from nemo.collections.vlm.clip.model import ClipConfig, CLIPModel, CLIPTextModelConfig, CLIPViTConfig
from nemo.lightning import io, teardown


@dataclass
class CLIPViTL_14_224_Config(CLIPViTConfig):
    """Clip vit large patch14 config"""

    # TOdo these are probably not super upto date but that's ok
    # Will handle it later
    vision_model_type: str = "clip"
    patch_dim: int = 16
    img_h: int = 224
    img_w: int = 224
    num_layers: int = 12
    num_attention_heads: int = 12
    hidden_size: int = 768
    hidden_dropout: float = 0.0
    attention_dropout: float = 0.0
    ffn_hidden_size: int = 3072
    gated_linear_unit: bool = False
    kv_channels: int = 64
    layernorm_zero_centered_gamma: bool = False
    apply_query_key_layer_scaling: bool = False
    bias_activation_fusion: bool = False
    bias_dropout_fusion: bool = True
    attention_softmax_in_fp32: bool = False
    normalization: str = 'LayerNorm'
    apply_rope_fusion: bool = False
    masked_softmax_fusion: bool = True
    persist_layer_norm: bool = True


@dataclass
class CLIPViTB_32_224_Config(CLIPViTConfig):
    """Clip vit large patch14 config"""

    # TOdo these are probably not super upto date but that's ok
    # Will handle it later
    vision_model_type: str = "clip"
    patch_dim: int = 32
    img_h: int = 224
    img_w: int = 224
    num_layers: int = 12
    num_attention_heads: int = 12
    hidden_size: int = 768
    hidden_dropout: float = 0.0
    attention_dropout: float = 0.0
    ffn_hidden_size: int = 3072
    gated_linear_unit: bool = False
    kv_channels: int = None
    class_token_len: int = 7
    init_method_std: float = 0.02
    layernorm_zero_centered_gamma: bool = False
    apply_query_key_layer_scaling: bool = False
    bias_activation_fusion: bool = False
    bias_dropout_fusion: bool = True
    attention_softmax_in_fp32: bool = False
    normalization: str = 'LayerNorm'
    apply_rope_fusion: bool = False
    masked_softmax_fusion: bool = True
    persist_layer_norm: bool = True

@dataclass
class CLIPTextModelB_32_224_Config(CLIPTextModelConfig):
    # model architecture
    max_seq_length: int = 80
    max_position_embeddings: int = 80
    num_layers: int = 12
    hidden_size: int = 512
    ffn_hidden_size: int = 2048  # Transformer FFN hidden size. Usually 4 * hidden_size.
    num_attention_heads: int = 8
    init_method_std: float = (
        0.02  # Standard deviation of the zero mean normal distribution used for weight initialization.')
    )
    use_scaled_init_method: bool = True  # use scaled residuals initialization
    hidden_dropout: float = 0.0  # Dropout probability for hidden state transformer.
    attention_dropout: float = 0.0
    apply_query_key_layer_scaling: bool = False  # scale Q * K^T by 1 / layer-number.
    attention_softmax_in_fp32: bool = False
    normalization: bool = "LayerNorm"
    do_layer_norm_weight_decay: bool = False  # True means weight decay on all params

    # TODO(askYu): Does these 3 makes sense?
    persist_layer_norm: bool = True  # Use of persistent fused layer norm kernel.
    masked_softmax_fusion: bool = True
    bias_dropout_fusion: bool = True
    bias_activation_fusion: False


@dataclass
class CLIPTextModelL_14_224_Config(CLIPTextModelConfig):
    # model architecture
    max_seq_length: int = 77
    max_position_embeddings: int = 77
    num_layers: int = 12
    hidden_size: int = 512
    ffn_hidden_size: int = 2048  # Transformer FFN hidden size. Usually 4 * hidden_size.
    num_attention_heads: int = 8
    init_method_std: float = (
        0.02  # Standard deviation of the zero mean normal distribution used for weight initialization.')
    )
    use_scaled_init_method: bool = True  # use scaled residuals initialization
    hidden_dropout: float = 0.0  # Dropout probability for hidden state transformer.
    attention_dropout: float = 0.0
    apply_query_key_layer_scaling: bool = False  # scale Q * K^T by 1 / layer-number.
    do_layer_norm_weight_decay: bool = False  # True means weight decay on all params

    # TODO(askYu): Does these 3 makes sense?
    persist_layer_norm: bool = True  # Use of persistent fused layer norm kernel.
    masked_softmax_fusion: bool = True
    bias_dropout_fusion: bool = True


@dataclass
class ClipConfigL14(ClipConfig):
    text_transformer_config: CLIPTextModelConfig = field(default_factory=lambda: CLIPTextModelL_14_224_Config())
    vision_transformer_config: CLIPViTConfig = field(default_factory=lambda: CLIPViTL_14_224_Config())


@dataclass
class ClipConfigB32(ClipConfig):
    text_transformer_config: CLIPTextModelConfig = field(default_factory=lambda: CLIPTextModelB_32_224_Config())
    vision_transformer_config: CLIPViTConfig = field(default_factory=lambda: CLIPViTB_32_224_Config())

@io.model_importer(CLIPModel, "hf")
class HFClipImporter(io.ModelConnector["CLIPModel", CLIPModel]):
    def init(self) -> CLIPModel:
        return CLIPModel(self.config, tokenizer=self.tokenizer)

    def apply(self, output_path: Path) -> Path:
        from transformers import CLIPModel

        source = CLIPModel.from_pretrained(str(self))
        target = self.init()
        trainer = self.nemo_setup(target)
        import pdb; pdb.set_trace()
        self.convert_state(source, target)
        print(f"Converted Clip model to Nemo, saving to {output_path}")

        self.nemo_save(output_path, trainer)

        print(f"Converted Clip model saved to {output_path}")

        teardown(trainer, target)
        del trainer, target

        return output_path

    def convert_state(self, source, target, image_newline=False):

        # Start with the heads
        mapping = {
            'text_projection.weight': "text_model.head.weight",
            'visual_projection.weight': 'vision_model.head.weight',
        }

        mapping.update(
            {
                "text_model.embeddings.token_embedding.weight": "text_model.embedding.word_embeddings.weight",
                "text_model.embeddings.position_embedding.weight": "text_model.embedding.position_embeddings.weight",
                "text_model.final_layer_norm.weight": "text_model.final_layernorm.weight",
                "text_model.final_layer_norm.bias": "text_model.final_layernorm.bias",
                "vision_model.embeddings.class_embedding": "vision_model.class_token",
                "vision_model.embeddings.patch_embedding.weight": "vision_model.conv1.weight",
                "vision_model.embeddings.position_embedding.weight": "vision_model.position_embeddings.weight",
                "vision_model.pre_layrnorm.weight": "vision_model.ln_pre.weight",
                "vision_model.pre_layrnorm.bias": "vision_model.ln_pre.bias",
                "vision_model.post_layernorm.weight": "vision_model.final_layernorm.weight",
                "vision_model.post_layernorm.bias": "vision_model.final_layernorm.bias",
                "text_model.encoder.layers.*.self_attn.out_proj.weight": "text_model.decoder.layers.*.self_attention.linear_proj.weight",
                "text_model.encoder.layers.*.self_attn.out_proj.bias": "text_model.decoder.layers.*.self_attention.linear_proj.bias",
                "text_model.encoder.layers.*.layer_norm1.weight": "text_model.decoder.layers.*.self_attention.linear_qkv.layer_norm_weight",
                "text_model.encoder.layers.*.layer_norm1.bias": "text_model.decoder.layers.*.self_attention.linear_qkv.layer_norm_bias",
                "text_model.encoder.layers.*.mlp.fc1.weight": "text_model.decoder.layers.*.mlp.linear_fc1.weight",
                "text_model.encoder.layers.*.mlp.fc1.bias": "text_model.decoder.layers.*.mlp.linear_fc1.bias",
                "text_model.encoder.layers.*.mlp.fc2.weight": "text_model.decoder.layers.*.mlp.linear_fc2.weight",
                "text_model.encoder.layers.*.mlp.fc2.bias": "text_model.decoder.layers.*.mlp.linear_fc2.bias",
                "text_model.encoder.layers.*.layer_norm2.weight": "text_model.decoder.layers.*.mlp.linear_fc1.layer_norm_weight",
                "text_model.encoder.layers.*.layer_norm2.bias": "text_model.decoder.layers.*.mlp.linear_fc1.layer_norm_bias",
                "vision_model.encoder.layers.*.self_attn.out_proj.weight": "vision_model.decoder.layers.*.self_attention.linear_proj.weight",
                "vision_model.encoder.layers.*.self_attn.out_proj.bias": "vision_model.decoder.layers.*.self_attention.linear_proj.bias",
                "vision_model.encoder.layers.*.layer_norm1.weight": "vision_model.decoder.layers.*.self_attention.linear_qkv.layer_norm_weight",
                "vision_model.encoder.layers.*.layer_norm1.bias": "vision_model.decoder.layers.*.self_attention.linear_qkv.layer_norm_bias",
                "vision_model.encoder.layers.*.mlp.fc1.weight": "vision_model.decoder.layers.*.mlp.linear_fc1.weight",
                "vision_model.encoder.layers.*.mlp.fc1.bias": "vision_model.decoder.layers.*.mlp.linear_fc1.bias",
                "vision_model.encoder.layers.*.mlp.fc2.weight": "vision_model.decoder.layers.*.mlp.linear_fc2.weight",
                "vision_model.encoder.layers.*.mlp.fc2.bias": "vision_model.decoder.layers.*.mlp.linear_fc2.bias",
                "vision_model.encoder.layers.*.layer_norm2.weight": "vision_model.decoder.layers.*.mlp.linear_fc1.layer_norm_weight",
                "vision_model.encoder.layers.*.layer_norm2.bias": "vision_model.decoder.layers.*.mlp.linear_fc1.layer_norm_bias",
            }
        )

        return io.apply_transforms(
            source,
            target,
            mapping=mapping,
            transforms=[
                _import_cls_token,
                _import_vision_qkv_bias,
                _import_vision_qkv,
                _import_language_qkv_bias,
                _import_language_qkv,
            ],
        )

    @property
    def tokenizer(self) -> "AutoTokenizer":
        from nemo.collections.common.tokenizers.huggingface.auto_tokenizer import AutoTokenizer

        return AutoTokenizer(str(self))

    @property
    def config(self) -> ClipConfig:
        from transformers import CLIPConfig as HFCLIPConfig

        source = HFCLIPConfig.from_pretrained(str(self))

        text_conifg = source.text_config

        #
        # text_conifg = {
        #     "attention_dropout": 0.0, #
        #     "bos_token_id": 49406,
        #     "dropout": 0.0, #
        #     "eos_token_id": 49407,
        #     "hidden_act": "quick_gelu", #
        #     "hidden_size": 768, #
        #     "initializer_factor": 1.0,
        #     "initializer_range": 0.02, #
        #     "intermediate_size": 3072, #
        #     "layer_norm_eps": 1e-05, #
        #     "max_position_embeddings": 77,
        #     "model_type": "clip_text_model",
        #     "num_attention_heads": 12, #
        #     "num_hidden_layers": 12, #
        #     "pad_token_id": 1,
        #     "projection_dim": 768, #
        #     "transformers_version": "4.46.0",
        #     "vocab_size": 49408
        # }

        def make_vocab_size_divisible_by(vocab_size):
            base = 128
            while vocab_size % base != 0:
                base //= 2
            return base


        language_transformer_config = CLIPTextModelConfig(
            output_dim=text_conifg.projection_dim,
            num_layers=text_conifg.num_hidden_layers,
            hidden_size=text_conifg.hidden_size,
            ffn_hidden_size=text_conifg.intermediate_size,
            num_attention_heads=text_conifg.num_attention_heads,
            init_method_std=text_conifg.initializer_range,
            layernorm_epsilon=text_conifg.layer_norm_eps,
            gated_linear_unit=False,
            make_vocab_size_divisible_by=make_vocab_size_divisible_by(text_conifg.vocab_size),
            share_embeddings_and_output_weights=False,
            attention_dropout=text_conifg.attention_dropout,
            hidden_dropout=text_conifg.dropout,
            activation_func=ApproxGELUActivation,
            max_seq_length=text_conifg.max_position_embeddings,
            apply_query_key_layer_scaling=False,

            # These are just to match the nemo1 exactly
            bf16=True,
            params_dtype=torch.bfloat16,
            autocast_dtype=torch.bfloat16,
            pipeline_dtype=torch.bfloat16,
            deallocate_pipeline_outputs=True,
            masked_softmax_fusion=True,
            persist_layer_norm=True,
            bias_dropout_fusion=True,
            distribute_saved_activations=False
        )

        vision_config = source.vision_config
        # from attrdict import AttrDict
        # vision_config = AttrDict({
        #     "attention_dropout": 0.0,
        #     "dropout": 0.0,
        #     "hidden_act": "quick_gelu",
        #     "hidden_size": 1024,
        #     "image_size": 224,
        #     "initializer_factor": 1.0,
        #     "initializer_range": 0.02,
        #     "intermediate_size": 4096,
        #     "layer_norm_eps": 1e-05,
        #     "model_type": "clip_vision_model",
        #     "num_attention_heads": 16,
        #     "num_channels": 3,
        #     "num_hidden_layers": 24,
        #     "patch_size": 14,
        #     "projection_dim": 768,
        #     "transformers_version": "4.46.0"
        # })
        #
        vision_transformer_config = CLIPViTConfig(
            vision_model_type="clip",
            patch_dim=vision_config.patch_size,
            img_h=vision_config.image_size,
            img_w=vision_config.image_size,
            num_layers=vision_config.num_hidden_layers,
            num_attention_heads=vision_config.num_attention_heads,
            hidden_size=vision_config.hidden_size,
            hidden_dropout=vision_config.dropout,
            attention_dropout=vision_config.attention_dropout,
            ffn_hidden_size=vision_config.intermediate_size,
            gated_linear_unit=False,  # TODO (ask Yao, This was False in the config) Does he knows if they use GLU?
            apply_query_key_layer_scaling=False,
            activation_func=ApproxGELUActivation,
            output_dim=vision_config.projection_dim,
            init_method_std=vision_config.initializer_range,
            layernorm_epsilon=vision_config.layer_norm_eps,
            # HF only uses one class token
            class_token_len=1,
            # bias_activation_fusion: bool = False
            # bias_dropout_fusion: bool = False
            # attention_softmax_in_fp32: bool = True
            # normalization: str = 'LayerNorm'
            # apply_rope_fusion: bool = False

            # These are just to match the nemo1 exactly
            bf16 = True,
            params_dtype=torch.bfloat16,
            autocast_dtype=torch.bfloat16,
            pipeline_dtype=torch.bfloat16,
            deallocate_pipeline_outputs=True,
            masked_softmax_fusion=True,
            persist_layer_norm=True,
            bias_dropout_fusion=True,
            distribute_saved_activations=False
        )

        output = ClipConfig(
            text_transformer_config=language_transformer_config, vision_transformer_config=vision_transformer_config
        )

        return output


def import_qkv(q, k, v, head_num, num_query_groups, heads_per_group, hidden_size, head_size):
    old_tensor_shape = q.size()
    new_q_tensor_shape = (head_num, head_size) + old_tensor_shape[1:]
    new_kv_tensor_shape = (num_query_groups, head_size) + old_tensor_shape[1:]

    q = q.view(*new_q_tensor_shape)
    k = k.view(*new_kv_tensor_shape)
    v = v.view(*new_kv_tensor_shape)

    qkv_weights_l = []
    for i in range(num_query_groups):
        qkv_weights_l.append(q[i * heads_per_group : (i + 1) * heads_per_group, :, :])
        qkv_weights_l.append(k[i : i + 1, :, :])
        qkv_weights_l.append(v[i : i + 1, :, :])
    qkv_weights = torch.cat(qkv_weights_l)
    assert qkv_weights.ndim == 3, qkv_weights.shape
    assert qkv_weights.shape[0] == (heads_per_group + 2) * num_query_groups, qkv_weights.shape
    assert qkv_weights.shape[1] == head_size, qkv_weights.shape
    assert qkv_weights.shape[2] == old_tensor_shape[1], qkv_weights.shape

    qkv_weights = qkv_weights.reshape([head_size * (head_num + 2 * num_query_groups), hidden_size])

    return qkv_weights


@io.state_transform(
    source_key=(
        "vision_model.encoder.layers.*.self_attn.q_proj.weight",
        "vision_model.encoder.layers.*.self_attn.k_proj.weight",
        "vision_model.encoder.layers.*.self_attn.v_proj.weight",
    ),
    target_key="vision_model.decoder.layers.*.self_attention.linear_qkv.weight",
)
def _import_vision_qkv(ctx: io.TransformCTX, q, k, v):
    megatron_config = ctx.target.config.vision_transformer_config
    return import_qkv(
        q,
        k,
        v,
        head_num=megatron_config.num_attention_heads,
        num_query_groups=megatron_config.num_query_groups,
        heads_per_group=megatron_config.num_attention_heads // megatron_config.num_query_groups,
        hidden_size=megatron_config.hidden_size,
        head_size=megatron_config.kv_channels,
    )


@io.state_transform(
    source_key=(
        "vision_model.encoder.layers.*.self_attn.q_proj.bias",
        "vision_model.encoder.layers.*.self_attn.k_proj.bias",
        "vision_model.encoder.layers.*.self_attn.v_proj.bias",
    ),
    target_key="vision_model.decoder.layers.*.self_attention.linear_qkv.bias",
)
def _import_vision_qkv_bias(ctx: io.TransformCTX, q_bias, k_bias, v_bias):
    megatron_config = ctx.target.config.vision_transformer_config
    return import_qkv(
        q_bias.unsqueeze(-1),
        k_bias.unsqueeze(-1),
        v_bias.unsqueeze(-1),
        head_num=megatron_config.num_attention_heads,
        num_query_groups=megatron_config.num_query_groups,
        heads_per_group=megatron_config.num_attention_heads // megatron_config.num_query_groups,
        hidden_size=1,
        head_size=megatron_config.kv_channels,
    ).squeeze(-1)


@io.state_transform(
    source_key=(
        "text_model.encoder.layers.*.self_attn.q_proj.bias",
        "text_model.encoder.layers.*.self_attn.k_proj.bias",
        "text_model.encoder.layers.*.self_attn.v_proj.bias",
    ),
    target_key="text_model.decoder.layers.*.self_attention.linear_qkv.bias",
)
def _import_language_qkv_bias(ctx: io.TransformCTX, q_bias, k_bias, v_bias):
    megatron_config = ctx.target.config.text_transformer_config
    return import_qkv(
        q_bias.unsqueeze(-1),
        k_bias.unsqueeze(-1),
        v_bias.unsqueeze(-1),
        head_num=megatron_config.num_attention_heads,
        num_query_groups=megatron_config.num_query_groups,
        heads_per_group=megatron_config.num_attention_heads // megatron_config.num_query_groups,
        hidden_size=1,
        head_size=megatron_config.kv_channels,
    ).squeeze(-1)


# from nemo.lightning.io.state import
@io.state_transform(
    source_key=(
        "text_model.encoder.layers.*.self_attn.q_proj.weight",
        "text_model.encoder.layers.*.self_attn.k_proj.weight",
        "text_model.encoder.layers.*.self_attn.v_proj.weight",
    ),
    target_key="text_model.decoder.layers.*.self_attention.linear_qkv.weight",
)
def _import_language_qkv(ctx: io.TransformCTX, q, k, v):
    megatron_config = ctx.target.config.text_transformer_config

    return import_qkv(
        q,
        k,
        v,
        head_num=megatron_config.num_attention_heads,
        num_query_groups=megatron_config.num_query_groups,
        heads_per_group=megatron_config.num_attention_heads // megatron_config.num_query_groups,
        hidden_size=megatron_config.hidden_size,
        head_size=megatron_config.kv_channels,
    )


# Todo (Ask Yu) about class token
@io.state_transform(
    source_key=("vision_model.embeddings.class_embedding",),
    target_key="vision_model.class_token",
)
def _import_cls_token(ctx: io.TransformCTX, cls_token):
    return cls_token.reshape(1, 1, -1)
