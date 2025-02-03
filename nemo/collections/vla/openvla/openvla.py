from dataclasses import dataclass, field
from pathlib import Path

import timm
import torch
import torch.distributed
from transformers import AutoConfig

from nemo.collections.llm import Llama2Config7B, LlamaConfig
from nemo.collections.llm.fn.activation import quick_gelu
from nemo.collections.nlp.modules.common.megatron.utils import ApproxGELUActivation
from nemo.collections.vla.openvla.base import OpenVLAModel, OpenVLAConfig, TimmCLIPVisionConfig, \
    MultimodalProjectorConfig
from nemo.collections.vlm.clip.model import ClipConfig, CLIPModel, CLIPTextModelConfig, CLIPViTConfig
from nemo.lightning import io, teardown



@io.model_importer(OpenVLAModel, "hf")
class HFOpenVLAModelImporter(io.ModelConnector["OpenVLAModel", OpenVLAModel]):
    def init(self) -> OpenVLAModel:
        return OpenVLAModel(self.config, tokenizer=self.tokenizer).to(torch.bfloat16)

    def apply(self, output_path: Path) -> Path:
        from transformers import AutoModelForVision2Seq

        source = AutoModelForVision2Seq.from_pretrained(
    "openvla/openvla-7b", # str(self)
            attn_implementation="flash_attention_2",  # [Optional] Requires `flash_attn`
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True
                )
        target = self.init()
        trainer = self.nemo_setup(target)
        self.convert_state(source, target)
        print(f"Converted Clip model to Nemo, saving to {output_path}")

        self.nemo_save(output_path, trainer)

        print(f"Converted Clip model saved to {output_path}")

        teardown(trainer, target)
        del trainer, target

        return output_path

    def convert_state(self, source, target, image_newline=False):

        import pdb; pdb.set_trace()
        target.module.language_model = target.module.language_model.to(torch.bfloat16)
        target.module.vision_projection = target.module.vision_projection.to(torch.bfloat16)

        mapping = {
            "language_model.model.embed_tokens.weight": "language_model.embedding.word_embeddings.weight",
            "language_model.model.layers.*.self_attn.o_proj.weight": "language_model.decoder.layers.*.self_attention.linear_proj.weight",
            "language_model.model.layers.*.mlp.down_proj.weight": "language_model.decoder.layers.*.mlp.linear_fc2.weight",
            "language_model.model.layers.*.input_layernorm.weight": "language_model.decoder.layers.*.self_attention.linear_qkv.layer_norm_weight",
            "language_model.model.layers.*.post_attention_layernorm.weight": "language_model.decoder.layers.*.mlp.linear_fc1.layer_norm_weight",
            "language_model.model.norm.weight": "language_model.decoder.final_layernorm.weight",
            "language_model.lm_head.weight": "language_model.output_layer.weight",
        }
        mapping.update(
            {
                "vision_backbone.featurizer.**": "vision_model.**",
                "vision_backbone.fused_featurizer.**": "secondary_vision_model.**",
            }
        )

        mapping.update(
            {
                "projector.fc1.weight": "vision_projection.0.weight",
                "projector.fc1.bias": "vision_projection.0.bias",
                "projector.fc2.weight": "vision_projection.2.weight",
                "projector.fc2.bias": "vision_projection.2.bias",
                "projector.fc3.weight": "vision_projection.4.weight",
                "projector.fc3.bias": "vision_projection.4.bias",
            }
        )

        return io.apply_transforms(
            source,
            target,
            mapping=mapping,
            transforms=[
                _import_language_qkv,
                _import_linear_fc1,
            ],
        )

    @property
    def tokenizer(self) -> "AutoTokenizer":
        from nemo.collections.common.tokenizers.huggingface.auto_tokenizer import AutoTokenizer

        return AutoTokenizer(str(self))

    @property
    def config(self) -> OpenVLAConfig:
        source = AutoConfig.from_pretrained("openvla/openvla-7b", trust_remote_code=True)

        assert source.hf_llm_id == 'meta-llama/Llama-2-7b-hf' # We use the Llama-2-7b model config directly

        language_transformer_config = Llama2Config7B(seq_length=source.text_config.max_position_embeddings,
                                                     bf16=True)
        # This hack is not  very pretty but it's ok to keep it for now
        object.__setattr__(language_transformer_config, "vocab_size", source.text_config.vocab_size)


        vision_model_primary = timm.create_model(
            source.timm_model_ids[0],
            pretrained=False,
            num_classes=0,
            img_size=source.image_sizes[0],
            act_layer=source.timm_override_act_layers[0],
        )
        vision_transformer_config = TimmCLIPVisionConfig(
            hidden_size = vision_model_primary.num_features,
            pretrained_model_name_or_path = source.timm_model_ids[0],
        )
        use_fused_vision_backbone = source.use_fused_vision_backbone
        if use_fused_vision_backbone:
            vision_model_secondary = timm.create_model(
                source.timm_model_ids[1],
                pretrained=False,
                num_classes=0,
                img_size=source.image_sizes[0],
                act_layer=source.timm_override_act_layers[1],
            )
            secondary_vision_transformer_config = TimmCLIPVisionConfig(
                hidden_size = vision_model_secondary.num_features,
                pretrained_model_name_or_path = source.timm_model_ids[1],
            )

        vision_projection_config = MultimodalProjectorConfig(
            num_vision_encoders=2 if use_fused_vision_backbone else 1,
            input_size=vision_model_primary.num_features + (secondary_vision_transformer_config.hidden_size if use_fused_vision_backbone else 0),
            output_size=source.text_config.hidden_size
        )
        output = OpenVLAConfig(
            language_transformer_config=language_transformer_config,
            vision_transformer_config=vision_transformer_config,
            secondary_vision_transformer_config= secondary_vision_transformer_config,
            vision_projection_config=vision_projection_config
        )
        return output

@dataclass
class OpenVLAVisionHFConfigDinoV2(TimmCLIPVisionConfig):
    hidden_size: int = 1024
    pretrained_model_name_or_path: str = "vit_large_patch14_reg4_dinov2.lvd142m"

@dataclass
class OpenVLAVisionHFConfigSigLip(TimmCLIPVisionConfig):
    hidden_size: int = 1152
    pretrained_model_name_or_path: str = "vit_so400m_patch14_siglip_224"

@dataclass
class OpenVLAProjectorHFConfig(MultimodalProjectorConfig):
    num_vision_encoders:int = 2
    input_size:int = 2176
    output_size: int = 4096

@dataclass
class OpenVLALlamaHFConfig(Llama2Config7B):
    seq_length: int = 2048
    vocab_size: int = 32064

@dataclass
class OpenVLAHFConfig(OpenVLAConfig):
    language_transformer_config: LlamaConfig = field(default_factory=OpenVLALlamaHFConfig)
    vision_transformer_config: TimmCLIPVisionConfig = field(default_factory=OpenVLAVisionHFConfigDinoV2)
    secondary_vision_transformer_config: TimmCLIPVisionConfig = field(default_factory=OpenVLAVisionHFConfigSigLip)
    vision_projection_config: MultimodalProjectorConfig = field(default_factory=OpenVLAProjectorHFConfig)


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


@io.state_transform(
    source_key=(
        "language_model.model.layers.*.mlp.gate_proj.weight",
        "language_model.model.layers.*.mlp.up_proj.weight",
    ),
    target_key="language_model.decoder.layers.*.mlp.linear_fc1.weight",
)
def _import_linear_fc1(down, gate):
    return torch.cat((down, gate), axis=0)

@io.state_transform(
    source_key=(
        "language_model.model.layers.*.self_attn.q_proj.weight",
        "language_model.model.layers.*.self_attn.k_proj.weight",
        "language_model.model.layers.*.self_attn.v_proj.weight",
    ),
    target_key="language_model.decoder.layers.*.self_attention.linear_qkv.weight",
)
def _import_language_qkv(ctx: io.TransformCTX, q, k, v):
    megatron_config = ctx.target.config.language_transformer_config
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
