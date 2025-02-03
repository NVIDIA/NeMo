from dataclasses import dataclass, field


@dataclass
class Base:
    x: int = 10  # Default value
    y: int = 20  # Default value

@dataclass
class Derived(Base):
    x: field = 50

d = Derived()
print(d)
# Derived(x=10, y=20)

@dataclass
class CLIPViTConfig:
    output_dim: int = 1024 # Getting this default from megatron_clip_VIT-H-14.yaml
    add_class_token: bool = True
    class_token_len: int = 8
    patch_dim: int = 16
    img_h: int = 224
    img_w: int = 224
    vision_model_type: str = "clip"  # ["clip", "siglip"]

    # Without these the init for transformer will give error
    num_layers: int = 1  # Placeholder, NOT used!
    num_attention_heads: int = 8  # Placeholder, NOT used!


@dataclass
class eeCLIPViTL_14_224_Config(Base):
    """Clip vit large patch14 config"""

    # TOdo these are probably not super upto date but that's ok
    # Will handle it later
    vision_model_type: str = "clip"
    patch_dim: int = 14
    img_h: int = 225
    img_w: int = 225
    num_layers: int = 24
    num_attention_heads: int = 16
    add_bias_linear: bool = True
    add_qkv_bias: bool = True
    hidden_size: int = 1024
    hidden_dropout: float = 0.0
    attention_dropout: float = 0.0
    ffn_hidden_size: int = 4096
    gated_linear_unit: bool = False
    kv_channels: int = 64
    num_query_groups: int = 16
    layernorm_zero_centered_gamma: bool = False
    apply_query_key_layer_scaling: bool = False
    bias_activation_fusion: bool = False
    bias_dropout_fusion: bool = False
    attention_softmax_in_fp32: bool = True
    normalization: str = 'LayerNorm'
    apply_rope_fusion: bool = False

# Create an instance of Derived
obj = eeCLIPViTL_14_224_Config()
print(obj)