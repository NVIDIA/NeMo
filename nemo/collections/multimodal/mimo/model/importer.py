from pathlib import Path

import torch
from transformers import LlavaForConditionalGeneration

from nemo.collections.common.tokenizers.huggingface.auto_tokenizer import AutoTokenizer
from nemo.collections.multimodal.mimo.config import CustomMimoConfig
from nemo.collections.multimodal.mimo.model.base import BaseMimoModel
from nemo.lightning import io, teardown


@io.model_importer(BaseMimoModel, "hf")
class HFLlavaMimoImporter(io.ModelConnector["LlavaForConditionalGeneration", BaseMimoModel]):
    def init(self) -> BaseMimoModel:
        return BaseMimoModel(self.config, tokenizer=self.tokenizer)

    def apply(self, output_path: Path) -> Path:

        source = LlavaForConditionalGeneration.from_pretrained(str(self))
        target = self.init()
        trainer = self.nemo_setup(target)
        self.convert_state(source, target)

        print(f"Converted Llava model to Nemo, saving to {output_path}")
        self.nemo_save(output_path, trainer)

        print(f"Converted Llava model saved to {output_path}")

        teardown(trainer, target)
        del trainer, target

        return output_path

    def convert_state(self, source, target):
        mapping = {}
        # vision module
        mapping.update(
            {
                "vision_tower.vision_model.embeddings.patch_embedding.weight": "vision_model.conv1.weight",
                "vision_tower.vision_model.embeddings.position_embedding.weight": "vision_model.position_embeddings.weight",
            }
        )
        # Update with pre-layer normalization
        mapping.update(
            {
                "vision_tower.vision_model.pre_layrnorm.weight": "vision_model.ln_pre.weight",
                "vision_tower.vision_model.pre_layrnorm.bias": "vision_model.ln_pre.bias",
            }
        )
        # Update with layer normalization layers
        mapping.update(
            {
                "vision_tower.vision_model.encoder.layers.*.layer_norm1.weight": "vision_model.decoder.layers.*.self_attention.linear_qkv.layer_norm_weight",
                "vision_tower.vision_model.encoder.layers.*.layer_norm1.bias": "vision_model.decoder.layers.*.self_attention.linear_qkv.layer_norm_bias",
                "vision_tower.vision_model.encoder.layers.*.layer_norm2.weight": "vision_model.decoder.layers.*.mlp.linear_fc1.layer_norm_weight",
                "vision_tower.vision_model.encoder.layers.*.layer_norm2.bias": "vision_model.decoder.layers.*.mlp.linear_fc1.layer_norm_bias",
            }
        )

        # Update with MLP layers (Feedforward block)
        mapping.update(
            {
                "vision_tower.vision_model.encoder.layers.*.mlp.fc1.weight": "vision_model.decoder.layers.*.mlp.linear_fc1.weight",
                "vision_tower.vision_model.encoder.layers.*.mlp.fc1.bias": "vision_model.decoder.layers.*.mlp.linear_fc1.bias",
                "vision_tower.vision_model.encoder.layers.*.mlp.fc2.weight": "vision_model.decoder.layers.*.mlp.linear_fc2.weight",
                "vision_tower.vision_model.encoder.layers.*.mlp.fc2.bias": "vision_model.decoder.layers.*.mlp.linear_fc2.bias",
            }
        )

        # Update with self-attention linear projection
        mapping.update(
            {
                "vision_tower.vision_model.encoder.layers.*.self_attn.out_proj.weight": "vision_model.decoder.layers.*.self_attention.linear_proj.weight",
                "vision_tower.vision_model.encoder.layers.*.self_attn.out_proj.bias": "vision_model.decoder.layers.*.self_attention.linear_proj.bias",
            }
        )

        # projection module

        mapping.update(
            {
                "multi_modal_projector.linear_1.weight": "vision_projection.encoder.linear_fc1.weight",
                "multi_modal_projector.linear_1.bias": "vision_projection.encoder.linear_fc1.bias",
                "multi_modal_projector.linear_2.weight": "vision_projection.encoder.linear_fc2.weight",
                "multi_modal_projector.linear_2.bias": "vision_projection.encoder.linear_fc2.bias",
            }
        )

        # Language model

        mapping.update(
            {
                # "language_model.model.embed_tokens.weight": "language_model.embedding.word_embeddings.weight",
                "language_model.model.layers.*.self_attn.o_proj.weight": "language_model.decoder.layers.*.self_attention.linear_proj.weight",
                "language_model.model.layers.*.mlp.down_proj.weight": "language_model.decoder.layers.*.mlp.linear_fc2.weight",
                "language_model.model.layers.*.input_layernorm.weight": "language_model.decoder.layers.*.self_attention.linear_qkv.layer_norm_weight",
                "language_model.model.layers.*.post_attention_layernorm.weight": "language_model.decoder.layers.*.mlp.linear_fc1.layer_norm_weight",
                "language_model.model.norm.weight": "language_model.decoder.final_layernorm.weight",
                # "language_model.lm_head.weight": "language_model.output_layer.weight",
            }
        )
        return io.apply_transforms(
            source,
            target,
            mapping=mapping,
            transforms=[
                _import_class_token,
                _import_linear_fc1,
                _import_language_qkv,
                _import_embed_tokens,
                _import_lm_head_weight,
                _import_vison_qkv,
                _transform_vision_qkv_bias,
            ],
        )

    @property
    def tokenizer(self) -> AutoTokenizer:
        """Returns the tokenizer with added special tokens, cached for reuse."""
        if not hasattr(self, "_tokenizer"):
            # Initialize and cache the tokenizer
            self._tokenizer = AutoTokenizer(str(self))

            # Define special tokens for images
            special_tokens = [f"IMG_{i}" for i in range(8)]

            # Add special tokens to the tokenizer
            self._tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})

        return self._tokenizer

    @property
    def config(self) -> CustomMimoConfig:
        image_special_tokens = [f"IMG_{i}" for i in range(8)]
        image_special_token_indices = [self.tokenizer.tokenizer.convert_tokens_to_ids(f"IMG_{i}") for i in range(8)]
        # vocab_size = get_vocab_size(self, self.tokenizer.vocab_size, 128)
        # print(f"new vocab_size {vocab_size}")
        return CustomMimoConfig(
            vocab_size=self.tokenizer.vocab_size,
            image_special_token_indices=image_special_token_indices,
            image_special_tokens=image_special_tokens,
        )


@io.state_transform(
    source_key=(
        "vision_tower.vision_model.encoder.layers.*.self_attn.q_proj.weight",
        "vision_tower.vision_model.encoder.layers.*.self_attn.k_proj.weight",
        "vision_tower.vision_model.encoder.layers.*.self_attn.v_proj.weight",
    ),
    target_key="vision_model.decoder.layers.*.self_attention.linear_qkv.weight",
)
def _import_vison_qkv(ctx: io.TransformCTX, q, k, v):
    megatron_config = ctx.target.vision_model.config
    head_num = megatron_config.num_attention_heads
    num_query_groups = megatron_config.num_query_groups
    heads_per_group = head_num // num_query_groups
    hidden_size = megatron_config.hidden_size
    head_size = megatron_config.kv_channels

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
        "language_model.model.layers.*.mlp.gate_proj.weight",
        "language_model.model.layers.*.mlp.up_proj.weight",
    ),
    target_key="language_model.decoder.layers.*.mlp.linear_fc1.weight",
)
def _import_linear_fc1(down, gate):
    return torch.cat((down, gate), axis=0)


@io.state_transform(
    source_key="vision_tower.vision_model.embeddings.class_embedding",
    target_key="vision_model.class_token",
)
def _import_class_token(ctx: io.TransformCTX, class_embedding):
    # Source shape: (1024,)
    # Target shape: (1, 1, 1024)

    # Reshape the class embedding to match the target shape
    class_token = class_embedding.view(1, 1, -1)

    # Ensure the transformation is correct
    assert class_token.shape == (1, 1, 1024), f"Expected shape (1, 1, 1024), but got {class_token.shape}"

    return class_token


@io.state_transform(
    source_key=(
        "language_model.model.layers.*.self_attn.q_proj.weight",
        "language_model.model.layers.*.self_attn.k_proj.weight",
        "language_model.model.layers.*.self_attn.v_proj.weight",
    ),
    target_key="language_model.decoder.layers.*.self_attention.linear_qkv.weight",
)
def _import_language_qkv(ctx: io.TransformCTX, q, k, v):
    megatron_config = ctx.target.language_model.config
    head_num = megatron_config.num_attention_heads
    num_query_groups = megatron_config.num_query_groups
    heads_per_group = head_num // num_query_groups
    hidden_size = megatron_config.hidden_size
    head_size = megatron_config.kv_channels

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
    source_key="language_model.model.embed_tokens.weight",
    target_key="language_model.embedding.word_embeddings.weight",
)
def _import_embed_tokens(ctx: io.TransformCTX, source_embed):

    target_shape = ctx.target.state_dict()["language_model.embedding.word_embeddings.weight"].shape

    target_vocab_size = target_shape[0]
    embedding_dim = target_shape[1]
    assert (
        source_embed.shape[1] == embedding_dim
    ), f"Embedding dimension mismatch: source={source_embed.shape[1]}, target={embedding_dim}"
    target_embed = torch.empty(target_vocab_size, embedding_dim, dtype=source_embed.dtype, device=source_embed.device)
    target_embed[: source_embed.shape[0], :] = source_embed
    average_embedding = source_embed.mean(dim=0)
    target_embed[source_embed.shape[0] :, :] = average_embedding

    return target_embed


@io.state_transform(
    source_key="language_model.lm_head.weight",
    target_key="language_model.output_layer.weight",
)
def _import_lm_head_weight(ctx: io.TransformCTX, source_weight):
    target_shape = ctx.target.state_dict()["language_model.output_layer.weight"].shape
    target_vocab_size, target_embedding_dim = target_shape
    source_vocab_size, source_embedding_dim = source_weight.shape

    # Ensure the embedding dimensions match between source and target
    assert target_embedding_dim == source_embedding_dim, (
        f"Embedding dimension mismatch: " f"source={source_embedding_dim}, target={target_embedding_dim}"
    )

    target_weight = torch.empty(
        target_vocab_size, target_embedding_dim, dtype=source_weight.dtype, device=source_weight.device
    )

    target_weight[:source_vocab_size, :] = source_weight

    average_weight = source_weight.mean(dim=0)
    target_weight[source_vocab_size:, :] = average_weight

    assert target_weight.shape == target_shape, f"Expected shape {target_shape}, but got {target_weight.shape}"

    return target_weight


@io.state_transform(
    source_key=(
        "vision_tower.vision_model.encoder.layers.*.self_attn.q_proj.bias",
        "vision_tower.vision_model.encoder.layers.*.self_attn.k_proj.bias",
        "vision_tower.vision_model.encoder.layers.*.self_attn.v_proj.bias",
    ),
    target_key="vision_model.decoder.layers.*.self_attention.linear_qkv.bias",
)
def _transform_vision_qkv_bias(ctx: io.TransformCTX, q_bias, k_bias, v_bias):
    """
    Transforms and concatenates Q, K, V biases from the source model to the target model.
    """

    # Concatenate the Q, K, V biases into a single bias tensor
    qkv_bias = torch.cat([q_bias, k_bias, v_bias], dim=0)  # (3072,)

    # Ensure the concatenated bias has the correct shape
    expected_shape = (3072,)
    assert qkv_bias.shape == expected_shape, f"Expected shape {expected_shape}, but got {qkv_bias.shape}"

    return qkv_bias
