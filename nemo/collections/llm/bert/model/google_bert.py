import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Callable, Optional

import torch
from torch import nn

from nemo.collections.common.tokenizers import AutoTokenizer
from nemo.collections.llm.bert.model.base import BertConfig, BertModel
from nemo.collections.llm.utils import Config
from nemo.lightning import OptimizerModule, io, teardown


@dataclass
class GoogleBertConfig(BertConfig):
    """Configs for models in https://huggingface.co/google-bert"""

    bert_type: str = 'huggingface'
    add_pooler: bool = True
    init_method_std: float = 0.02
    hidden_dropout: float = 0.1
    normalization: float = 'LayerNorm'
    layernorm_epsilon: float = 1e-5
    apply_query_key_layer_scaling: bool = False
    position_embedding_type: str = "learned_absolute"


@dataclass
class GoogleBertBaseConfig(GoogleBertConfig):
    """Configs for model in https://huggingface.co/google-bert/bert-base-uncased"""

    num_layers: int = 12
    hidden_size: int = 768
    ffn_hidden_size: int = 3072
    num_attention_heads: int = 12


@dataclass
class GoogleBertLargeConfig(GoogleBertConfig):
    """Configs for model in https://huggingface.co/google-bert/bert-large-uncased"""

    num_layers: int = 24
    hidden_size: int = 1024
    ffn_hidden_size: int = 4096
    num_attention_heads: int = 16


class GoogleBertModel(BertModel):
    """Google Bert Model."""

    def __init__(
        self,
        config: Annotated[Optional[BertConfig], Config[BertConfig]] = None,
        optim: Optional[OptimizerModule] = None,
        tokenizer: Optional["TokenizerSpec"] = None,
        model_transform: Optional[Callable[[nn.Module], nn.Module]] = None,
    ):
        super().__init__(config or BertConfig(), optim=optim, tokenizer=tokenizer, model_transform=model_transform)


@io.model_importer(GoogleBertModel, "hf")
class HFGoogleBERTImporter(io.ModelConnector["BertForMaskedLM", BertModel]):
    """Importer Connector for converting HF Google Bert Model to NeMo"""

    def __init__(self, *args, **kwargs):
        if sys.version_info > (3, 11):
            # In Python versions <= 3.11, *Path classes don’t have a __init__ method,
            # and do all their initialization in __new__/ helper methods.
            # Only need to call super().__init__ if version > 3.11
            super().__init__(*args)
        self.type = kwargs.get('type', 'model')

    def init(self) -> GoogleBertModel:
        return GoogleBertModel(self.config, tokenizer=self.tokenizer)

    def apply(self, output_path: Path) -> Path:
        from transformers import BertForMaskedLM, BertForNextSentencePrediction, BertForPreTraining, BertModel

        source = BertForPreTraining.from_pretrained(str(self), torch_dtype='auto')
        # Depending on add_lm_head, bert_binary_head, we initialize the bert model differently:
        if self.type == 'model':
            source = BertModel.from_pretrained(str(self), torch_dtype='auto')
        elif self.type == 'pretraining':
            source = BertForPreTraining.from_pretrained(str(self), torch_dtype='auto')
        elif self.type == 'masked':
            source = BertForMaskedLM.from_pretrained(str(self), torch_dtype='auto')
        elif self.type == 'classification':
            source = BertForNextSentencePrediction.from_pretrained(str(self), torch_dtype='auto')

        print(
            f"Initializing Bert Model with pooler={self.config.add_pooler} "
            f"lm_head={self.config.add_lm_head}  binary_head={self.config.bert_binary_head}"
        )
        target = self.init()
        trainer = self.nemo_setup(target)

        breakpoint()
        self.convert_state(source, target)
        self.nemo_save(output_path, trainer)

        print(f"Converted Bert model to Nemo, model saved to {output_path}")

        teardown(trainer, target)
        del trainer, target

        return output_path

    def convert_state(self, source, target):
        """Converting HF state dict to NeMo state dict."""
        mapping = {
            "embeddings.position_embeddings.weight": "embedding.position_embeddings.weight",
            "embeddings.token_type_embeddings.weight": "embedding.tokentype_embeddings.weight",
            "embeddings.LayerNorm.weight": "encoder.initial_layernorm.weight",
            "embeddings.LayerNorm.bias": "encoder.initial_layernorm.bias",
            "encoder.layer.*.attention.output.dense.weight": "encoder.layers.*.self_attention.linear_proj.weight",
            "encoder.layer.*.attention.output.dense.bias": "encoder.layers.*.self_attention.linear_proj.bias",
            "encoder.layer.*.attention.output.LayerNorm.weight": "encoder.layers.*.post_att_layernorm.weight",
            "encoder.layer.*.attention.output.LayerNorm.bias": "encoder.layers.*.post_att_layernorm.bias",
            "encoder.layer.*.intermediate.dense.weight": "encoder.layers.*.mlp.linear_fc1.weight",
            "encoder.layer.*.intermediate.dense.bias": "encoder.layers.*.mlp.linear_fc1.bias",
            "encoder.layer.*.output.dense.weight": "encoder.layers.*.mlp.linear_fc2.weight",
            "encoder.layer.*.output.dense.bias": "encoder.layers.*.mlp.linear_fc2.bias",
            "encoder.layer.*.output.LayerNorm.weight": "encoder.layers.*.post_mlp_layernorm.weight",
            "encoder.layer.*.output.LayerNorm.bias": "encoder.layers.*.post_mlp_layernorm.bias",
        }
        if self.config.add_pooler:
            mapping.update(
                {
                    "pooler.dense.weight": "pooler.dense.weight",
                    "pooler.dense.bias": "pooler.dense.bias",
                }
            )

        # When instantiated HF's BertModel, or BertModelForPretraining, BertForMaskedLM, BertForNextSentencePrediction
        # The prefix for state dict is slightly different, therefore we need different transforms to take care of.
        if self.type == 'model':
            transforms = [_import_qkv_2, _import_qkv_bias_2, _import_embedding_2]
        else:
            transforms = [_import_qkv, _import_qkv_bias, _import_embedding]

        if self.type != 'model':
            # adding the 'bert.' prefix so that the state dict matches.
            mapping = {f'bert.{k}': v for k, v in mapping.items()}

        if self.config.add_lm_head:
            mapping.update(
                {
                    "cls.predictions.transform.dense.weight": "lm_head.dense.weight",
                    "cls.predictions.transform.dense.bias": "lm_head.dense.bias",
                    "cls.predictions.transform.LayerNorm.weight": "lm_head.layer_norm.weight",
                    "cls.predictions.transform.LayerNorm.bias": "lm_head.layer_norm.bias",
                }
            )
        if self.config.bert_binary_head:
            mapping.update(
                {
                    "cls.seq_relationship.weight": "binary_head.weight",
                    "cls.seq_relationship.bias": "binary_head.bias",
                }
            )
        return io.apply_transforms(source, target, mapping=mapping, transforms=transforms)

    @property
    def tokenizer(self) -> "AutoTokenizer":
        """Retrieve Tokenizer from HF"""
        from nemo.collections.common.tokenizers.huggingface.auto_tokenizer import AutoTokenizer

        return AutoTokenizer(self.save_hf_tokenizer_assets(str(self)))

    @property
    def config(self) -> BertConfig:
        """Generate NeMo Config based on HF config"""
        from transformers import BertConfig as HFBertConfig

        source = HFBertConfig.from_pretrained(str(self))

        output = GoogleBertConfig(
            bert_type='huggingface',
            num_layers=source.num_hidden_layers,
            hidden_size=source.hidden_size,
            ffn_hidden_size=source.intermediate_size,
            num_attention_heads=source.num_attention_heads,
            init_method_std=source.initializer_range,
            layernorm_epsilon=source.layer_norm_eps,
            seq_length=source.max_position_embeddings,
            add_lm_head=self.type == 'pretraining' or self.type == 'masked',
            bert_binary_head=self.type == 'pretraining' or self.type == 'classification',
            add_pooler=self.type != 'masked',
        )
        return output


@io.state_transform(
    source_key=(
        "bert.encoder.layer.*.attention.self.query.weight",
        "bert.encoder.layer.*.attention.self.key.weight",
        "bert.encoder.layer.*.attention.self.value.weight",
    ),
    target_key="encoder.layers.*.self_attention.linear_qkv.weight",
)
def _import_qkv(ctx: io.TransformCTX, q, k, v):
    # Bert Model does not support GQA
    megatron_config = ctx.target.config

    head_num = megatron_config.num_attention_heads
    hidden_size = megatron_config.hidden_size
    head_size = megatron_config.kv_channels

    old_tensor_shape = q.size()
    new_q_tensor_shape = (head_num, head_size) + old_tensor_shape[1:]

    q = q.view(*new_q_tensor_shape)
    k = k.view(*new_q_tensor_shape)
    v = v.view(*new_q_tensor_shape)

    qkv_weights = torch.empty((0, head_size) + old_tensor_shape[1:])
    for i in range(head_num):
        qkv_weights = torch.cat((qkv_weights, q[i : i + 1, :, :]))
        qkv_weights = torch.cat((qkv_weights, k[i : i + 1, :, :]))
        qkv_weights = torch.cat((qkv_weights, v[i : i + 1, :, :]))
    qkv_weights = qkv_weights.reshape([head_size * (3 * head_num), hidden_size])

    return qkv_weights


@io.state_transform(
    source_key=(
        "bert.encoder.layer.*.attention.self.query.bias",
        "bert.encoder.layer.*.attention.self.key.bias",
        "bert.encoder.layer.*.attention.self.value.bias",
    ),
    target_key="encoder.layers.*.self_attention.linear_qkv.bias",
)
def _import_qkv_bias(ctx: io.TransformCTX, qb, kb, vb):
    # Bert Model does not support GQA
    megatron_config = ctx.target.config

    head_num = megatron_config.num_attention_heads
    head_size = megatron_config.kv_channels

    new_q_tensor_shape_bias = (head_num, head_size)

    bias_q = qb.view(*new_q_tensor_shape_bias)
    bias_k = kb.view(*new_q_tensor_shape_bias)
    bias_v = vb.view(*new_q_tensor_shape_bias)

    qkv_biases = torch.empty((0, head_size))
    for i in range(head_num):
        qkv_biases = torch.cat((qkv_biases, bias_q[i : i + 1]))
        qkv_biases = torch.cat((qkv_biases, bias_k[i : i + 1]))
        qkv_biases = torch.cat((qkv_biases, bias_v[i : i + 1]))
    qkv_biases = qkv_biases.reshape([head_size * (3 * head_num)])

    return qkv_biases


@io.state_transform(
    source_key=("bert.embeddings.word_embeddings.weight",),
    target_key="embedding.word_embeddings.weight",
)
def _import_embedding(ctx: io.TransformCTX, embedding):
    divisible = ctx.target.config.make_vocab_size_divisible_by
    emb_size = embedding.size(0)
    padded_emb_size = int(math.ceil(emb_size / divisible) * divisible)
    print(padded_emb_size, divisible, emb_size)
    if padded_emb_size > emb_size:
        zeros_to_add = torch.zeros(
            padded_emb_size - emb_size,
            embedding.size(1),
            dtype=embedding.dtype,
            device=embedding.device,
        )
        # Concatenate the two tensors along rows
        padded_embedding = torch.cat((embedding, zeros_to_add), dim=0)
        print(padded_embedding.size())
        return padded_embedding
    return embedding


@io.state_transform(
    source_key=(
        "encoder.layer.*.attention.self.query.weight",
        "encoder.layer.*.attention.self.key.weight",
        "encoder.layer.*.attention.self.value.weight",
    ),
    target_key="encoder.layers.*.self_attention.linear_qkv.weight",
)
def _import_qkv_2(ctx: io.TransformCTX, q, k, v):
    # Bert Model does not support GQA
    megatron_config = ctx.target.config

    head_num = megatron_config.num_attention_heads
    hidden_size = megatron_config.hidden_size
    head_size = megatron_config.kv_channels

    old_tensor_shape = q.size()
    new_q_tensor_shape = (head_num, head_size) + old_tensor_shape[1:]

    q = q.view(*new_q_tensor_shape)
    k = k.view(*new_q_tensor_shape)
    v = v.view(*new_q_tensor_shape)

    qkv_weights = torch.empty((0, head_size) + old_tensor_shape[1:])
    for i in range(head_num):
        qkv_weights = torch.cat((qkv_weights, q[i : i + 1, :, :]))
        qkv_weights = torch.cat((qkv_weights, k[i : i + 1, :, :]))
        qkv_weights = torch.cat((qkv_weights, v[i : i + 1, :, :]))
    qkv_weights = qkv_weights.reshape([head_size * (3 * head_num), hidden_size])

    return qkv_weights


@io.state_transform(
    source_key=(
        "encoder.layer.*.attention.self.query.bias",
        "encoder.layer.*.attention.self.key.bias",
        "encoder.layer.*.attention.self.value.bias",
    ),
    target_key="encoder.layers.*.self_attention.linear_qkv.bias",
)
def _import_qkv_bias_2(ctx: io.TransformCTX, qb, kb, vb):
    # Bert Model does not support GQA
    megatron_config = ctx.target.config

    head_num = megatron_config.num_attention_heads
    head_size = megatron_config.kv_channels

    new_q_tensor_shape_bias = (head_num, head_size)

    bias_q = qb.view(*new_q_tensor_shape_bias)
    bias_k = kb.view(*new_q_tensor_shape_bias)
    bias_v = vb.view(*new_q_tensor_shape_bias)

    qkv_biases = torch.empty((0, head_size))
    for i in range(head_num):
        qkv_biases = torch.cat((qkv_biases, bias_q[i : i + 1]))
        qkv_biases = torch.cat((qkv_biases, bias_k[i : i + 1]))
        qkv_biases = torch.cat((qkv_biases, bias_v[i : i + 1]))
    qkv_biases = qkv_biases.reshape([head_size * (3 * head_num)])

    return qkv_biases


@io.state_transform(
    source_key=("embeddings.word_embeddings.weight",),
    target_key="embedding.word_embeddings.weight",
)
def _import_embedding_2(ctx: io.TransformCTX, embedding):
    divisible = ctx.target.config.make_vocab_size_divisible_by
    emb_size = embedding.size(0)
    padded_emb_size = int(math.ceil(emb_size / divisible) * divisible)
    print(padded_emb_size, divisible, emb_size)
    if padded_emb_size > emb_size:
        zeros_to_add = torch.zeros(
            padded_emb_size - emb_size,
            embedding.size(1),
            dtype=embedding.dtype,
            device=embedding.device,
        )
        # Concatenate the two tensors along rows
        padded_embedding = torch.cat((embedding, zeros_to_add), dim=0)
        print(padded_embedding.size())
        return padded_embedding
    return embedding
