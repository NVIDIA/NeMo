from dataclasses import dataclass

from nemo.collections.llm.bert.model.bert import HuggingFaceBertModel, HuggingFaceBertConfig


@dataclass
class ESM650MConfig(HuggingFaceBertConfig):
    num_layers: int = 33
    hidden_size: int = 1280
    ffn_hidden_size: int = 5120
    num_attention_heads: int = 20
    bert_binary_head: bool = False

@dataclass
class ESM3BConfig(HuggingFaceBertConfig):
    num_layers: int = 36
    hidden_size: int = 2560
    ffn_hidden_size: int = 10240
    num_attention_heads: int = 40
    bert_binary_head: bool = True

@dataclass
class ESM15BConfig(HuggingFaceBertConfig):
    num_layers: int = 48
    hidden_size: int = 5120
    ffn_hidden_size: int = 20480
    num_attention_heads: int = 40
    bert_binary_head: bool = False