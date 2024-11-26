from nemo.collections.llm.bert.model.bert import (HuggingFaceBertBaseConfig, HuggingFaceBertLargeConfig,
                                                  MegatronBertConfig, MegatronBertLargeConfig)

def test_huggingface_bert_base_config():
    config = HuggingFaceBertBaseConfig()
    assert config.bert_type == 'huggingface'
    assert config.num_layers == 12
    assert config.hidden_size == 768
    assert config.ffn_hidden_size == 3072
    assert config.num_attention_heads == 12

def test_huggingface_bert_large_config():
    config = HuggingFaceBertLargeConfig()
    assert config.bert_type == 'huggingface'
    assert config.num_layers == 24
    assert config.hidden_size == 1024
    assert config.ffn_hidden_size == 4096
    assert config.num_attention_heads == 16

def test_megatron_bert_base_config():
    config = MegatronBertConfig()
    assert config.bert_type == 'megatron'
    assert config.num_layers == 12
    assert config.hidden_size == 768
    assert config.ffn_hidden_size == 3072
    assert config.num_attention_heads == 12

def test_megatron_bert_large_config():
    config = MegatronBertLargeConfig()
    assert config.bert_type == 'megatron'
    assert config.num_layers == 24
    assert config.hidden_size == 1024
    assert config.ffn_hidden_size == 4096
    assert config.num_attention_heads == 16