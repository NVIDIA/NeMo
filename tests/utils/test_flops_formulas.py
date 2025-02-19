import pytest

from nemo.utils.flops_formulas import FLOPSConfig, bert, gpt3, llama2, llama3, mixtral, nemotron
from nemo.utils.hyena_flops_formulas import hyena


@pytest.fixture
def flops_config():
    return FLOPSConfig(
        gbs=1,
        enc_seq_len=128,
        hs=768,
        layers=12,
        ffn_hs=3072,
        attention_heads=12,
        moe_router_topk=2,
        query_groups=12,
        vocab_size=50257,
        model_pattern="SDH*",
    )


def test_gpt3(flops_config):
    expected_flops = 97240743936
    assert gpt3(flops_config) == expected_flops


def test_llama2(flops_config):
    expected_flops = 107659395072.0
    assert llama2(flops_config) == expected_flops


def test_llama3(flops_config):
    expected_flops = 164433494016.0
    assert llama3(flops_config) == expected_flops


def test_nemotron(flops_config):
    expected_flops = 218036699136.0
    assert nemotron(flops_config) == expected_flops


def test_mixtral(flops_config):
    expected_flops = 172889210880.0
    assert mixtral(flops_config) == expected_flops


def test_bert(flops_config):
    expected_flops = 84146651135.99998
    assert bert(flops_config) == expected_flops


def test_hyena(flops_config):
    expected_flops = 116883062784.0
    assert hyena(flops_config) == expected_flops
