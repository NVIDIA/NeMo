import tensorrt as trt

from ..model_config import (
    DECODER_GPT2,
    DECODER_GPTJ,
    DECODER_GPTNEXT,
    DECODER_LLAMA,
    QUANTIZATION_NONE,
)
from .gpt2 import GPT2DecoderLayer, GPT2DecoderLayerConfigBuilder
from .gptj import GPTJDecoderLayer, GPTJDecoderLayerConfigBuilder
from .gptnext import GPTNextDecoderLayer
from .llama import LLAMADecoderLayer, LLAMADecoderLayerConfigBuilder

DECODER_CONFIG_REGISTRY = {
    DECODER_GPT2: GPT2DecoderLayerConfigBuilder,
    DECODER_GPTJ: GPTJDecoderLayerConfigBuilder,
    DECODER_LLAMA: LLAMADecoderLayerConfigBuilder,
}


def build_decoder_layer_config(layer, decoder: str, dtype=trt.float16, rank=0, tensor_parallel=1):
    assert decoder in DECODER_CONFIG_REGISTRY, f"{decoder} not supported"
    return DECODER_CONFIG_REGISTRY[decoder](  # type: ignore
        decoder, dtype, rank, tensor_parallel
    ).build_layer(layer)


DECODER_REGISTRY = {
    DECODER_GPT2: GPT2DecoderLayer,
    DECODER_GPTJ: GPTJDecoderLayer,
    DECODER_LLAMA: LLAMADecoderLayer,
    DECODER_GPTNEXT: GPTNextDecoderLayer,
}


def build_decoder_layer(
    layer,
    num_layers: int,
    dtype=trt.float16,
    quantization=QUANTIZATION_NONE,
    rank=0,
    tensor_parallel=1,
):
    assert layer.decoder_type in DECODER_REGISTRY, f"{layer.decoder_type} not supported"
    return DECODER_REGISTRY[layer.decoder_type](
        layer, num_layers, dtype, quantization, rank, tensor_parallel
    )
