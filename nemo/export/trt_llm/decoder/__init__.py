from transformers.models.gpt2.modeling_gpt2 import GPT2Block
from transformers.models.gptj.modeling_gptj import GPTJBlock
from transformers.models.llama.modeling_llama import LlamaDecoderLayer

from .gpt2 import GPT2DecoderLayer
from .gptj import GPTJDecoderLayer
from .llama import LLAMADecoderLayer
from .nemo import NemoDecoderLayer, NemoLayer

DECODER_REGISTRY = {
    GPT2Block: GPT2DecoderLayer,
    GPTJBlock: GPTJDecoderLayer,
    LlamaDecoderLayer: LLAMADecoderLayer,
    NemoLayer: NemoDecoderLayer,
}


def build_decoder_layer(layer, num_layers, dtype, rank, tensor_parallel):
    assert type(layer) in DECODER_REGISTRY, f"{layer} not supported"
    return DECODER_REGISTRY[type(layer)](layer, num_layers, dtype, rank, tensor_parallel)
