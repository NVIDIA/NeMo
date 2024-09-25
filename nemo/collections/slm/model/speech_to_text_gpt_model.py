import torch
from megatron.core import parallel_state, tensor_parallel
from megatron.core.transformer import MegatronModule

from nemo.collections.llm.gpt.model import GPTModel


class SpeechToTextLLM(MegatronModule):
    pass
