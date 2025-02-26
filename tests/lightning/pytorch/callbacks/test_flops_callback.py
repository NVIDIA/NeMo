import pytest
import torch
import lightning.pytorch as pl
from nemo.lightning.pytorch.callbacks.flops_callback import FLOPsMeasurementCallback
from nemo.collections.llm.gpt.model.base import GPTConfig
from nemo.collections.llm.gpt.model.hyena import HyenaConfig

class MockDataModule:
    def __init__(self, global_batch_size, vocab_size):
        self.global_batch_size = global_batch_size
        self.tokenizer = self
        self.vocab_size = vocab_size



def test_flops_measurement_callback_bert():
    model_config = GPTConfig(
        seq_length=128,
        hidden_size=768,
        num_layers=12,
        ffn_hidden_size=3072,
        num_attention_heads=12,
        moe_router_topk=0,
        num_query_groups=12)

    train_step_time = 1.23
    global_batch_size=1
    num_devices = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
    model_name = "bert"
    data_module = MockDataModule(global_batch_size=global_batch_size, vocab_size=100)
    callback = FLOPsMeasurementCallback(model_config, data_module, model_name)
    total_flops, flops_per_gpu = callback.eval_model_flops()

    expected_total_flops = 84146651135.99998
    expected_flops_per_gpu = expected_total_flops / num_devices

    assert total_flops == expected_total_flops
    assert flops_per_gpu ==  expected_flops_per_gpu

    tflops_per_sec_per_gpu = callback.eval_tflops_per_sec_per_gpu(train_step_time)
    expected_tflops_per_sec_per_gpu = expected_flops_per_gpu / (1e12 * train_step_time)
    assert tflops_per_sec_per_gpu == expected_tflops_per_sec_per_gpu