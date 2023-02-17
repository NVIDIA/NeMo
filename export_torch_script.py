import argparse
import io
import sys
import time
from collections import OrderedDict
from contextlib import nullcontext

import numpy as np
import onnxruntime
import torch
import torch.nn
import torch_tensorrt
import torch_tensorrt.fx.tracer.acc_tracer.acc_tracer as acc_tracer
import torchvision.models as models
from torch.profiler import ProfilerActivity, profile, record_function

import nemo
import nemo.collections.asr as nemo_asr
from nemo.collections.asr.models.ctc_bpe_models import EncDecCTCModelBPE
from nemo.core.classes.common import typecheck
from nemo.core.neural_types import (
    AudioSignal,
    ChannelType,
    LabelsType,
    LengthsType,
    LogprobsType,
    NeuralType,
    SpectrogramType,
)

nemo_file = '/git/models/streaming-conformer.nemo'


def get_dummy_input(asr_model, batch_size):
    cache_last_channel, cache_last_time, cache_last_channel_len = asr_model.encoder.get_initial_cache_state(
        batch_size=batch_size, device=torch.device("cuda"),
    )
    audio_signal = torch.randn((batch_size, 80, 57), device=torch.device("cuda"), requires_grad=False)
    length = torch.full((batch_size,), 57, device=torch.device("cuda"), requires_grad=False)
    return audio_signal, length, cache_last_channel, cache_last_time, cache_last_channel_len


def export_script(batch_size=128):
    asr_model = nemo_asr.models.ctc_bpe_models.EncDecCTCModelBPE.restore_from(nemo_file)
    with torch.inference_mode(), torch.no_grad():
        asr_model.to(torch.device("cuda")).freeze()
        asr_model.eval()
        asr_model.encoder.export_streaming_support = True
        asr_model.encoder.setup_streaming_params()

        inputs = get_dummy_input(asr_model, batch_size)

        # monkey patch forward
        asr_model.forward = asr_model.forward_for_export

        traced_model = torch.jit.trace(asr_model, inputs)
        return traced_model, inputs


class WrapNemoExport(torch.nn.Module):
    def __init__(self, asr_model):
        super().__init__()
        self.asr_model = asr_model

    def forward(self, audio_signal, length, cache_last_channel, cache_last_time, cache_last_channel_len):
        return self.asr_model.forward_for_export(
            input=audio_signal,
            length=length,
            cache_last_channel=cache_last_channel,
            cache_last_time=cache_last_time,
            cache_last_channel_len=cache_last_channel_len,
        )


def export_trt(batch_size=128):
    asr_model = nemo_asr.models.ctc_bpe_models.EncDecCTCModelBPE.restore_from(nemo_file)
    with torch.inference_mode(), torch.no_grad():
        asr_model.to(torch.device("cuda")).freeze()
        asr_model.eval()
        asr_model.encoder.export_streaming_support = True
        asr_model.encoder.eport_cache_support = True
        asr_model.encoder.setup_streaming_params()

        cache_last_channel = torch.randn(
            (batch_size, 18, 104, 512), dtype=torch.float32, device=torch.device("cuda:0"), requires_grad=False
        )
        cache_last_time = torch.randn(
            (batch_size, 18, 512, 30), dtype=torch.float32, device=torch.device("cuda:0"), requires_grad=False
        )
        cache_last_channel_len = torch.randint(
            1, 104, (batch_size,), dtype=torch.int64, device=torch.device("cuda:0"), requires_grad=False
        )
        audio_signal = torch.randn((batch_size, 80, 57), device=torch.device("cuda"), requires_grad=False)
        length = torch.full((batch_size,), 57, device=torch.device("cuda"), requires_grad=False)
        inputs = (audio_signal, length, cache_last_channel, cache_last_time, cache_last_channel_len)

        # monkey patch forward
        asr_model.forward = asr_model.forward_for_export
        traced_model = torch.jit.trace(asr_model, inputs)
        # torch.jit.save(traced_model, "streaming-conformer.pt")
        # return
        # serialized_engine = torch_tensorrt.convert_method_to_trt_engine(
        #    traced_model,
        #    "forward",
        torch_tensorrt.logging.set_reportable_log_level(torch_tensorrt.logging.Level.Debug)

        traced_model = torch_tensorrt.compile(
            traced_model,
            inputs=[
                torch_tensorrt.Input(
                    min_shape=[batch_size, 80, 57],
                    opt_shape=[batch_size, 80, 57],
                    max_shape=[batch_size, 80, 57],
                    dtype=torch.float32,
                ),
                torch_tensorrt.Input(
                    min_shape=[1], opt_shape=[batch_size], max_shape=[batch_size], dtype=torch.int32,
                ),
                torch_tensorrt.Input(
                    min_shape=[1, 18, 104, 512],
                    opt_shape=[batch_size, 18, 104, 512],
                    max_shape=[batch_size, 18, 104, 512],
                    dtype=torch.float32,
                ),
                torch_tensorrt.Input(
                    min_shape=[1, 18, 512, 30],
                    opt_shape=[batch_size, 18, 512, 30],
                    max_shape=[batch_size, 18, 512, 30],
                    dtype=torch.float32,
                ),
                torch_tensorrt.Input(
                    min_shape=[1], opt_shape=[batch_size], max_shape=[batch_size], dtype=torch.int32,
                ),
            ],
            enabled_precisions={torch.float32},
            truncate_long_and_double=True,
            require_full_compilation=True,
        )

    # with open("streaming-conformer.plan", "wb") as outf:
    #    outf.write(serialized_engine)
    torch.jit.save(traced_model, "streaming-conformer.pt")


def load_and_run(batch_size=128, model_fn=None, inputs=None):
    with open("streaming-conformer.pt", "rb") as f:
        buffer = io.BytesIO(f.read())
    asr_model = torch.jit.load(buffer, map_location=torch.device("cuda:0"))

    output = asr_model(*inputs)
    assert len(output) == 5
    print("logprobs", output[0].size())
    print("encoded_len", output[1].size())
    print("cache_last_channel_next", output[2].size())
    print("cache_last_time_next", output[3].size())
    print("cache_last_channel_next_len", output[4].size())


if __name__ == "__main__":
    model, inputs = export_script()
    # torch.jit.save(model, "streaming-conformer.pt")
    # load_and_run(128, "streaming-conformer.pt", inputs)
    export_trt(128)
