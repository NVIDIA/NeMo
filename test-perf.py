import argparse
import io
import sys
import time
from collections import OrderedDict
from contextlib import nullcontext

import numpy as np
import onnxruntime
import torch
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


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


def create_ort_gpu_value(x, device="cuda", device_id=0):
    return onnxruntime.OrtValue.ortvalue_from_numpy(x.detach().cpu().numpy(), device, device_id)


def main_perf_onnx_full_ctx(batch_size=128, profile=True):
    # Inputs
    audio_signal = create_ort_gpu_value(
        torch.randn((batch_size, 80, 57), device=torch.device("cpu"), dtype=torch.float32, requires_grad=False)
    )
    length = create_ort_gpu_value(
        torch.full((batch_size,), 57, device=torch.device("cpu"), dtype=torch.int64, requires_grad=False)
    )

    # Outputs
    logprobs = create_ort_gpu_value(
        torch.zeros((batch_size, 15, 1025), device=torch.device("cpu"), dtype=torch.float32, requires_grad=False)
    )

    opts = onnxruntime.SessionOptions()
    opts.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
    if profile:
        opts.enable_profiling = True
    sess = onnxruntime.InferenceSession(
        "full-context.onnx",
        providers=[
            (
                'CUDAExecutionProvider',
                {
                    "gpu_mem_limit": 24 * 1024 * 1024 * 1024,
                    "do_copy_in_default_stream": False,
                    "cudnn_conv_algo_search": "EXHAUSTIVE",
                },
            ),
        ],
        sess_options=opts,
    )
    sess.disable_fallback()
    io_binding = sess.io_binding()

    io_binding.bind_input("audio_signal", "cuda", 0, np.float32, [batch_size, 80, 57], audio_signal.data_ptr())
    io_binding.bind_input("length", "cuda", 0, np.int64, [batch_size,], length.data_ptr())

    ort_outs = sess.get_outputs()

    io_binding.bind_output(ort_outs[0].name, "cuda", 0, np.float32, [batch_size, 15, 1025], logprobs.data_ptr())

    io_binding.synchronize_inputs()
    total_time = []
    total_audio_len = []
    for _ in range(64):
        start = time.time()
        sess.run_with_iobinding(io_binding)
        io_binding.synchronize_outputs()
        stop = time.time()
        total_time.append(stop - start)
        total_audio_len.append((57 / 100) * batch_size)
    print("============================================================")
    print("Streaming onnx Performance")
    print(f"batch size {batch_size}")
    print(f"Inference took {sum(total_time)}s")
    print(f"Total audio {sum(total_audio_len)}s")
    print(f"min RTFx {total_audio_len[0] / min(total_time)}")
    print(f"mean RTFx {sum(total_audio_len) / sum(total_time)}")
    print("============================================================")

    if profile:
        prof_file = sess.end_profiling()
        print("Saving onnx profile to ", prof_file)


def main_perf_pt(batch_size=128):
    with open("streaming-conformer.ts", "rb") as f:
        buffer = io.BytesIO(f.read())
    asr_model = torch.jit.load(buffer, map_location=torch.device("cuda"))

    cache_last_channel = torch.zeros(
        (batch_size, 18, 104, 512), dtype=torch.float32, device=torch.device("cuda:0"), requires_grad=False
    )
    cache_last_time = torch.zeros(
        (batch_size, 18, 512, 30), dtype=torch.float32, device=torch.device("cuda:0"), requires_grad=False
    )
    cache_last_channel_len = torch.zeros(
        (batch_size,), dtype=torch.int64, device=torch.device("cuda:0"), requires_grad=False
    )
    audio_signal = torch.randn((batch_size, 80, 57), device=torch.device("cuda"), requires_grad=False)
    length = torch.full((batch_size,), 57, device=torch.device("cuda"), requires_grad=False)

    total_time = []
    total_audio_len = []

    for _ in range(4):
        start = time.time()
        (log_probs, encoded_len, cache_last_channel, cache_last_time, cache_last_channel_len,) = asr_model(
            input=audio_signal,
            length=length,
            cache_last_channel=cache_last_channel,
            cache_last_time=cache_last_time,
            cache_last_channel_len=cache_last_channel_len,
        )
        stop = time.time()

    for _ in range(64):
        start = time.time()
        (log_probs, encoded_len, cache_last_channel, cache_last_time, cache_last_channel_len,) = asr_model(
            input=audio_signal,
            length=length,
            cache_last_channel=cache_last_channel,
            cache_last_time=cache_last_time,
            cache_last_channel_len=cache_last_channel_len,
        )
        stop = time.time()
        total_time.append(stop - start)
        total_audio_len.append((57 / 100) * batch_size)
    print("============================================================")
    print("Streaming Performance")
    print(f"batch size {batch_size}")
    print(f"Inference took {sum(total_time)}s")
    print(f"Total audio {sum(total_audio_len)}s")
    print(f"min RTFx {total_audio_len[0] / min(total_time)}")
    print(f"mean RTFx {sum(total_audio_len) / sum(total_time)}")
    print("============================================================")


def main_perf_onnx(batch_size=128, profile=True):
    # Inputs
    cache_last_channel = create_ort_gpu_value(
        torch.randn((batch_size, 18, 104, 512), device=torch.device("cpu"), dtype=torch.float32, requires_grad=False)
    )
    cache_last_time = create_ort_gpu_value(
        torch.randn((batch_size, 18, 512, 30), device=torch.device("cpu"), dtype=torch.float32, requires_grad=False)
    )
    cache_last_channel_len = create_ort_gpu_value(
        torch.randint(1, 104, (batch_size,), device=torch.device("cpu"), dtype=torch.int64, requires_grad=False)
    )

    audio_signal = create_ort_gpu_value(
        torch.randn((batch_size, 80, 57), device=torch.device("cpu"), dtype=torch.float32, requires_grad=False)
    )
    length = create_ort_gpu_value(
        torch.full((batch_size,), 57, device=torch.device("cpu"), dtype=torch.int64, requires_grad=False)
    )

    # Outputs
    logprobs = create_ort_gpu_value(
        torch.zeros((batch_size, 13, 1025), device=torch.device("cpu"), dtype=torch.float32, requires_grad=False)
    )
    encoded_len = create_ort_gpu_value(
        torch.zeros((batch_size,), device=torch.device("cpu"), dtype=torch.int32, requires_grad=False)
    )
    cache_last_channel_next = create_ort_gpu_value(
        torch.zeros((batch_size, 18, 104, 512), device=torch.device("cpu"), dtype=torch.float32, requires_grad=False)
    )
    cache_last_time_next = create_ort_gpu_value(
        torch.zeros((batch_size, 18, 512, 30), device=torch.device("cpu"), dtype=torch.float32, requires_grad=False)
    )
    cache_last_channel_next_len = create_ort_gpu_value(
        torch.zeros((batch_size,), device=torch.device("cpu"), dtype=torch.int64, requires_grad=False)
    )

    opts = onnxruntime.SessionOptions()
    opts.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
    if profile:
        opts.enable_profiling = True
    sess = onnxruntime.InferenceSession(
        "/git/models/streaming-conformer-14.onnx",
        providers=[
            (
                'CUDAExecutionProvider',
                {
                    "gpu_mem_limit": 40 * 1024 * 1024 * 1024,
                    "do_copy_in_default_stream": False,
                    "cudnn_conv_algo_search": "EXHAUSTIVE",
                },
            )
        ],
        sess_options=opts,
    )
    sess.disable_fallback()
    io_binding = sess.io_binding()

    io_binding.bind_input("audio_signal", "cuda", 0, np.float32, [batch_size, 80, 57], audio_signal.data_ptr())
    io_binding.bind_input("length", "cuda", 0, np.int64, [batch_size,], length.data_ptr())
    io_binding.bind_input(
        "cache_last_channel", "cuda", 0, np.float32, [batch_size, 18, 104, 512], cache_last_channel.data_ptr()
    )
    io_binding.bind_input(
        "cache_last_time", "cuda", 0, np.float32, [batch_size, 18, 512, 30], cache_last_time.data_ptr()
    )
    io_binding.bind_input(
        "cache_last_channel_len", "cuda", 0, np.int64, [batch_size,], cache_last_channel_len.data_ptr()
    )

    io_binding.bind_output("logprobs", "cuda", 0, np.float32, [batch_size, 13, 1025], logprobs.data_ptr())
    io_binding.bind_output("encoded_lengths", "cuda", 0, np.int32, [batch_size,], encoded_len.data_ptr())
    io_binding.bind_output(
        "cache_last_channel_next", "cuda", 0, np.float32, [batch_size, 18, 104, 512], cache_last_channel.data_ptr(),
    )
    io_binding.bind_output(
        "cache_last_time_next", "cuda", 0, np.float32, [batch_size, 18, 512, 30], cache_last_time.data_ptr()
    )
    io_binding.bind_output(
        "cache_last_channel_next_len", "cuda", 0, np.int64, [batch_size,], cache_last_channel_len.data_ptr()
    )

    io_binding.synchronize_inputs()
    total_time = []
    total_audio_len = []
    for _ in range(64):
        start = time.time()
        sess.run_with_iobinding(io_binding)
        # io_binding.synchronize_outputs()
        stop = time.time()
        total_time.append(stop - start)
        total_audio_len.append((57 / 100) * batch_size)
    print("============================================================")
    print("Streaming onnx Performance")
    print(f"batch size {batch_size}")
    print(f"Inference took {sum(total_time)}s")
    print(f"Total audio {sum(total_audio_len)}s")
    print(f"min RTFx {total_audio_len[0] / min(total_time)}")
    print(f"mean RTFx {sum(total_audio_len) / sum(total_time)}")
    print("============================================================")

    if profile:
        prof_file = sess.end_profiling()
        print("Saving onnx profile to ", prof_file)


nemo_file = '/git/models/streaming-conformer.nemo'


def main_perf(batch_size=128):
    asr_model = nemo_asr.models.ctc_bpe_models.EncDecCTCModelBPE.restore_from(nemo_file)
    with torch.inference_mode(), torch.no_grad():
        asr_model.to(torch.device("cuda")).freeze()
        asr_model.eval()
        asr_model.encoder.export_streaming_support = False
        asr_model.encoder.setup_streaming_params()

        cache_last_channel, cache_last_time, cache_last_channel_len = asr_model.encoder.get_initial_cache_state(
            batch_size=batch_size, device=torch.device("cuda"),
        )
        audio_signal = torch.randn((batch_size, 80, 57), device=torch.device("cuda"), requires_grad=False)
        length = torch.full((batch_size,), 57, device=torch.device("cuda"), requires_grad=False)

        total_audio_len = []
        total_time = []
        for _ in range(64):
            start = time.time()
            (
                log_probs,
                encoded_len,
                cache_last_channel,
                cache_last_time,
                cache_last_channel_len,
            ) = asr_model.forward_for_export(
                input=audio_signal,
                length=length,
                cache_last_channel=cache_last_channel,
                cache_last_time=cache_last_time,
                cache_last_channel_len=cache_last_channel_len,
            )
            stop = time.time()
            total_time.append(stop - start)
            total_audio_len.append((57 / 100) * batch_size)
        print("============================================================")
        print("Streaming Performance")
        print(f"batch size {batch_size}")
        print(f"Inference took {sum(total_time)}s")
        print(f"Total audio {sum(total_audio_len)}s")
        print(f"min RTFx {total_audio_len[0] / min(total_time)}")
        print(f"mean RTFx {sum(total_audio_len) / sum(total_time)}")
        print("============================================================")


def main_perf_full_context(batch_size=128):
    asr_model = nemo_asr.models.ctc_bpe_models.EncDecCTCModelBPE.restore_from(nemo_file)
    with torch.inference_mode(), torch.no_grad():
        asr_model.to(torch.device("cuda")).freeze()
        asr_model.eval()
        asr_model.encoder.export_streaming_support = True
        asr_model.encoder.setup_streaming_params()

        audio_signal = torch.randn((batch_size, 80, 57), device=torch.device("cuda"))
        length = torch.full((batch_size,), 57, device=torch.device("cuda"))

        total_audio_len = []
        total_time = []
        for _ in range(10):
            start = time.time()
            log_probs = asr_model.forward_for_export(input=audio_signal, length=length,)
            stop = time.time()
            total_time.append(stop - start)
            total_audio_len.append((57 / 100) * batch_size)
        print("============================================================")
        print("Full context Performance")
        print(f"batch size {batch_size}")
        print(f"Inference took {sum(total_time)}s")
        print(f"Total audio {sum(total_audio_len)}s")
        print(f"min RTFx {total_audio_len[0] / min(total_time)}")
        print(f"mean RTFx {sum(total_audio_len) / sum(total_time)}")
        print("============================================================")


if __name__ == "__main__":
    bs = int(sys.argv[1])

    # with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
    #    with record_function("model_inference"):
    # with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True, profile_memory=True, with_stack=True) as prof:
    #    with record_function("model_inference"):
    # main_perf_onnx(bs, False)
    # for bs in (128,):
    # main_perf(bs)
    main_perf_pt(bs)

    # print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=100))
    # print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
    # print(prof.key_averages().table(sort_by="cpu_memory_usage", row_limit=10))
    # print(prof.key_averages(group_by_stack_n=5).table(sort_by="self_cuda_time_total", row_limit=2))

    # prof.export_chrome_trace("trace2.json")
    # prof.export_stacks("/tmp/profiler_stacks.txt", "self_cuda_time_total")
