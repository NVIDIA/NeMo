import itertools

import torch
from torch import Tensor
from torch.utils import benchmark

from nemo.collections.asr.modules import ConformerEncoder

CHECK_CORRECTNESS = True
torch.backends.cuda.matmul.allow_tf32 = False

p = 0.0
device = torch.device('cuda')


def product_dict(**kwargs):
    keys = kwargs.keys()
    vals = kwargs.values()
    for instance in itertools.product(*vals):
        yield dict(zip(keys, instance))


def create_tensors(shape, dtype, requires_grad=False):
    B, length, feature = shape
    audio_signal = torch.rand([B, feature, length], device=device, dtype=dtype, requires_grad=requires_grad)
    lengths = torch.zeros(B, device=device)
    lengths.fill_(length)
    return audio_signal, lengths


def benchmark_forward(shape, dtype, memory_efficient):
    encoder = ConformerEncoder(
        feat_in=80,
        feat_out=-1,
        n_layers=18,
        d_model=512,
        subsampling='striding',
        subsampling_factor=4,
        subsampling_conv_channels=-1,
        causal_downsampling=False,
        ff_expansion_factor=4,
        self_attention_model='abs_pos',  # should be rel pos, but mem efficient does not support.
        n_heads=8,
        att_context_size=[-1, -1],
        att_context_style='regular',
        xscaling=True,
        untie_biases=True,
        pos_emb_max_len=5000,
        conv_kernel_size=31,
        conv_norm_type='batch_norm',
        conv_context_size=None,
        dropout=0.1,
        dropout_pre_encoder=0.1,
        dropout_emb=0.0,
        dropout_att=0.1,
        memory_efficient=memory_efficient,
    ).to(device)

    if dtype == torch.bfloat16:
        encoder = encoder.bfloat16()
    elif dtype == torch.half:
        encoder = encoder.half()

    def forward_standard(audio_signal: Tensor, length: Tensor):
        return encoder(audio_signal=audio_signal, length=length)

    audio_signal, lengths = create_tensors(shape, dtype)

    dtype_str = {torch.bfloat16: "b16", torch.half: "f16", torch.float: "f32",}[dtype]
    sub_label = f"{dtype_str} B={audio_signal.size(0)}, T={audio_signal.size(2)}, F={audio_signal.size(1)}"

    return benchmark.Timer(
        stmt="fn(audio_signal, length)",
        globals={"audio_signal": audio_signal, "length": lengths, "fn": forward_standard,},
        label=f"standard",
        sub_label=sub_label,
    )


print("Standard Attention")
for seq_len in (1500, 2500):
    for dtype in (torch.float32, torch.bfloat16, torch.float16):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        mem_begin = torch.cuda.max_memory_allocated() / 2 ** 20

        if torch.float32 and seq_len > 15000:
            continue

        results = benchmark_forward(shape=(16, seq_len, 80), dtype=dtype, memory_efficient=False,)
        print(results.timeit(100))

        torch.cuda.synchronize()
        memory = torch.cuda.max_memory_allocated() / 2 ** 20 - mem_begin
        print(f"Memory used {memory:.2f}MiB")

print("Triton Memory Efficient")

for seq_len in (1500,):
    for dtype in (torch.bfloat16, torch.float16):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        mem_begin = torch.cuda.max_memory_allocated() / 2 ** 20

        results = benchmark_forward(shape=(16, seq_len, 80), dtype=dtype, memory_efficient=True,)
        print(results.timeit(100))

        torch.cuda.synchronize()
        memory = torch.cuda.max_memory_allocated() / 2 ** 20 - mem_begin
        print(f"Memory used {memory:.2f}MiB")
