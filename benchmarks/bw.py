import torch
from torch.cuda.amp import autocast
from torch.utils import benchmark
from torch.utils.benchmark.utils.common import select_unit

from nemo.collections.asr.modules import ConformerEncoder

torch.backends.cuda.matmul.allow_tf32 = False

device = torch.device('cuda')


def _create_tensors(shape, dtype, requires_grad=False):
    B, length, feature = shape
    audio_signal = torch.rand([B, feature, length], device=device, requires_grad=requires_grad)
    lengths = torch.zeros(B, device=device)
    lengths.fill_(length)
    return audio_signal, lengths


def _create_model(dtype, hidden_size, memory_efficient, num_heads):
    encoder = ConformerEncoder(
        feat_in=80,
        feat_out=-1,
        n_layers=18,
        d_model=hidden_size,
        subsampling='striding',
        subsampling_factor=4,
        subsampling_conv_channels=-1,
        causal_downsampling=False,
        ff_expansion_factor=4,
        self_attention_model='alibi_pos',
        n_heads=num_heads,
        # att_context_size=[-1, -1],
        # att_context_style='regular',
        # xscaling=True,
        # untie_biases=True,
        # pos_emb_max_len=5000,
        conv_kernel_size=31,
        conv_norm_type='batch_norm',
        conv_context_size=None,
        dropout=0.1,
        dropout_pre_encoder=0.1,
        dropout_emb=0.0,
        dropout_att=0.0,
        memory_efficient=memory_efficient,
    ).to(device)
    return encoder


def benchmark_forward(shape, dtype, hidden_size, memory_efficient, num_heads):
    encoder = _create_model(dtype, hidden_size, memory_efficient, num_heads=num_heads)

    audio_signal, lengths = _create_tensors(shape, dtype)

    def forward(audio_signal, length):
        with autocast(dtype=dtype):
            return encoder(audio_signal=audio_signal, length=length)

    dtype_str = {torch.bfloat16: "b16", torch.half: "f16", torch.float: "f32",}[dtype]
    sub_label = f"{dtype_str} B={audio_signal.size(0)}, T={audio_signal.size(2)}, F={audio_signal.size(1)}"

    return benchmark.Timer(
        stmt="fn(audio_signal, length)",
        globals={"audio_signal": audio_signal, "length": lengths, "fn": forward,},
        label=f"standard",
        sub_label=sub_label,
    )


def benchmark_backward(shape, dtype, hidden_size, memory_efficient, num_heads):
    encoder = _create_model(dtype, hidden_size, memory_efficient, num_heads)

    audio_signal, lengths = _create_tensors(shape, dtype)

    out, length = encoder(audio_signal=audio_signal, length=lengths)
    grad_benchmark = torch.ones_like(out)

    dtype_str = {torch.bfloat16: "b16", torch.half: "f16", torch.float: "f32",}[dtype]
    sub_label = f"{dtype_str} B={audio_signal.size(0)}, T={audio_signal.size(2)}, F={audio_signal.size(1)}"

    return benchmark.Timer(
        stmt="out.backward(grad, retain_graph=True)",
        globals={"out": out, "grad": grad_benchmark,},
        label=f"backward",
        sub_label=sub_label,
    )


def benchmark_forward_backward(shape, dtype, hidden_size, memory_efficient, num_heads):
    encoder = _create_model(dtype, hidden_size, memory_efficient, num_heads)

    audio_signal, lengths = _create_tensors(shape, dtype)

    def forward_backward(audio_signal, length):
        with autocast(dtype=dtype):
            out, length = encoder(audio_signal=audio_signal, length=length)
            grad = torch.ones_like(out)
            out.backward(grad, retain_graph=True)

    dtype_str = {torch.bfloat16: "b16", torch.half: "f16", torch.float: "f32",}[dtype]
    sub_label = f"{dtype_str} B={audio_signal.size(0)}, T={audio_signal.size(2)}, F={audio_signal.size(1)}"

    return benchmark.Timer(
        stmt="fn(audio_signal, length)",
        globals={"audio_signal": audio_signal, "length": lengths, "fn": forward_backward,},
        label=f"standard",
        sub_label=sub_label,
    )


def run(dtype, seq_len, bs, hidden_size, benchmark, memory_efficient, num_heads):
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    mem_begin = torch.cuda.max_memory_allocated() / 2 ** 20

    if benchmark == "FWD":
        m = benchmark_forward(
            shape=(bs, seq_len, 80),
            dtype=dtype,
            hidden_size=hidden_size,
            memory_efficient=memory_efficient,
            num_heads=num_heads,
        )

    elif benchmark == "BWD":
        m = benchmark_backward(
            shape=(bs, seq_len, 80),
            dtype=dtype,
            hidden_size=hidden_size,
            memory_efficient=memory_efficient,
            num_heads=num_heads,
        )
    else:
        m = benchmark_forward_backward(
            shape=(bs, seq_len, 80),
            dtype=dtype,
            hidden_size=hidden_size,
            memory_efficient=memory_efficient,
            num_heads=num_heads,
        )

    measurement = m.timeit(10)
    torch.cuda.synchronize()
    memory = torch.cuda.max_memory_allocated() / 2 ** 20 - mem_begin

    time_unit, time_scale = select_unit(measurement.median)
    raw_time = measurement.median / time_scale
    time = f"{raw_time:.2f}{time_unit}"
    return memory, time, raw_time


batch_sizes = (8,)
hidden_dims = (1024,)
seq_len = 1500
for name in (
    "FWD",
    "FWD/BWD",
):
    print(f'\nRunning {name} tests')
    for bs, hidden_size in zip(batch_sizes, hidden_dims):
        print(f"running {bs} {hidden_size}")
        for dtype in (torch.bfloat16,):
            for heads in (16, 32, 64):
                print(f"Heads: {heads}")
                memory, time, r = run(
                    dtype, seq_len, bs, hidden_size, benchmark=name, memory_efficient=False, num_heads=heads
                )
                print(f"Standard Attention hidden_dim={hidden_size} T={seq_len} dtype={dtype} MiB={memory} ms={time}")

                trit_memory, trit_time, trit_r = run(
                    dtype, seq_len, bs, hidden_size, benchmark=name, memory_efficient=True, num_heads=heads
                )
                print(
                    f"Triton Attention hidden_dim={hidden_size} T={seq_len} dtype={dtype} MiB={trit_memory} ms={trit_time}\n"
                )

                print(f"Improvement: MiB={memory / trit_memory} ms={r / trit_r}\n")
