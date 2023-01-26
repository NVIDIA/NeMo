import torch
from torch.cuda.amp import autocast
from torch.utils import benchmark
from torch.utils.benchmark.utils.common import select_unit

from nemo.collections.asr.parts.submodules.conformer_modules import ConformerLayer

torch.backends.cuda.matmul.allow_tf32 = False

device = torch.device('cuda')


def _create_tensors(batch_size, D):
    audio_signal = torch.randn(batch_size, seq_len, D, device=device, dtype=dtype)
    mask = torch.ones(batch_size, seq_len, seq_len, device=device).bool()
    return audio_signal, mask


def _create_model(hidden_size, memory_efficient, num_heads):
    encoder = ConformerLayer(
        d_model=hidden_size,
        d_ff=hidden_size * 4,
        self_attention_model='alibi_pos',
        n_heads=num_heads,
        conv_kernel_size=9,
        conv_norm_type='layer_norm',
        conv_context_size=None,
        dropout=0.1,
        dropout_att=0.0,
        memory_efficient=memory_efficient,
    ).to(device)
    return encoder


def benchmark_forward(shape, dtype, hidden_size, memory_efficient, num_heads):
    encoder = _create_model(hidden_size, memory_efficient, num_heads)

    audio_signal, att_mask = _create_tensors(shape[0], shape[-1])

    def forward(audio_signal, att_mask):
        with autocast(dtype=dtype):
            out = encoder(x=audio_signal, att_mask=att_mask, pad_mask=None,)

    return benchmark.Timer(
        stmt="fn(audio_signal, att_mask)",
        globals={"audio_signal": audio_signal, "att_mask": att_mask, "fn": forward},
        label=f"standard",
    )


def benchmark_forward_backward(shape, dtype, hidden_size, memory_efficient, num_heads):
    encoder = _create_model(hidden_size, memory_efficient, num_heads)

    audio_signal, att_mask = _create_tensors(shape[0], shape[-1])

    def forward_bwd(audio_signal, att_mask):
        with autocast(dtype=dtype):
            out = encoder(x=audio_signal, att_mask=att_mask, pad_mask=None,)
            grad = torch.ones_like(out)
            out.backward(grad, retain_graph=True)

    return benchmark.Timer(
        stmt="fn(audio_signal, att_mask)",
        globals={"audio_signal": audio_signal, "att_mask": att_mask, "fn": forward_bwd},
        label=f"standard",
    )


def run(dtype, seq_len, bs, hidden_size, benchmark, memory_efficient, num_heads):
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    mem_begin = torch.cuda.max_memory_allocated() / 2 ** 20

    if benchmark == "FWD":
        m = benchmark_forward(
            shape=(bs, seq_len, hidden_size),
            dtype=dtype,
            hidden_size=hidden_size,
            memory_efficient=memory_efficient,
            num_heads=num_heads,
        )
    else:
        m = benchmark_forward_backward(
            shape=(bs, seq_len, hidden_size),
            dtype=dtype,
            hidden_size=hidden_size,
            memory_efficient=memory_efficient,
            num_heads=num_heads,
        )
    measurement = m.timeit(100)
    memory = torch.cuda.max_memory_allocated() / 2 ** 20 - mem_begin

    time_unit, time_scale = select_unit(measurement.median)
    raw_time = measurement.median / time_scale
    time = f"{raw_time:.2f}{time_unit}"
    return memory, time, raw_time


batch_sizes = (16,)
hidden_dims = (1024,)
seq_len = 512
for name in ("FWD", "FWD/BWD"):
    print(f'\nRunning {name} tests')
    for bs, hidden_size in zip(batch_sizes, hidden_dims):
        print(f"running {bs} {hidden_size}")
        for dtype in (torch.bfloat16,):
            for heads in (16, 32, 64):
                print(f"Heads: {heads}")
                trit_memory, trit_time, trit_r = run(
                    dtype, seq_len, bs, hidden_size, benchmark=name, memory_efficient=True, num_heads=heads
                )

                memory, time, r = run(
                    dtype, seq_len, bs, hidden_size, benchmark=name, memory_efficient=False, num_heads=heads
                )
                print(f"Standard Attention hidden_dim={hidden_size} T={seq_len} dtype={dtype} MiB={memory} ms={time}")

                print(
                    f"Triton Attention hidden_dim={hidden_size} T={seq_len} dtype={dtype} MiB={trit_memory} ms={trit_time}\n"
                )

                print(f"Improvement: MiB={memory / trit_memory} ms={r / trit_r}\n")
