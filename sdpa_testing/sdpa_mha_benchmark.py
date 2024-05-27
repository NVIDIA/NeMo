import torch
import torch.nn as nn
import torch.utils.benchmark as benchmark
from nemo.collections.asr.parts.submodules.multi_head_attention import MultiHeadAttention as SDPAMultiHeadAttention
from old_multi_head_attention import MultiHeadAttention
from nemo.utils import avoid_float16_autocast_context

torch.manual_seed(123)

device = "cpu"
batch_size = 32
seq_len = 1024
d_model = 512
n_head = 8

query = torch.rand(batch_size, seq_len, d_model, device=device, requires_grad=True)
key = torch.rand(batch_size, seq_len, d_model, device=device, requires_grad=True)
value = torch.rand(batch_size, seq_len, d_model, device=device, requires_grad=True)
mask = torch.ones(batch_size, seq_len, seq_len, device=device, requires_grad=False)
mask = torch.triu(mask, diagonal=1).bool()
# mask = None

attention_sdpa = SDPAMultiHeadAttention(n_head, d_model, 0.0).to(device)
attention_original = MultiHeadAttention(n_head, d_model, 0.0).to(device)
for original_param, sdpa_param in zip(attention_original.parameters(), attention_sdpa.parameters()):
    original_param.data.copy_(sdpa_param.data)
# attention_sdpa = torch.compile(attention_sdpa)
# attention_original = torch.compile(attention_original)

def measure_time(attention, query, key, value, mask):
    with torch.no_grad():
        timer = benchmark.Timer(
            stmt='attention(query, key, value, mask)',
            setup='torch.cuda.synchronize()',
            globals={'attention': attention, 'query': query, 'key': key, 'value': value, 'mask': mask}
        )
        torch.cuda.synchronize()
        
        with torch.no_grad(), torch.backends.cuda.sdp_kernel(enable_math=False, enable_mem_efficient = False, enable_flash = True):
            results = timer.blocked_autorange(min_run_time=10)
            forward_time = results.mean
            output = attention(query, key, value, mask)
    return forward_time, output

def measure_backward_time(attention, query, key, value, mask):
    timer = benchmark.Timer(
        stmt='loss.backward()',
        setup='''
torch.cuda.empty_cache()
output = attention(query, key, value, mask)
loss = output.sum()
torch.cuda.synchronize()
''',
        globals={'attention': attention, 'query': query, 'key': key, 'value': value, 'mask': mask}
    )
    torch.cuda.synchronize()
    results = timer.blocked_autorange(min_run_time=10)
    backward_time = results.mean
    return backward_time


# time_original, output_original = measure_time(attention_original, query, key, value, mask)
# time_sdpa, output_sdpa = measure_time(attention_sdpa, query, key, value, mask)

# print(f"Original implementation time: {time_original:.6f} seconds")
# print(f"SDPA implementation time: {time_sdpa:.6f} seconds")
# print(f"SDPA boost {(time_original - time_sdpa) / time_original * 100:.3f}%")

time_backward_original = measure_backward_time(attention_original, query, key, value, mask)
time_backward_sdpa = measure_backward_time(attention_sdpa, query, key, value, mask)

print(f"Original implementation backward time: {time_backward_original:.6f} seconds")
print(f"SDPA implementation backward time: {time_backward_sdpa:.6f} seconds")
print(f"SDPA backward boost {(time_backward_original - time_backward_sdpa) / time_backward_original * 100:.3f}%")

# print(f"Outputs are {'the same' if torch.allclose(output_original, output_sdpa, atol=1e-5) else 'different'}")
