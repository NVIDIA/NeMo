import torch
import torch.nn as nn
import torch.utils.benchmark as benchmark
from nemo.collections.asr.parts.submodules.multi_head_attention import RelPositionMultiHeadAttention as NewRelPositionMultiHeadAttention
from old_multi_head_attention import RelPositionMultiHeadAttention
from nemo.utils import avoid_float16_autocast_context

torch.manual_seed(123)

device = "cuda"
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
pos_emb = torch.rand(batch_size, seq_len, d_model, device=device, requires_grad=True)

# torch.manual_seed(123)
attention_sdpa = NewRelPositionMultiHeadAttention(n_head, d_model, 0.0, None, None).to(device)
# torch.manual_seed(123)
attention_original = RelPositionMultiHeadAttention(n_head, d_model, 0.0, None, None).to(device)

def measure_time(attention, query, key, value, mask, pos_emb):
    with torch.no_grad():
        timer = benchmark.Timer(
            stmt='attention(query, key, value, mask, pos_emb)',
            setup='torch.cuda.synchronize()',
            globals={'attention': attention, 'query': query, 'key': key, 'value': value, 'mask': mask, 'pos_emb': pos_emb}
        )
        torch.cuda.synchronize()
        results = timer.blocked_autorange(min_run_time=10)
        forward_time = results.mean
        output = attention(query, key, value, mask, pos_emb)
    return forward_time, output

def measure_backward_time(attention, query, key, value, mask, pos_emb):
    timer = benchmark.Timer(
        stmt='loss.backward()',
        setup='''
query.grad = None
key.grad = None
value.grad = None
pos_emb.grad = None
output = attention(query, key, value, mask, pos_emb)
loss = output.sum()
torch.cuda.synchronize()
''',
        globals={'attention': attention, 'query': query, 'key': key, 'value': value, 'mask': mask, 'pos_emb': pos_emb}
    )
    torch.cuda.synchronize()
    results = timer.blocked_autorange(min_run_time=10)
    backward_time = results.mean
    return backward_time


time_original, output_original = measure_time(attention_original, query, key, value, mask, pos_emb)
time_backward_original = measure_backward_time(attention_original, query, key, value, mask, pos_emb)

time_sdpa, output_sdpa = measure_time(attention_sdpa, query, key, value, mask, pos_emb)
time_backward_sdpa = measure_backward_time(attention_sdpa, query, key, value, mask, pos_emb)

print(f"Original implementation time: {time_original:.6f} seconds")
print(f"SDPA implementation time: {time_sdpa:.6f} seconds")
print(f"SDPA boost {(time_original - time_sdpa) / time_original * 100:.3f}%")

print(f"Original implementation backward time: {time_backward_original:.6f} seconds")
print(f"SDPA implementation backward time: {time_backward_sdpa:.6f} seconds")
print(f"SDPA backward boost {(time_backward_original - time_backward_sdpa) / time_backward_original * 100:.3f}%")

print(f"Outputs are {'the same' if torch.allclose(output_original, output_sdpa, atol=1e-5) else 'different'}")
# Original implementation time: 0.042344 seconds
# SDPA implementation time: 0.028285 seconds
# SDPA boost 33.202%
# Original implementation backward time: 0.079229 seconds
# SDPA implementation backward time: 0.082796 seconds
# SDPA backward boost -4.502%
# Outputs are the same