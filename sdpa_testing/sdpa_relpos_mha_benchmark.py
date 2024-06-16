import torch
import torch.nn as nn
import torch.utils.benchmark as benchmark
from nemo.collections.asr.parts.submodules.multi_head_attention import RelPositionMultiHeadAttention as SDPARelPositionMultiHeadAttention
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
mask = None
pos_emb = torch.rand(batch_size, seq_len, d_model, device=device, requires_grad=True)

attention_sdpa = SDPARelPositionMultiHeadAttention(n_head, d_model, 0.0, None, None).to(device)
attention_original = RelPositionMultiHeadAttention(n_head, d_model, 0.0, None, None).to(device)
for original_param, sdpa_param in zip(attention_original.parameters(), attention_sdpa.parameters()):
    original_param.data.copy_(sdpa_param.data)

# attention_sdpa = torch.compile(attention_sdpa)
# attention_original = torch.compile(attention_original)


def measure_time(attention, query, key, value, mask, pos_emb):
    with torch.no_grad():
        timer = benchmark.Timer(
            stmt='attention(query, key, value, mask, pos_emb);torch.cuda.synchronize()',
            setup='torch.cuda.synchronize()',
            globals={'attention': attention, 'query': query, 'key': key, 'value': value, 'mask': mask, 'pos_emb': pos_emb}
        )
        
        with torch.no_grad():
            results = timer.blocked_autorange(min_run_time=10)
            forward_time = results.mean
            output = attention(query, key, value, mask, pos_emb)
    return forward_time, output


def measure_fwd_bwd_time(attention, query, key, value, mask, pos_emb):
    timer = benchmark.Timer(
        stmt='loss=attention(query, key, value, mask, pos_emb).sum();torch.cuda.synchronize();loss.backward();torch.cuda.synchronize()',
        setup='torch.cuda.synchronize()',
            globals={'attention': attention, 'query': query, 'key': key, 'value': value, 'mask': mask, 'pos_emb': pos_emb}
    )
    torch.cuda.synchronize()
    results = timer.blocked_autorange(min_run_time=10)
    fwd_bwd_time = results.mean
    return fwd_bwd_time


time_fwd_original, output_original = measure_time(attention_original, query, key, value, mask, pos_emb)
time_fwd_sdpa, output_sdpa = measure_time(attention_sdpa, query, key, value, mask, pos_emb)

print(f"Original implementation time: {time_fwd_original:.6f} seconds")
print(f"SDPA implementation time: {time_fwd_sdpa:.6f} seconds")
print(f"SDPA boost {(time_fwd_original - time_fwd_sdpa) / time_fwd_original * 100:.2f}%")

time_fwd_bwd_original = measure_fwd_bwd_time(attention_original, query, key, value, mask, pos_emb)
time_fwd_bwd_sdpa = measure_fwd_bwd_time(attention_sdpa, query, key, value, mask, pos_emb)
time_bwd_original = time_fwd_bwd_original - time_fwd_original
time_bwd_sdpa = time_fwd_bwd_sdpa - time_fwd_sdpa

print(f"Original implementation backward time: {time_bwd_original:.6f} seconds")
print(f"SDPA implementation backward time: {time_bwd_sdpa:.6f} seconds")
print(f"SDPA backward boost {(time_bwd_original - time_bwd_sdpa) / time_bwd_original * 100:.2f}%")

print(f"Outputs are {'the same' if torch.allclose(output_original, output_sdpa, atol=1e-5) else 'different'}")

# Original implementation time: 0.031546 seconds
# SDPA implementation time: 0.025657 seconds
# SDPA boost 18.67%
# Original implementation backward time: 0.069962 seconds
# SDPA implementation backward time: 0.078320 seconds
# SDPA backward boost -11.95%
# Outputs are the same