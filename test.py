import torch
import torch.nn as nn
import torch.utils.benchmark as benchmark
import time
from nemo.collections.asr.parts.submodules.multi_head_attention import RelPositionMultiHeadAttention as NewRelPositionMultiHeadAttention
from old_multi_head_attention import RelPositionMultiHeadAttention
from nemo.utils import avoid_float16_autocast_context


torch.manual_seed(123)


device = "cuda"
batch_size = 32
seq_len = 1024
d_model = 512
n_head = 8

query = torch.rand(batch_size, seq_len, d_model).to(device)
key = torch.rand(batch_size, seq_len, d_model).to(device)
value = torch.rand(batch_size, seq_len, d_model).to(device)
mask = torch.ones(batch_size, seq_len, seq_len)
mask = torch.triu(mask).bool().to(device)
# mask = None
pos_emb = torch.rand(batch_size, seq_len, d_model).to(device)

torch.manual_seed(123)
attention_sdpa = NewRelPositionMultiHeadAttention(n_head, d_model, 0.0, None, None).to(device)
torch.manual_seed(123)
attention_original = RelPositionMultiHeadAttention(n_head, d_model, 0.0, None, None).to(device)

output_sdpa = attention_sdpa(query, key, value, mask, pos_emb)
output_original = attention_original(query, key, value, mask, pos_emb)

def measure_time(attention, query, key, value, mask, pos_emb):
    timer = benchmark.Timer(
        stmt='attention(query, key, value, mask, pos_emb)',
        globals={'attention': attention, 'query': query, 'key': key, 'value': value, 'mask': mask, 'pos_emb': pos_emb}
    )
    results = timer.blocked_autorange(min_run_time=10)
    return results.mean, results

time_original, _ = measure_time(attention_original, query, key, value, mask, pos_emb)
time_sdpa, _ = measure_time(attention_sdpa, query, key, value, mask, pos_emb)

print(f"Original implementation time: {time_original:.6f} seconds")
print(f"SDPA implementation time: {time_sdpa:.6f} seconds")
print(f"SDPA boost {(time_original - time_sdpa) / time_original * 100:.3f}%")
print(f"Outputs are {'the same' if torch.allclose(output_original, output_sdpa, atol=1e-5) else 'different'}")
# Original implementation time: 0.042316 seconds
# SDPA implementation time: 0.030923 seconds
# SDPA boost 26.924%
# Outputs are the same