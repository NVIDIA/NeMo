import math

import torch
import xformers.ops as xops
from xformers.ops import fmha

batch_size, num_heads, seq_len = 16, 16, 128
D = 64
device = torch.device('cuda')

q = torch.randn(batch_size, seq_len, num_heads, D, device=device, dtype=torch.bfloat16)
k = torch.randn(batch_size, seq_len, num_heads, D, device=device, dtype=torch.bfloat16)
v = torch.randn(batch_size, seq_len, num_heads, D, device=device, dtype=torch.bfloat16)


def get_slopes(n):
    def get_slopes_power_of_2(n):
        start = 2 ** (-(2 ** -(math.log2(n) - 3)))
        ratio = start
        return [start * ratio ** i for i in range(n)]

    if math.log2(n).is_integer():
        return get_slopes_power_of_2(
            n
        )  # In the paper, we only train models that have 2^a heads for some a. This function has
    else:  # some good properties that only occur when the input is a power of 2. To maintain that even
        closest_power_of_2 = 2 ** math.floor(
            math.log2(n)
        )  # when the number of heads is not a power of 2, we use this workaround.
        return (
            get_slopes_power_of_2(closest_power_of_2)
            + get_slopes(2 * closest_power_of_2)[0::2][: n - closest_power_of_2]
        )


def alibi_attn_bias():
    context_position = torch.arange(seq_len)[:, None]
    memory_position = torch.arange(seq_len)[None, :]
    relative_position = memory_position - context_position
    relative_position = torch.abs(relative_position).unsqueeze(0).expand(num_heads, -1, -1)

    slopes = torch.Tensor(get_slopes(num_heads)) * -1
    alibi = slopes.unsqueeze(1).unsqueeze(1) * relative_position
    return alibi.view(1, num_heads, seq_len, seq_len).expand(batch_size, -1, -1, -1).type_as(q)


attn_bias = alibi_attn_bias()
# attn_bias = fmha.attn_bias.LowerTriangularMaskWithTensorBias(attn_bias)

print(attn_bias)

attn = xops.memory_efficient_attention(q, k, v, attn_bias=attn_bias.view(batch_size * num_heads, seq_len, seq_len))
print(attn)
