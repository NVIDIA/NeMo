import torch
import torch.nn as nn
from torch.nn import functional as F


def sandwich_pos_bias(qlen, klen, hidden_size, num_attention_heads, device):
    context_position = torch.arange(qlen, dtype=torch.long, device=device)[:, None]
    memory_position = torch.arange(klen, dtype=torch.long, device=device)[None, :]
    relative_position = memory_position - context_position  # shape (qlen, klen)

    inv_freq = 1.0 / (10000 ** (2 * torch.arange(1, hidden_size / 2, device=device) / hidden_size))

    _bias = torch.sum(relative_position[:, :, None].repeat(1, 1, len(inv_freq)) * inv_freq, axis=2)
    bias = _bias.repeat(num_attention_heads, 1, 1)

    _bias_scales = torch.arange(1, num_attention_heads + 1, 1, device=device)
    bias_scales = torch.stack(
        list(map(lambda x, y: x * y, _bias_scales, torch.ones(num_attention_heads, qlen, klen, device=device)))
    )
    scaled_bias = (bias - hidden_size / 2) / (bias_scales * 8 / num_attention_heads)

    return scaled_bias
