import torch


def as_bfloat16(d: dict[str, torch.Tensor]):
    for k, v in d.items():
        if torch.is_tensor(v) and torch.is_floating_point(v):
            d[k] = v.bfloat16()
    return d
