
import torch
from torch.nn import functional as F


def interleave_blanks(text, text_len, vocab):
    text = [
        interleave(
            x=torch.empty(len(t) + 1, dtype=torch.long, device=t.device, ).fill_(vocab.blank), y=t,
        )
        for t in text
    ]
    text = merge(text, value=vocab.pad, dtype=torch.long)
    text_len = text_len * 2 + 1
    return text, text_len


def repeat_interleave(text, durs):
    text = merge(
        tensors=[torch.repeat_interleave(text1, durs1) for text1, durs1 in zip(text, durs)], dtype=torch.long,
    )
    text_len = durs.sum(-1)
    return text, text_len


def interleave(x, y):
    """Interleave two tensors."""
    xy = torch.stack([x[:-1], y], dim=1).view(-1)
    xy = F.pad(xy, pad=[0, 1], value=x[-1])
    return xy


def merge(tensors, dim=0, value=0, dtype=None):
    """Merges list of tensors into one."""
    tensors = [tensor if isinstance(tensor, torch.Tensor) else torch.tensor(tensor) for tensor in tensors]
    dim = dim if dim != -1 else len(tensors[0].shape) - 1
    dtype = tensors[0].dtype if dtype is None else dtype
    max_len = max(tensor.shape[dim] for tensor in tensors)
    new_tensors = []
    for tensor in tensors:
        pad = (2 * len(tensor.shape)) * [0]
        pad[-2 * dim - 1] = max_len - tensor.shape[dim]
        new_tensors.append(F.pad(tensor, pad=pad, value=value))
    return torch.stack(new_tensors).to(dtype=dtype)