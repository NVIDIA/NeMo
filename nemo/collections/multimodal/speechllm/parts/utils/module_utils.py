import inspect
import numpy as np
import torch


def convert_dataclass_to_dict(dataclass, drop_private=True, drop_protected=False, primitive_only=False):
    """Convert a dataclass to a dictionary."""
    dictionary = {}
    for k, v in inspect.getmembers(dataclass):
        if drop_private and k.startswith('__'):
            continue
        if drop_protected and (k.startswith('_') and not k.startswith('__')):
            continue
        if inspect.isbuiltin(v) or inspect.ismethod(v) or inspect.isfunction(v) or inspect.isgenerator(v):
            continue
        if isinstance(v, np.ndarray):
            v = v.tolist()
        if isinstance(v, torch.Tensor):
            v = v.detach().cpu().numpy().tolist()
        if primitive_only and not isinstance(v, (str, int, float, bool, list, dict)):
            continue
        dictionary[k] = v
    return dictionary
