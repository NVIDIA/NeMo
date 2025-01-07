from importlib.metadata import version
from packaging.version import Version as PkgVersion


def get_torch_version_str():
    import torch

    if hasattr(torch, '__version__'):
        return str(torch.__version__)
    else:
        return version("torch")
