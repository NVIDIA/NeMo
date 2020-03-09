# Copyright (c) 2019 NVIDIA Corporation
import functools
import glob
import os
import tarfile
from collections.abc import Iterable
from pathlib import Path
from typing import Optional

import torch
import wget

import nemo
from nemo.utils import logging

# logging = nemo.logging


def rgetattr(obj, attr, *args):
    def _getattr(obj, attr):
        return getattr(obj, attr, *args)

    return functools.reduce(_getattr, [obj] + attr.split("."))


def rsetattr(obj, attr, val):
    pre, _, post = attr.rpartition(".")
    return setattr(rgetattr(obj, pre) if pre else obj, post, val)


def get_checkpoint_from_dir(module_names, cpkt_dir, ckpt_pattern=''):
    """ Grab all the modules with match a certain pattern in cpkt_dir
    If multiple checkpoints found, by default, use the one last created.
    """
    if not os.path.isdir(cpkt_dir):
        raise ValueError(f"{cpkt_dir} isn't a directory")

    if not isinstance(module_names, Iterable):
        module_names = [module_names]

    ckpts = []

    for module in module_names:
        if not isinstance(module, str):
            raise ValueError(f"Module {module} is not a string")
        if not isinstance(ckpt_pattern, str):
            raise ValueError(f"Pattern {ckpt_pattern} is not a string")

        module_ckpts = glob.glob(f'{cpkt_dir}/{module}*{ckpt_pattern}*')
        if not module_ckpts:
            raise ValueError(f'For module {module}, ' f'no file matches {ckpt_pattern} in {cpkt_dir}')

        # if multiple checkpoints match a pattern, take the latest one
        def step_from_checkpoint(checkpoint_name):
            # return step number given a checkpoint filename
            step_str = checkpoint_name.split('-')[-1].split('.')[0]
            return int(step_str)

        module_ckpt = module_ckpts[0]
        if len(module_ckpts) > 1:
            module_ckpt = max(module_ckpts, key=step_from_checkpoint)
        ckpts.append(module_ckpt)

    return ckpts


def _call_args_to_string(call_args):
    call_dict = {inport: value.name for inport, value in call_args.items()}
    result = "(force_pt=True,"
    counter = 0
    for key, value in call_dict.items():
        result += f"{key}={value}" if counter == 0 else f", {key}={value}"
        counter += 1
    result += ")"
    return result


def _get_instance_call_line(output_ports, instance_ref, call_str):
    result = ""
    counter = 0
    for out_port in output_ports:
        result += out_port if counter == 0 else (", " + out_port)
        counter += 1
    result += " = " + instance_ref + call_str
    return result


def get_device(local_rank: Optional[int]):
    if local_rank is not None:
        return nemo.core.DeviceType.AllGpu
    return nemo.core.DeviceType.GPU


def get_cuda_device(placement):
    """
    Converts NeMo nemo.core.DeviceType to torch.device
    Args:
        placement: nemo.core.DeviceType

    Returns:
        torch.device
    """
    gpu_devices = [nemo.core.DeviceType.GPU, nemo.core.DeviceType.AllGpu]
    return torch.device("cuda" if placement in gpu_devices else "cpu")


# def get_neural_factory(local_rank,
#                        precision,
#                        backend):
#     """
#     Helper function to create NeuralModuleFactory
#     Args:
#         local_rank:
#         precision: (nemo.core.Optimization) AMP mixed precision level
#         backend: (nemo.core.Backend) NeMo backend (defaults to Pytorch)

#     Returns:
#         An instance of the NeuralModuleFactory
#     """
#     device = nemo.utils.get_device(local_rank)
#     return nemo.core.NeuralModuleFactory(backend=backend,
#                                          local_rank=local_rank,
#                                          optimization_level=precision,
#                                          placement=device)


def maybe_download_from_cloud(url, filename) -> str:
    """
    Helper function to download pre-trained weights from the cloud
    Args:
        url: (str) URL of storage
        filename: (str) what to download. The request will be issued
        to url/filename or url/filename.tar.gz

    Returns:
        If successful - absolute local path to the directory where
        checkpoints are
        else - empty string
    """
    try:
        nfname = ".nemo_files"
        # check if ~/.nemo_files exists, if not - create
        home_folder = Path.home()
        nf_absname = os.path.join(home_folder, nfname)
        if not os.path.exists(nf_absname):
            os.mkdir(nf_absname)
        # check if thing is already downloaded and unpacked
        if filename.endswith('.tar.gz'):
            name = filename[:-7]
        else:
            name = filename
        destination = os.path.join(nf_absname, name)
        if os.path.exists(destination):
            return str(destination)
        # download file
        wget.download(url + name + ".tar.gz", str(nf_absname))
        tf = tarfile.open(os.path.join(nf_absname, name + ".tar.gz"))
        tf.extractall(nf_absname)
        if os.path.exists(destination):
            return destination
        else:
            return ""
    except (FileNotFoundError, ConnectionError, OSError):
        logging.info(f"Could not obtain {filename} from the cloud")
        return ""
