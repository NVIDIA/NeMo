# Copyright (c) 2019 NVIDIA Corporation
import functools
import glob
import os
from collections.abc import Iterable
from pathlib import Path
from typing import Optional

import torch
import wget

import nemo
from nemo.utils import logging

# from nemo.utils import logging


def rgetattr(obj, attr, *args):
    def _getattr(obj, attr):
        return getattr(obj, attr, *args)

    return functools.reduce(_getattr, [obj] + attr.split("."))


def rsetattr(obj, attr, val):
    pre, _, post = attr.rpartition(".")
    return setattr(rgetattr(obj, pre) if pre else obj, post, val)


def get_checkpoint_from_dir(module_names, cpkt_dir, ckpt_pattern='', return_steps=False):
    """ Grab all the modules with match a certain pattern in cpkt_dir
    If multiple checkpoints found, by default, use the one last created.
    """
    if not os.path.isdir(cpkt_dir):
        raise ValueError(f"{cpkt_dir} isn't a directory")

    if not isinstance(module_names, Iterable):
        module_names = [module_names]

    ckpts = []
    steps = []

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
        steps.append(step_from_checkpoint(module_ckpt))

    if return_steps:
        return ckpts, steps
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
#     return nemo.core.NeuralModuleFactory(local_rank=local_rank,
#                                          optimization_level=precision,
#                                          placement=device)


def maybe_download_from_cloud(url, filename, subfolder=None, cache_dir=None, referesh_cache=False) -> str:
    """
    Helper function to download pre-trained weights from the cloud
    Args:
        url: (str) URL of storage
        filename: (str) what to download. The request will be issued to url/filename
        subfolder: (str) subfolder within cache_dir. The file will be stored in cache_dir/subfolder. Subfolder can
            be empty
        cache_dir: (str) a cache directory where to download. If not present, this function will attempt to create it.
            If None (default), then it will be $HOME/.cache/torch/NeMo
        referesh_cache: (bool) if True and cached file is present, it will delete it and re-fetch

    Returns:
        If successful - absolute local path to the downloaded file
        else - empty string
    """
    # try:
    if cache_dir is None:
        cache_location = Path.joinpath(Path.home(), '.cache/torch/NeMo')
    else:
        cache_location = cache_dir
    if subfolder is not None:
        destination = Path.joinpath(cache_location, subfolder)
    else:
        destination = cache_location

    if not os.path.exists(destination):
        os.makedirs(destination, exist_ok=True)

    destination_file = Path.joinpath(destination, filename)

    if os.path.exists(destination_file):
        logging.info(f"Found existing object {destination_file}.")
        if referesh_cache:
            logging.info("Asked to refresh the cache.")
            logging.info(f"Deleting file: {destination_file}")
            os.remove(destination_file)
        else:
            logging.info(f"Re-using file from: {destination_file}")
            return str(destination_file)
    # download file
    wget_uri = url + filename
    logging.info(f"Downloading from: {wget_uri} to {str(destination_file)}")
    wget.download(wget_uri, str(destination_file))
    if os.path.exists(destination_file):
        return destination_file
    else:
        return ""
