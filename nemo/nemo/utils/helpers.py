# Copyright (c) 2019 NVIDIA Corporation
from collections.abc import Iterable
import functools
import glob
import os


def rgetattr(obj, attr, *args):
    def _getattr(obj, attr):
        return getattr(obj, attr, *args)

    return functools.reduce(_getattr, [obj] + attr.split("."))


def rsetattr(obj, attr, val):
    pre, _, post = attr.rpartition(".")
    return setattr(rgetattr(obj, pre) if pre else obj, post, val)


def get_checkpoint_from_dir(module_names, cpkt_dir, ckpt_pattern=''):
    if not isinstance(module_names, Iterable):
        module_names = [module_names]

    ckpts = []

    for module in module_names:
        if not isinstance(module, str):
            raise ValueError(f"Module {module} is not a string")

        module_ckpts = glob.glob(f'{cpkt_dir}/{module}*{ckpt_pattern}*')
        if not module_ckpts:
            raise ValueError(f'No file matches {ckpt_pattern} in {cpkt_dir}')

        # if multiple checkpoints match a pattern, take the latest one
        module_ckpts = sorted(module_ckpts, key=os.path.getmtime)
        ckpts.append(module_ckpts[-1])

    return ckpts


def _call_args_to_string(call_args):
    call_dict = {inport: value.name for inport, value in call_args.items()}
    result = "(force_pt=True,"
    counter = 0
    for key, value in call_dict.items():
        result += (
            "{0}={1}".format(key, value)
            if counter == 0
            else ", {0}={1}".format(key, value)
        )
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
