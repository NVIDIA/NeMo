# Copyright (c) 2019 NVIDIA Corporation
import functools
import os


def rgetattr(obj, attr, *args):
    def _getattr(obj, attr):
        return getattr(obj, attr, *args)

    return functools.reduce(_getattr, [obj] + attr.split("."))


def rsetattr(obj, attr, val):
    pre, _, post = attr.rpartition(".")
    return setattr(rgetattr(obj, pre) if pre else obj, post, val)


def get_latest_checkpoint_from_dir(module_names, directory, step=None):
    if not isinstance(module_names, list):
        module_names = [module_names]
    most_recent_step = []
    module_checkpoint = []
    for module in module_names:
        if not isinstance(module, str):
            raise ValueError("module {} is not a string".format(module))
        if step is None:
            most_recent_step.append(0)
        module_checkpoint.append(None)

    for file in os.listdir(directory):
        if os.path.isfile(os.path.join(directory, file)):
            for i, module in enumerate(module_names):
                if module in file:
                    step_num = int(file.split("STEP-")[-1].split(".")[0])
                    if step is None and step_num > most_recent_step[i]:
                        most_recent_step[i] = step_num
                        module_checkpoint[i] = os.path.join(directory, file)
                    elif step_num == step:
                        module_checkpoint[i] = os.path.join(directory, file)

    for i, checkpoint in enumerate(module_checkpoint):
        if checkpoint is None:
            if step:
                raise ValueError(
                    "Unable to found checkpoint for {}"
                    " at step {}".format(module_names[i], step)
                )
            raise ValueError(
                "Unable to found checkpoint for {}".format(module_names[i])
            )
    return module_checkpoint


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
