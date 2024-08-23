from typing import Callable, Dict, Optional

import torch
from prettytable import PrettyTable
from pytorch_lightning.callbacks import Callback

from nemo.lightning.pytorch.optim.megatron import MegatronOptimizerModule
from nemo.utils import logging


def collect_precision(tensor: torch.Tensor):
    return {"Precision": str(tensor.dtype)}


def collect_precision_and_shape(tensor: torch.Tensor):
    return {"Shape": str(tensor.shape), "Precision": str(tensor.dtype)}


class ParameterDebugger(Callback):
    def __init__(
        self,
        param_attr_fn: Optional[Callable[[torch.Tensor], Dict[str, str]]] = collect_precision_and_shape,
        grad_attr_fn: Optional[Callable[[torch.Tensor], Dict[str, str]]] = collect_precision,
    ):
        self.param_attr_fn = param_attr_fn
        self.grad_attr_fn = grad_attr_fn

    def on_train_start(self, trainer, pl_module):

        def find_grad_tensor(param):
            """If using MCore optimizer, search the grad buckets for param's grad tensor."""
            if not isinstance(pl_module.optim, MegatronOptimizerModule):
                return param.grad

            for buf in pl_module.buffers:
                if param in buf.param_to_bucket:
                    return buf.param_to_bucket[param].grad_data

        debug_table = PrettyTable()
        debug_table.align = "l"

        param_keys = self.param_attr_fn(torch.zeros(0)).keys() if self.param_attr_fn else []
        grad_keys = self.grad_attr_fn(torch.zeros(0)).keys() if self.grad_attr_fn else []
        debug_table.field_names = ["Parameter"] + ["Param " + k for k in param_keys] + ["Grad " + k for k in grad_keys]

        for param_name, param_tensor in pl_module.named_parameters():
            short_name = param_name.replace("module.", "").replace(".weight", "")
            grad_tensor = find_grad_tensor(param_tensor)

            row = [short_name]
            for tensor, attr_fn, attr_keys in zip(
                [param_tensor, grad_tensor], [self.param_attr_fn, self.grad_attr_fn], [param_keys, grad_keys]
            ):
                if attr_fn is not None:
                    if tensor is not None:
                        # iterate instead of just appending attrs.values() to ensure order
                        attrs = attr_fn(tensor)
                        row += [attrs[k] for k in attr_keys]
                    else:
                        row += [None for k in attr_keys]

            debug_table.add_row(row)

        logging.info("\n" + debug_table.get_string())
