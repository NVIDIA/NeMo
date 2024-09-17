from typing import Callable, Dict, List, Optional, Union

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
        log_on_hooks: Union[List[str], str] = "on_train_start",
    ):
        self.param_attr_fn = param_attr_fn
        self.grad_attr_fn = grad_attr_fn

        valid_hooks = set(
            [
                "teardown",
                "on_fit_end",
                "on_sanity_check_start",
                "on_sanity_check_end",
                "on_train_batch_start",
                "on_train_batch_end",
                "on_train_epoch_start",
                "on_train_epoch_end",
                "on_validation_epoch_start",
                "on_validation_epoch_end",
                "on_test_epoch_start",
                "on_test_epoch_end",
                "on_predict_epoch_start",
                "on_predict_epoch_end",
                "on_validation_batch_start",
                "on_validation_batch_end",
                "on_test_batch_start",
                "on_test_batch_end",
                "on_predict_batch_start",
                "on_predict_batch_end",
                "on_train_start",
                "on_train_end",
                "on_validation_start",
                "on_validation_end",
                "on_test_start",
                "on_test_end",
                "on_predict_start",
                "on_predict_end",
                "on_exception",
                "on_save_checkpoint",
                "on_load_checkpoint",
                "on_before_backward",
                "on_after_backward",
                "on_before_optimizer_step",
                "on_before_zero_grad",
            ]
        )

        if isinstance(log_on_hooks, str):
            log_on_hooks = [log_on_hooks]
        for hook_name in log_on_hooks:
            assert (
                hook_name in valid_hooks
            ), f"Hook {hook_name} supplied to log_on_hooks is not valid or can not be used. Valid hooks are {valid_hooks}"
            setattr(self, hook_name, self._log_param_and_grad_attrs)

    def _log_param_and_grad_attrs(self, trainer, pl_module):

        def find_grad_tensor(param):
            """If using MCore optimizer, search the grad buckets for param's grad tensor."""
            if not isinstance(pl_module.optim, MegatronOptimizerModule):
                return param.grad

            for buf in pl_module.buffers:
                if param in buf.param_to_bucket:
                    return buf.param_to_bucket[param].grad_data

        # create table and get table column headers
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
