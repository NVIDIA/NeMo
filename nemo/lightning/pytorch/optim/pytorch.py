# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Callable, List, Optional

import pytorch_lightning as pl
import pytorch_lightning as L
from torch.optim import Optimizer

from nemo.lightning.megatron_parallel import MegatronParallel
from nemo.lightning.pytorch.optim.base import LRSchedulerModule, OptimizerModule


def _param_does_not_have_wd(param_name, param):
    return 'bias' in param_name


class PytorchOptimizerModule(OptimizerModule):
    """A OptimizerModule for pytorch optimizers.

    Attributes:
        optimizer (Partial(optim_cls, lr=...)): Configuration for the optimizer.
        no_weight_decay_cond (Optional[Callable]): Condition for no weight decay.
        scale_lr_cond (Optional[Callable]): Condition for scaling learning rate.
        lr_mult (float): Learning rate multiplier.

    Example::

        config = OptimizerConfig(...)
        lr_scheduler = MyLRSchedulerModule(...)
        optimizer_module = PytorchOptimizerModule(config, lr_scheduler)

    Methods:
        setup(model): Sets up the optimizer.
        optimizers(model): Defines the optimizers.
    """

    def __init__(
        self,
        optimizer_fn,
        lr_scheduler: Optional[LRSchedulerModule] = None,
        no_weight_decay_cond: Optional[Callable] = _param_does_not_have_wd,
        scale_lr_cond: Optional[Callable] = None,
        lr_mult: float = 1.0,
    ):
        """Initializes the PytorchOptimizerModule.

        Args:
            config (OptimizerConfig): Configuration for the optimizer.
            lr_scheduler (Optional[LRSchedulerModule]): The learning rate scheduler module.
            no_weight_decay_cond (Optional[Callable]): Condition for no weight decay.
            scale_lr_cond (Optional[Callable]): Condition for scaling learning rate.
            lr_mult (float): Learning rate multiplier.
        """

        super().__init__(lr_scheduler=lr_scheduler)
        self.optimizer_fn = optimizer_fn
        self.config = config
        self.no_weight_decay_cond = no_weight_decay_cond
        self.scale_lr_cond = scale_lr_cond
        self.lr_mult = lr_mult
        self.optim_cls = optim_cls

    def on_fit_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"):
        # Noop
        pass

    def optimizers(self, model) -> List[Optimizer]:
        """Defines the optimizers.

        Args:
            model (nn.Module): The model for which the optimizers are being defined.

        Returns:
            List[Optimizer]: The list of optimizers.

        Raises:
            ValueError: If the model is an instance of MegatronParallel.
        """

        if isinstance(model, MegatronParallel):
            raise ValueError("Model cannot be an instance of MegatronParallel")

        params_with_wd, params_without_wd = [], []
        if self.no_weight_decay_cond is not None:
            for name, param in model.named_parameters():
                if self.no_weight_decay_cond(name, param):
                    params_without_wd.append(param)
                else:
                    params_with_wd.append(param)
        else:
            params_with_wd = model.parameters()

        assert max(map(len, (params_with_wd, params_without_wd))) > 0, "Expected at least one optimizer with params"

        return self.optimizer_fn(
            [
                {'params': params, 'weight_decay': weight_decay}
                for params, weight_decay in zip(
                    (params_with_wd, params_without_wd), (self.config.get('weight_decay', 0), 0)
                )
            ]
        )

    def finalize_model_grads(self, *args, **kwargs):
        # Noop
        pass

    def connect(self, model: L.LightningModule) -> None:
        """Connects the optimizer module to the model and trainer.

        Args:
            model (L.LightningModule): The model to which the optimizer module is being connected.
        """
        model.configure_optimizers = lambda: self.optimizers(model)
