import torch.nn as nn

import nemo.lightning as nl
from nemo.common.plan.plan import Plan


class LightningEnv(Plan[nn.Module]):
    def __init__(self, trainer: nl.Trainer | nl.Fabric, can_load: bool = True):
        if isinstance(trainer, nl.Trainer):
            self.trainer = trainer
            self.fabric = trainer.to_fabric()
        else:
            self.fabric = trainer
        self.can_load = can_load

    def execute(self):
        self.fabric.launch()

    def extra_repr(self) -> str:
        _io = self.fabric.__io__

        _accelerator = _io.accelerator
        if not self.can_load:
            _accelerator = "meta"

        out = f"accelerator={_accelerator}, devices={_io.devices}"
        # strategy_cfg = _io.strategy
        # if hasattr(strategy_cfg, "__fn_or_cls__"):
        #     strategy_cfg = strategy_cfg.__fn_or_cls__
        # out += f", strategy={strategy_cfg.__name__}()"

        if _io.precision:
            out += f", precision={_io.precision}"

        return out