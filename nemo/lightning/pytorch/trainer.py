from copy import deepcopy

import fiddle as fdl
import lightning as L
from typing_extensions import Self

from nemo.io.mixin import IOMixin


class Trainer(L.Trainer, IOMixin):
    def io_init(self, **kwargs) -> fdl.Config[Self]:
        # Each argument of the trainer can be stateful so we copy them
        cfg_kwargs = {k: deepcopy(v) for k, v in kwargs.items()}

        return fdl.Config(type(self), **cfg_kwargs)
