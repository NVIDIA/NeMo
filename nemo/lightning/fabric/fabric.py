from typing import Union, Type, TypeVar, Protocol, Optional, runtime_checkable
from pathlib import Path
from typing_extensions import Self
from copy import deepcopy

from torch import nn
import lightning_fabric as lb
import fiddle as fdl

from nemo.lightning.io.mixin import IOMixin


ModelT = TypeVar("ModelT", bound=nn.Module)


class Fabric(lb.Fabric, IOMixin):
    def io_init(self, **kwargs) -> fdl.Config[Self]:
        # Each argument of the trainer can be stateful so we copy them
        cfg_kwargs = {k: deepcopy(v) for k, v in kwargs.items()}

        return fdl.Config(type(self), **cfg_kwargs)
    
    def load_model(
        self,
        path: Union[str, Path],
        model: Optional[ModelT] = None,
    ) -> "DistributedModel[ModelT]":
        self.launch()
        
        from nemo.lightning.io import load_context
        
        if model is None:
            context = load_context(path)
            model = context.model

        dist_model = self.setup_module(model)
        self.load(path, {"state_dict": dist_model})

        return dist_model

    def import_model(
        self,
        path: Union[str, Path],
        model_type: Type[ModelT],
    ) -> "DistributedModel[ModelT]":
        from nemo.lightning.io import ConnectorMixin
        
        if not issubclass(model_type, ConnectorMixin):
            raise TypeError("The provided model class must be a subclass of ConnectorMixin")
        
        model: ModelT = model_type.import_from(path)
        
        return self.load_model(model.ckpt_path, model)
        
        
@runtime_checkable
class DistributedModel(Protocol[ModelT]):
    module: ModelT
