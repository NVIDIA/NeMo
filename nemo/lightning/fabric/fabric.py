from copy import deepcopy
from pathlib import Path
from typing import Optional, Protocol, Type, TypeVar, Union, runtime_checkable

import fiddle as fdl
import lightning_fabric as lb
from torch import nn
from typing_extensions import Self

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
        """
        Import a model from a given path and set it up for distributed training.

        This method imports a model of the specified type from the given path, loads it,
        and sets it up for distributed training using the current Fabric instance.

        Args:
            path (Union[str, Path]): The path to the model. Can be a local path or a
                Hugging Face model identifier.
            model_type (Type[ModelT]): The type of the model to import. Must be a subclass
                of ConnectorMixin.

        Returns:
            DistributedModel[ModelT]: The imported and distributed model.

        Raises:
            TypeError: If the provided model_type is not a subclass of ConnectorMixin.

        Example:
            >>> from nemo import lightning as nl
            >>> from nemo.collections.llm import MistralModel
            >>> 
            >>> trainer = nl.Trainer(
            ...     devices=2,
            ...     strategy=nl.MegatronStrategy(tensor_model_parallel_size=2),
            ...     plugins=nl.MegatronMixedPrecision(precision='16-mixed')
            ... )
            >>> fabric = trainer.to_fabric()
            >>> model = fabric.import_model("hf://mistralai/Mistral-7B-v0.1", MistralModel)
            >>> 
            >>> # The model is now ready for distributed training
            >>> # You can use it with your data and training loop
        """
        from nemo.lightning.io import ConnectorMixin

        if not issubclass(model_type, ConnectorMixin):
            raise TypeError("The provided model class must be a subclass of ConnectorMixin")

        model: ModelT = model_type.import_from(path)

        return self.load_model(model.ckpt_path, model)


@runtime_checkable
class DistributedModel(Protocol[ModelT]):
    module: ModelT
