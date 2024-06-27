from functools import wraps
from typing import Any, Callable, Optional, TypeVar

import pytorch_lightning as pl
from pytorch_lightning.callbacks import model_summary
from torch import nn

from nemo.utils import logging

MODEL_TRANSFORM: Optional[Callable[[nn.Module], nn.Module]] = None


class ModelTransform(pl.Callback):
    """
    A PyTorch Lightning callback that applies a model transformation function at the start of fitting or validation.

    This callback is designed to apply a transformation to the model when fitting or validation begins.
    This design allows for loading the original checkpoint first and then applying the transformation,
    which is particularly useful for techniques like Parameter-Efficient Fine-Tuning (PEFT).

    The transformation function is expected to be defined on the LightningModule
    as an attribute called 'model_transform'.

    Key Features:
    - Applies transformation at the start of fit or validation, not during initialization.
    - Allows loading of original checkpoints before transformation.
    - Supports PEFT and similar techniques that modify model structure.

    Example:
        >>> class MyLightningModule(pl.LightningModule):
        ...     def __init__(self):
        ...         super().__init__()
        ...         self.model = SomeModel()
        ...         self.model_transform = lambda m: SomePEFTMethod()(m)
        ...
        >>> model = MyLightningModule()
        >>> # Load original checkpoint here if needed
        >>> model.load_state_dict(torch.load('original_checkpoint.pth'))
        >>> trainer = pl.Trainer(callbacks=[ModelTransform()])
        >>> # The model will be transformed when trainer.fit() or trainer.validate() is called
        >>> trainer.fit(model)

    Note:
        The transformation is applied only once, at the start of fitting or validation,
        whichever comes first. This ensures that the model structure is modified before
        any forward passes or parameter updates occur, but after the original weights
        have been loaded.
    """

    def setup(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", stage: str) -> None:
        """
        Setup the model transformation.

        This method is called by PyTorch Lightning before training begins. It checks if
        the LightningModule has a 'model_transform' attribute and, if so, stores it in
        a global variable for later use.

        Args:
            trainer (pl.Trainer): The PyTorch Lightning trainer instance.
            pl_module (pl.LightningModule): The LightningModule being trained.
            stage (str): The stage of training ('fit', 'validate', 'test', or 'predict').
        """

        global MODEL_TRANSFORM

        # Check if the model has 'model_transform' attribute
        if hasattr(pl_module, 'model_transform') and MODEL_TRANSFORM is None:
            # Store it in the global variable and wrap it in _call_counter
            MODEL_TRANSFORM = _call_counter(pl_module.model_transform)
            # Replace the model's transform function with the wrapped one
            pl_module.model_transform = MODEL_TRANSFORM

    def on_load_checkpoint(self, trainer, pl_module, checkpoint):
        """
        Apply the model transformation before training.
        This method is called by PyTorch Lightning immediately after loading the
        distributed checkpoint from disk into a dictionary, but before the dictionary
        is loaded into the model.

        It calls the _maybe_apply_transform method to apply the transformation if necessary.

        Args:
            trainer (pl.Trainer): The PyTorch Lightning trainer instance.
            pl_module (pl.LightningModule): The LightningModule being trained.
        """
        self._maybe_apply_transform(trainer)

    def _maybe_apply_transform(self, trainer):
        """
        Apply the model transformation if it hasn't been applied yet.

        This method checks if the global MODEL_TRANSFORM function exists and hasn't been
        called yet. If so, it applies the transformation to the trainer's model.

        Args:
            trainer (pl.Trainer): The PyTorch Lightning trainer instance.
        """
        global MODEL_TRANSFORM
        if MODEL_TRANSFORM and MODEL_TRANSFORM.__num_calls__ == 0:
            MODEL_TRANSFORM(trainer.model.pipeline)
        logging.info('After model transform:\n' + str(model_summary.summarize(trainer.model.pipeline)))


T = TypeVar('T', bound=Callable[..., Any])


def _call_counter(func: T) -> T:
    """
    A decorator that counts the number of times a function is called.

    This decorator wraps a function and adds a '__num_calls__' attribute to it,
    which is incremented each time the function is called.

    Args:
        func (Callable): The function to be wrapped.

    Returns:
        Callable: The wrapped function with a call counter.
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        wrapper.__num_calls__ += 1
        return func(*args, **kwargs)

    wrapper.__num_calls__ = 0
    return wrapper  # type: ignore
