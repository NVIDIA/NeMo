from typing import Callable, TypeVar

from torch import nn
from nemo.lightning import io
from nemo.common.plan.plan import Plan

ModelT = TypeVar("ModelT", bound=nn.Module)


class StateConverter(Plan[nn.Module]):
    def __init__(self, mapping: dict[str, str], transforms: list[Callable] = None):
        super().__init__()
        self.mapping = mapping
        self.transforms = transforms

    def execute(self, source: nn.Module, target: ModelT) -> ModelT:
        return io.apply_transforms(
            source,
            target,
            mapping=self.mapping,
            transforms=self.transforms,
        )

    def extra_repr(self) -> str:
        """Return a string representation of the StateConverter's parameters."""
        lines = []

        # Format the mapping dictionary
        if self.mapping:
            lines.append("mapping={")
            for key, value in self.mapping.items():
                lines.append(f"  {repr(key)}: {repr(value)},")
            lines.append("}")
        else:
            lines.append("mapping={}")

        # Format the transforms list with full import paths
        if self.transforms:
            transform_paths = []
            for fn in self.transforms:
                # Get the full import path of the function
                if hasattr(fn, "__module__") and hasattr(fn, "__qualname__"):
                    transform_paths.append(f"{fn.__module__}.{fn.__qualname__}")
                else:
                    transform_paths.append(repr(fn))

            lines.append("transforms=[")
            for path in transform_paths:
                lines.append(f"  {path},")
            lines.append("]")

        return "\n".join(lines)
