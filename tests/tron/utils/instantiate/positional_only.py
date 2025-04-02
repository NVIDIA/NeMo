from typing import Any


class PosOnlyArgsClass:
    def __init__(self, a: Any, b: Any, /, **kwargs: Any) -> None:
        assert isinstance(kwargs, dict)
        self.a = a
        self.b = b
        self.kwargs = kwargs

    def __repr__(self) -> str:
        return f"{self.a=},{self.b},{self.kwargs=}"

    def __eq__(self, other: Any) -> Any:
        if isinstance(other, PosOnlyArgsClass):
            return self.a == other.a and self.b == other.b and self.kwargs == other.kwargs
        else:
            return NotImplemented
