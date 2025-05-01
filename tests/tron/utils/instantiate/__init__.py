# Based on https://github.com/facebookresearch/hydra/tree/main/tests/instantiate

import collections
import collections.abc
from dataclasses import dataclass, field
from functools import partial
from typing import Any, Dict, List, NoReturn, Optional, Tuple

from omegaconf import MISSING, DictConfig, ListConfig

from nemo.tron.utils.instantiate_utils import instantiate
from tests.tron.utils.instantiate.module_shadowed_by_function import a_function

module_shadowed_by_function = a_function


def _convert_type(obj: Any) -> Any:
    if isinstance(obj, DictConfig):
        obj = dict(obj)
    elif isinstance(obj, ListConfig):
        obj = list(obj)
    return obj


def partial_equal(obj1: Any, obj2: Any) -> bool:
    if obj1 == obj2:
        return True

    obj1, obj2 = _convert_type(obj1), _convert_type(obj2)

    if type(obj1) is not type(obj2):
        return False
    if isinstance(obj1, dict):
        if len(obj1) != len(obj2):
            return False
        for i in obj1.keys():
            if not partial_equal(obj1[i], obj2[i]):
                return False
        return True
    if isinstance(obj1, list):
        if len(obj1) != len(obj2):
            return False
        return all(partial_equal(o1, o2) for o1, o2 in zip(obj1, obj2))
    if not (isinstance(obj1, partial) and isinstance(obj2, partial)):
        return False
    return all([partial_equal(getattr(obj1, attr), getattr(obj2, attr)) for attr in ["func", "args", "keywords"]])


class ArgsClass:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        assert isinstance(args, tuple)
        assert isinstance(kwargs, dict)
        self.args = args
        self.kwargs = kwargs

    def __repr__(self) -> str:
        return f"self.args={self.args},self.kwarg={self.kwargs}"

    def __eq__(self, other: Any) -> Any:
        if isinstance(other, ArgsClass):
            return self.args == other.args and self.kwargs == other.kwargs
        else:
            return NotImplemented


class OuterClass:
    def __init__(self) -> None:
        pass

    @staticmethod
    def method() -> str:
        return "OuterClass.method return"

    class Nested:
        def __init__(self) -> None:
            pass

        @staticmethod
        def method() -> str:
            return "OuterClass.Nested.method return"


def add_values(a: int, b: int) -> int:
    return a + b


def module_function(x: int) -> int:
    return x


def module_function2() -> str:
    return "fn return"


class ExceptionTakingNoArgument(Exception):
    def __init__(self) -> None:
        """Init method taking only one argument (self)"""
        super().__init__("Err message")


def raise_exception_taking_no_argument() -> NoReturn:
    raise ExceptionTakingNoArgument()


@dataclass
class AClass:
    a: Any
    b: Any
    c: Any
    d: Any = "default_value"

    @staticmethod
    def static_method(z: int) -> int:
        return z


@dataclass
class BClass:
    a: Any
    b: Any
    c: Any = "c"
    d: Any = "d"


@dataclass
class KeywordsInParamsClass:
    target: Any
    partial: Any


@dataclass
class UntypedPassthroughConf:
    _target_: str = "tests.tron.utils.instantiate.UntypedPassthroughClass"
    a: Any = MISSING


@dataclass
class UntypedPassthroughClass:
    a: Any


# Type not legal in a config
class IllegalType:
    def __eq__(self, other: Any) -> Any:
        return isinstance(other, IllegalType)


@dataclass
class AnotherClass:
    x: int


class ASubclass(AnotherClass):
    @classmethod
    def class_method(cls, y: int) -> Any:
        return cls(y + 1)

    @staticmethod
    def static_method(z: int) -> int:
        return z


class Parameters:
    def __init__(self, params: List[float]):
        self.params = params

    def __eq__(self, other: Any) -> Any:
        if isinstance(other, Parameters):
            return self.params == other.params
        return False

    def __deepcopy__(self, memodict: Any = {}) -> Any:
        raise NotImplementedError("Pytorch parameters does not support deepcopy")


@dataclass
class Adam:
    params: Parameters
    lr: float = 0.001
    betas: Tuple[float, ...] = (0.9, 0.999)
    eps: float = 1e-08
    weight_decay: int = 0
    amsgrad: bool = False


@dataclass
class NestingClass:
    a: ASubclass = field(default_factory=lambda: ASubclass(10))


nesting = NestingClass()


class ClassWithMissingModule:
    def __init__(self) -> None:
        import some_missing_module  # type: ignore # noqa: F401

        self.x = 1


@dataclass
class AdamConf:
    _target_: str = "tests.tron.utils.instantiate.Adam"
    lr: float = 0.001
    betas: Tuple[float, ...] = (0.9, 0.999)
    eps: float = 1e-08
    weight_decay: int = 0
    amsgrad: bool = False


@dataclass
class User:
    name: str = MISSING
    age: int = MISSING


@dataclass
class UserGroup:
    name: str = MISSING
    users: List[User] = MISSING


# RECURSIVE
# Classes
class Transform: ...


class CenterCrop(Transform):
    def __init__(self, size: int):
        self.size = size

    def __eq__(self, other: Any) -> Any:
        if isinstance(other, type(self)):
            return self.size == other.size
        else:
            return False


class Rotation(Transform):
    def __init__(self, degrees: int):
        self.degrees = degrees

    def __eq__(self, other: Any) -> Any:
        if isinstance(other, type(self)):
            return self.degrees == other.degrees
        else:
            return False


class Compose:
    transforms: List[Transform]

    def __init__(self, transforms: List[Transform]):
        self.transforms = transforms

    def __eq__(self, other: Any) -> Any:
        return partial_equal(self.transforms, other.transforms)


class Tree:
    value: Any
    # annotated any because of non recursive instantiation tests
    left: Any = None
    right: Any = None

    def __init__(self, value: Any, left: Any = None, right: Any = None) -> None:
        self.value = value
        self.left = left
        self.right = right

    def __eq__(self, other: Any) -> Any:
        if isinstance(other, type(self)):
            return (
                partial_equal(self.value, other.value)
                and partial_equal(self.left, other.left)
                and partial_equal(self.right, other.right)
            )

        else:
            return False

    def __repr__(self) -> str:
        return f"Tree(value={self.value}, left={self.left}, right={self.right})"


class Mapping:
    dictionary: Optional[Dict[str, "Mapping"]] = None
    value: Any = None

    def __init__(self, value: Any = None, dictionary: Optional[Dict[str, "Mapping"]] = None) -> None:
        self.dictionary = dictionary
        self.value = value

    def __eq__(self, other: Any) -> Any:
        if isinstance(other, type(self)):
            return partial_equal(self.dictionary, other.dictionary) and partial_equal(self.value, other.value)
        else:
            return False

    def __repr__(self) -> str:
        return f"dictionary={self.dictionary}"


# Configs
@dataclass
class TransformConf: ...


@dataclass
class CenterCropConf(TransformConf):
    _target_: str = "tests.tron.utils.instantiate.CenterCrop"
    _partial_: bool = False
    size: int = MISSING


@dataclass
class RotationConf(TransformConf):
    _target_: str = "tests.tron.utils.instantiate.Rotation"
    degrees: int = MISSING


@dataclass
class ComposeConf:
    _target_: str = "tests.tron.utils.instantiate.Compose"
    _partial_: bool = False
    transforms: List[TransformConf] = MISSING


@dataclass
class TreeConf:
    _target_: str = "tests.tron.utils.instantiate.Tree"
    _partial_: bool = False
    left: Optional["TreeConf"] = None
    right: Optional["TreeConf"] = None
    value: Any = MISSING


@dataclass
class MappingConf:
    _target_: str = "tests.tron.utils.instantiate.Mapping"
    _partial_: bool = False
    dictionary: Optional[Dict[str, "MappingConf"]] = None

    def __init__(
        self,
        dictionary: Optional[Dict[str, "MappingConf"]] = None,
        _partial_: bool = False,
    ):
        self.dictionary = dictionary
        self._partial_ = _partial_


@dataclass
class SimpleDataClass:
    a: Any = None
    b: Any = None


class SimpleClass:
    a: Any = None
    b: Any = None

    def __init__(self, a: Any, b: Any) -> None:
        self.a = a
        self.b = b

    def __eq__(self, other: Any) -> Any:
        if isinstance(other, SimpleClass):
            return self.a == other.a and self.b == other.b
        return False

    @property
    def _fields(self) -> List[str]:
        return ["a", "b"]


@dataclass
class SimpleClassPrimitiveConf:
    _target_: str = "tests.tron.utils.instantiate.SimpleClass"
    str = "partial"
    a: Any = None
    b: Any = None


@dataclass
class SimpleClassNonPrimitiveConf:
    _target_: str = "tests.tron.utils.instantiate.SimpleClass"
    str = "none"
    a: Any = None
    b: Any = None


@dataclass
class SimpleClassDefaultPrimitiveConf:
    _target_: str = "tests.tron.utils.instantiate.SimpleClass"
    a: Any = None
    b: Any = None


@dataclass
class NestedConf:
    _target_: str = "tests.tron.utils.instantiate.SimpleClass"
    a: Any = field(default_factory=lambda: User(name="a", age=1))
    b: Any = field(default_factory=lambda: User(name="b", age=2))


class TargetWithInstantiateInInit:
    def __init__(self, user_config: Optional[DictConfig], user: Optional[User] = None) -> None:
        if user:
            self.user = user
        else:
            user_config["_target_"] = "tests.tron.utils.instantiate.User"
            self.user = instantiate(user_config)

    def __eq__(self, other: Any) -> bool:
        return self.user.__eq__(other.user)


def recisinstance(got: Any, expected: Any) -> bool:
    """Compare got with expected type, recursively on dict and list."""
    if not isinstance(got, type(expected)):
        return False
    if isinstance(expected, collections.abc.Mapping):
        return all(recisinstance(got[key], expected[key]) for key in expected)
    elif isinstance(expected, collections.abc.Iterable) and not isinstance(expected, str):
        return all(recisinstance(got[idx], exp) for idx, exp in enumerate(expected))
    elif hasattr(expected, "_fields"):
        return all(recisinstance(getattr(got, key), getattr(expected, key)) for key in expected._fields)
    return True


an_object = object()
