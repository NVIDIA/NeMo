from torch import nn
from typing_extensions import Self

from nemo.collections.llm.fn import base as fn


class FNMixin:
    """
    A mixin class providing utility methods for operating on PyTorch modules.

    This mixin class offers methods to apply functions, check predicates, and modify
    the state (freeze/unfreeze) of PyTorch modules within a container. It is designed
    to be used with classes that are composed of multiple PyTorch modules, facilitating
    operations that affect all contained modules either directly or recursively.

    Methods
    -------
        forall: Checks if a predicate holds for all modules.
        map: Applies a function to each module.
        walk: Traverses each module, applying a function.
        freeze: Freezes the parameters of all modules.
        unfreeze: Unfreezes the parameters of all modules.

    Examples
    --------
        >>> class MyModel(nn.Module, FNMixin):
        ...     def __init__(self):
        ...         super().__init__()
        ...         self.layer1 = nn.Linear(10, 10)
        ...         self.layer2 = nn.Linear(10, 10)
        ...
        >>> model = MyModel()
        >>> model.freeze()  # Freezes all parameters in the model
        >>> model.forall(lambda module: not module.parameters().requires_grad, recurse=True)
        True
    """
    
    def forall(self, func: fn.ModulePredicate, recurse: bool = False) -> bool:
        """
        Evaluates a predicate for all modules in the container, optionally recursively.

        This method checks if a given predicate holds for all modules in the container.
        If `recurse` is True, it also checks all submodules recursively.

        Args:
            func (fn.ModulePredicate): A predicate function to apply to each module.
            recurse (bool, optional): Whether to apply the predicate recursively. Defaults to False.

        Returns
        -------
            bool: True if the predicate holds for all modules, False otherwise.

        Example:
            >>> model = MyModel()
            >>> model.forall(lambda module: isinstance(module, nn.Linear), recurse=True)
            True
        """
        assert isinstance(self, nn.Module), "self is not a nn.Module"
        
        return fn.forall(self, func, recurse=recurse)

    def map(self, func: fn.ModuleFunc, leaf_only: bool = False) -> Self:
        """
        Applies a function to each module in the container, optionally to leaf modules only.

        This method applies a given function to each module in the container. If `leaf_only`
        is True, the function is applied to leaf modules only.

        Args:
            func (fn.ModuleFunc): A function to apply to each module.
            leaf_only (bool, optional): Whether to apply the function to leaf modules only. Defaults to False.

        Returns
        -------
            Self: The container itself after applying the function.

        Example:
            >>> model = MyModel()
            >>> model.map(lambda module: module.double() if isinstance(module, nn.Linear) else module)
            <MyModel object>
        """
        assert isinstance(self, nn.Module), "self is not a nn.Module"

        return fn.map(self, func, leaf_only=leaf_only, _skip_map=True)

    def walk(self, func: fn.ModuleFunc, leaf_only: bool = False) -> Self:
        """
        Traverses each module in the container, applying a function, optionally to leaf modules only.

        This method is similar to `map`, but it is typically used for operations that do not
        modify the modules but instead collect information or perform checks.

        Args:
            func (fn.ModuleFunc): A function to apply to each module.
            leaf_only (bool, optional): Whether to traverse leaf modules only. Defaults to False.

        Returns
        -------
            Self: The container itself after the traversal.

        Example:
            >>> model = MyModel()
            >>> model.walk(print, leaf_only=True)
            <MyModel object>
        """
        assert isinstance(self, nn.Module), "self is not a nn.Module"

        return fn.walk(self, func, leaf_only=leaf_only, _skip_map=True)

    def freeze(self) -> None:
        """
        Freezes the parameters of all modules in the container
        by setting `requires_grad` to False.
        """
        assert isinstance(self, nn.Module), "self is not a nn.Module"
        
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze(self) -> None:
        """
        Unfreezes the parameters of all modules in the container
        by setting `requires_grad` to True.
        """
        assert isinstance(self, nn.Module), "self is not a nn.Module"
        
        for param in self.parameters():
            param.requires_grad = True


# TODO: Implement this:
# class FNListMixin(FNMixin):
#     def filter(self, func: ModulePredicate, recurse: bool = False) -> Self:
#         """
#         Returns a new container with modules that satisfy the filtering function.

#         Example usage::
#             >>> block = Block(nn.LazyLinear(10))
#             >>> block.filter(lambda module: isinstance(module, nn.Linear))
#             Block(nn.Linear(10, 10))

#         Parameters
#         ----------
#         func (Callable[[Module], bool]): A function that takes a module and returns
#             a boolean or a boolean-like object.
#         recurse (bool, optional): Whether to recursively filter modules
#             within sub-containers. Default is False.

#         Returns
#         -------
#             Self: A new container with the filtered modules.
#         """

#         _to_call = _recurse(func, "filter") if recurse else func
#         output = cast(List, self.__class__())

#         for module in cast(Iterable, self):
#             filtered = _to_call(module)
#             if filtered:
#                 if isinstance(filtered, bool):
#                     output.append(module)
#                 else:
#                     output.append(filtered)

#         return cast(Self, output)

#     def flatmap(self, func: ModuleFunc) -> Self:
#         """
#         Applies a function to each module and flattens the results into a new container.

#         Example usage::
#             >>> block = Block(nn.LazyLinear(10))
#             >>> container.flatmap(lambda module: [module, module])
#             Block(nn.LazyLinear(10), nn.LazyLinear(10))

#         Parameters
#         ----------
#         func : Callable[[Module], Iterable[Module]]
#             A function that takes a module and returns an iterable of modules.

#         Returns
#         -------
#         Self
#             A new container with the flattened modules.

#         Raises
#         ------
#         TypeError
#             If the input function is not callable.
#         RuntimeError
#             If an exception occurs during mapping the function over the module.
#         """

#         if not callable(func):
#             raise TypeError(f"Expected callable function, received: {type(func).__name__}")

#         try:
#             mapped = cast(List, self.map(func))
#         except Exception as e:
#             raise RuntimeError("Failed to map function over the module") from e

#         output = self.__class__()
#         assert isinstance(output, Iterable), "self is not an iterable"

#         try:
#             for sublist in mapped:
#                 for item in sublist:
#                     output.append(item)
#         except TypeError as e:
#             raise TypeError("Function did not return an iterable object") from e

#         return output
    
#     def choose(self, func: ModuleMapFunc, recurse: bool = False) -> Self:
#         """
#         Returns a new container with modules that are selected by the given function.

#         Example usage::
#             >>> block = Block(nn.LazyLinear(10), nn.Relu())
#             >>> container.choose(lambda m: m if isinstance(m, nn.Linear) else None)
#             Block(nn.LazyLinear(10))

#         Parameters
#         ----------
#         func : Callable[[Module], Union[Module, None]]
#             A function that takes a module and returns a module or None.
#         recurse : bool, optional
#             Whether to recursively choose modules within sub-containers. Default is False.

#         Returns
#         -------
#         Self
#             A new container with the chosen modules.
#         """

#         to_add = []
#         _to_call = _recurse(func, "choose") if recurse else func
        
#         assert isinstance(self, Iterable), "self is not an iterable"

#         for module in self:
#             f_out = _to_call(module)
#             if f_out:
#                 to_add.append(f_out)

#         return self.__class__(*to_add)
    
#     def zip(self, other: Iterable[_TModule]) -> Iterable[Tuple[_TModule, _TModule]]:
#         """
#         Zips the modules of the container with the modules from another iterable into pairs.

#         Example usage::
#             >>> list(Block(nn.Linear(10)).zip(Block(nn.Linear(10))))
#             [(nn.Linear(10), nn.Linear(10))]

#         Parameters
#         ----------
#         other : Iterable[Self]
#             Another iterable containing modules to be zipped with.

#         Returns
#         -------
#         Iterable[Tuple[Self, Self]]
#             An iterable of pairs containing modules from the container
#             and the other iterable.
#         """

#         return builtins.zip(cast(Iterable, self), other)
    
#     def __add__(self, module) -> Self:
#         if hasattr(module, "__iter__"):
#             return self.__class__(*self, *module)

#         return self.__class__(*self, module)

#     def __radd__(self, module) -> Self:
#         if hasattr(module, "__iter__"):
#             return self.__class__(*module, *self)

#         return self.__class__(module, *self)