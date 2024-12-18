from typing import Any, Callable, TypeVar
from functools import partial

import fiddle as fdl

T = TypeVar("T")


class PredicateDispatch:
    def __init__(self):
        self.handlers = []

    def register(self, predicate: Callable[[Any], bool]):
        def decorator(func: Callable[[T], Any]):
            self.handlers.append((predicate, func))
            return func

        return decorator

    def __call__(self, value):
        for predicate, handler in self.handlers:
            if predicate(value):
                return handler(value)
        return value  # default case: return unchanged


to_config = PredicateDispatch()


@to_config.register(lambda v: isinstance(v, partial))
def handle_partial(value: partial):
    return fdl.Partial(value.func, *value.args, **value.keywords)
