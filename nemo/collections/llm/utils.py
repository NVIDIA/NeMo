from dataclasses import asdict, dataclass
from typing import Any, Callable, Generic, Optional, TypeVar, Union, overload

T = TypeVar('T', bound=Callable[..., Any])

try:
    import nemo_sdk as sdk

    Config = sdk.Config
    Partial = sdk.Partial
except ImportError:
    _T = TypeVar('_T')

    class Config(Generic[_T]):
        pass

    class Partial(Generic[_T]):
        pass


def task(*args: Any, **kwargs: Any) -> Callable[[T], T]:
    try:
        import nemo_sdk as sdk

        return sdk.task(*args, **kwargs)
    except ImportError:
        # Return a no-op function
        def noop_decorator(func: T) -> T:
            return func

        return noop_decorator


@overload
def factory() -> Callable[[T], T]: ...


@overload
def factory(*args: Any, **kwargs: Any) -> Callable[[T], T]: ...


def factory(*args: Any, **kwargs: Any) -> Union[Callable[[T], T], T]:
    try:
        import nemo_sdk as sdk

        if not args and not kwargs:
            # Used as @factory without arguments
            return sdk.factory()
        else:
            # Used as @factory(*args, **kwargs)
            return sdk.factory(*args, **kwargs)
    except ImportError:
        # Return a no-op function
        def noop_decorator(func: T) -> T:
            return func

        if not args and not kwargs:
            return noop_decorator
        else:
            return noop_decorator


@dataclass
class PreTrainRecipe:
    model: Config
    trainer: Config
    data: Config
    optim: Config
    log: Config

    @property
    def partial(self) -> Partial:
        from nemo.collections.llm.api import pretrain

        recipy_kwargs = {}
        for attr, value in asdict(self).items():
            recipy_kwargs[attr] = value.as_factory()

        return Partial(pretrain, **recipy_kwargs)


@dataclass
class FineTuneRecipe:
    model: Config
    trainer: Config
    data: Config
    optim: Config
    log: Config
    peft: Optional[Config] = None
    resume: Optional[Config] = None

    @property
    def partial(self) -> Partial:
        from nemo.collections.llm.api import finetune

        recipy_kwargs = {}
        for attr, value in asdict(self).items():
            recipy_kwargs[attr] = value.as_factory()

        return Partial(finetune, **recipy_kwargs)


def recipe_aware_parse_partial(recipe_type):
    def _parser(fn, args):
        import nemo_sdk as sdk
        from nemo_sdk.config import Partial, set_value
        from nemo_sdk.core.lark_parser import parse_args

        parsed_kwargs, parsed_overrides = parse_args(args)

        if "recipe" in parsed_kwargs:
            recipe = sdk.resolve(recipe_type, parsed_kwargs["recipe"])

            recipe_kwargs = {}
            for attr in dir(recipe):
                recipe_kwargs[attr] = getattr(recipe, attr).as_factory()

            config = Partial(fn, **recipe_kwargs)
            for key, value in parsed_kwargs.items():
                set_value(config, key, value)
            for key, value in parsed_overrides.items():
                set_value(config, key, value)
        else:
            config = Partial(fn, **parsed_kwargs)
            for key, value in parsed_overrides.items():
                set_value(config, key, value)

        return config

    return _parser
