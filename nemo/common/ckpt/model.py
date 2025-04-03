from typing import Type, TypeVar, overload, Union, Callable
from collections import OrderedDict, defaultdict
import gc
import os
from pathlib import Path
from copy import deepcopy

from torch import nn
import rich

import nemo_run as run
from nemo.lightning.io.mixin import IOMixin
from nemo_run.config import Config, ConfigurableMixin
from nemo.common.plan.plan import Plan
import nemo.lightning as nl
from lightning import LightningModule
from nemo.common.plan.missing import Missing
from nemo.common.plan.env import LightningEnv
from nemo.common.plan.model_context import ModelContext
from nemo.common.ckpt.registry import detect_checkpoint_type, init_model
from nemo.common.plan.registry import PlanRegistryMixin


class PreTrainedModel(Plan[nn.Module], IOMixin, ConfigurableMixin, PlanRegistryMixin):
    @overload
    def __new__(
        cls,
        model: str | run.Config,
        *,
        env: Plan | nl.Trainer | str | None = None,
        parse: Plan | str | None = None,
        lazy: bool = False,
        trust_remote_code: bool = False,
        cache_dir: str | Path | None = None,
        hub_token: str | bool | None = None,
        path_resolver: str | Callable[[str], Path] | None = None,
        **kwargs: Plan,
    ) -> nn.Module: ...

    @overload
    def __new__(
        cls,
        model: str | run.Config,
        *,
        env: Plan | nl.Trainer | str | None = None,
        parse: Plan | str | None = None,
        lazy: bool = True,
        trust_remote_code: bool = False,
        cache_dir: str | Path | None = None,
        hub_token: str | bool | None = None,
        path_resolver: str | Callable[[str], Path] | None = None,
        **plans: Plan,
    ) -> "PreTrainedModel": ...

    def __new__(
        cls,
        model: str | run.Config,
        *,
        env: Plan | nl.Trainer | str | None = None,
        parse: Plan | str | None = None,
        lazy: bool = False,
        trust_remote_code: bool = False,
        cache_dir: str | Path | None = None,
        hub_token: str | bool | None = None,
        path_resolver: str | Callable[[str], Path] | None = None,
        **kwargs: Plan,
    ) -> Union[nn.Module, "PreTrainedModel"]:
        instance = super().__new__(cls)

        if not env:
            lazy = True

        instance.__init__(
            model,
            parse=parse,
            env=env,
            trust_remote_code=trust_remote_code,
            cache_dir=cache_dir,
            hub_token=hub_token,
            path_resolver=path_resolver,
            lazy=lazy,
            **kwargs,
        )
        if lazy:
            return instance

        rich.print(f"Executing: {instance}")
        return instance()

    def __init__(
        self,
        model: str | run.Config,
        *,
        env: Plan | nl.Trainer | str | None = None,
        parse: Plan | str | None = None,
        peft: Plan | str | None = None,
        trust_remote_code: bool = False,
        cache_dir: str | Path | None = None,
        hub_token: str | bool | None = None,
        path_resolver: str | Callable[[str], Path] | None = None,
        lazy: bool = False,
        **kwargs: Plan | str,
    ):
        """Initialize the AutoModel with a model identifier and execution plan.

        Args:
            model: The model identifier or config. For HuggingFace Hub models, you can specify
                revision using '@', e.g. 'hf://meta-llama/Llama-2-7b@main'
            parse: Optional importer plan or string identifier
            env: Optional setup configuration or trainer
            trust_remote_code: Whether to trust remote code
            cache_dir: Optional cache directory for downloaded models
            hub_token: Optional HuggingFace Hub token for private models
            **plans: Additional plans to include in the execution plan, can be Plan objects or string identifiers
        """
        # super().__init__()
        self.model = model
        if isinstance(model, str):
            self._ckpt_type = detect_checkpoint_type(model, path_resolver)
            # self.context = ModelContext(model, path_resolver=path_resolver)
        self.trust_remote_code = trust_remote_code
        self.cache_dir = cache_dir or Path(os.environ.get("NEMO_HOME")) / "models"
        self.hub_token = hub_token
        self.lazy = lazy
        self._partial_ckpt = False
        _parse = self.get_parse(parse, model, path_resolver) if parse else None

        if isinstance(env, (nl.Trainer, nl.Fabric)):
            env = LightningEnv(env)
        elif isinstance(env, str):
            env = self.get_env(env, _parse)
        self.env = env or Missing()

        if _parse:
            self.parse = _parse
        elif isinstance(model, (str, Path)):
            self.parse = ModelContext(model, path_resolver=path_resolver)
        else:
            raise ValueError(f"Invalid model: {model}")

        if _parse:
            init = self.parse.loader(self.env, peft=peft)
            if init:
                self.init = init
        else:
            # TODO: Fix this hardcoded context
            # TODO: add kwargs to init_model
            # Do we need the env?
            self.init = init_model(self.model, peft=peft)

        # Process the additional plans
        for name, val in kwargs.items():
            if name == "lazy":
                continue
            if isinstance(val, str):
                # String identifier - look it up in the registry
                if name in self._registry and val in self._registry[name]:
                    # This is a registered plan handler, call it to get the plan
                    plan_handler = self._registry[name][val]
                    # The importer might be needed as context for some plans
                    plan = plan_handler(getattr(self, "importer", None), val)
                    setattr(self, name, plan)
                else:
                    available_handlers = (
                        list(self._registry[name].keys())
                        if name in self._registry
                        else []
                    )
                    raise ValueError(
                        f"Unknown {name} handler: {val}. Available handlers: {available_handlers}"
                    )
            else:
                if not isinstance(val, Plan):
                    raise ValueError(
                        f"{name} must be a Plan object or a string, got {type(val)}"
                    )

                # Already a Plan object
                setattr(self, name, val)

    def setup(self, env: Plan | nl.Trainer | str) -> nn.Module:
        if isinstance(env, str):
            env = self.get_env(env, self.parse)
        self.env = env
        return self()
    
    @classmethod
    def register(cls, plan_type: str, *names_or_patterns: str):
        """Decorator to register a plan handler function.

        Args:
            plan_type: Type of plan to register (e.g., "importer", "setup")
            *names_or_patterns: One or more string identifiers or patterns

        Returns:
            Decorator function that registers the handler
        """

        def decorator(func):
            for name in names_or_patterns:
                cls._registry[plan_type][name] = func
            return func

        return decorator

    @classmethod
    def get_parse(
        cls,
        parser: Plan | str,
        model: str | run.Config,
        path_resolver: str | Callable[[str], Path] | None = None,
        overwrite: bool = False,
    ) -> Plan:
        if isinstance(parser, str):
            if parser in cls._registry["parse"]:
                parser = cls._registry["parse"][parser]()
            else:
                raise ValueError(
                    f"Unknown parser: {parser}. Available parsers: {list(cls._registry['parse'].keys())}"
                )
        if not isinstance(parser, Plan):
            raise ValueError(f"Parser must be a Plan, got {type(parser)}")

        parser.setup(model, path_resolver, overwrite=overwrite)
        return parser

    @classmethod
    def get_env(cls, env: str, parse: Plan | None = None) -> Plan:
        return cls.get_plan("env", env, parse)

    @classmethod
    def get_plan(
        cls, plan_type: str, plan_name: str, importer: Plan | None = None
    ) -> Plan:
        """Get a plan by type and name.

        Args:
            plan_type: Type of plan to get (e.g., "importer", "setup")
            plan_name: Name or identifier of the plan
            importer: Optional importer to provide as context

        Returns:
            Instantiated plan object

        Raises:
            ValueError: If the plan type or name is not found in the registry
        """
        registry = cls._registry[plan_type]

        # First try exact match
        if plan_name in registry:
            return registry[plan_name](importer, plan_name)

        # Then try pattern matching for setup-like registries
        import re

        for pattern, handler in registry.items():
            if "*" in pattern or "[" in pattern:
                # Convert our pattern syntax to regex
                regex_pattern = (
                    pattern.replace("*", ".*").replace("[", r"\[").replace("]", r"\]")
                )
                if re.match(regex_pattern, plan_name):
                    return handler(importer, plan_name)

        raise ValueError(
            f"Unknown {plan_type} handler: {plan_name}. Available handlers: {list(registry.keys())}"
        )

    def to_config(self) -> Config:
        return deepcopy(self.__io__)
    

def register_parser(name: str):
    def decorator(cls):
        # Apply the original decorator
        decorated_class = PreTrainedModel.register("parse", name)(cls)
        # Set the name class variable
        decorated_class.name = name
        return decorated_class

    return decorator


def register_context_convert(
    name: str, *, source: Type[nn.Module], target: Type[nn.Module]
):
    if name not in PreTrainedModel._registry["parse"]:
        raise ValueError(
            f"Unknown parser: {name}. Available parsers: {list(PreTrainedModel._registry['parse'].keys())}"
        )

    return PreTrainedModel._registry["parse"][name].context_converter(source, target)


def register_state_convert(
    name: str, *, source: Type[nn.Module], target: Type[nn.Module]
):
    if name not in PreTrainedModel._registry["parse"]:
        raise ValueError(
            f"Unknown parser: {name}. Available parsers: {list(PreTrainedModel._registry['parse'].keys())}"
        )

    return PreTrainedModel._registry["parse"][name].state_converter(source, target)


@PreTrainedModel.register("env", "megatron_cpu", "megatron_cpu[*]")
def megatron_cpu(importer: Plan | None = None, setup: str = "cpu") -> LightningEnv:
    if not importer:
        raise ValueError("Cannot setup model on CPU without importer")

    return _setup_full_megatron(importer, setup, meta=False)


@PreTrainedModel.register("env", "megatron_meta", "megatron_meta[*]")
def megatron_meta(importer: Plan | None = None, setup: str = "meta") -> LightningEnv:
    if not importer:
        raise ValueError("Cannot setup model on CPU without importer")

    return _setup_full_megatron(importer, setup, meta=True, can_load=False)


@PreTrainedModel.register("env", "megatron_single", "megatron_single[*]")
def megatron_single(importer: Plan | None = None, setup: str = "meta") -> LightningEnv:
    if not importer:
        raise ValueError("Cannot setup model on CPU without importer")

    return _setup_full_megatron(importer, setup, can_load=False)


def _setup_full_megatron(
    importer: Plan, setup: str, meta: bool = False, can_load: bool = True
) -> LightningEnv:
    # This is a megatron-based module
    model_cls = importer.model_class
    if issubclass(model_cls, LightningModule) and hasattr(model_cls, "forward_step"):
        trainer_kwargs = {
            "devices": 1,
            "accelerator": "cpu",
            "strategy": nl.MegatronStrategy(),
        }

        # Parse precision config if present
        if "[" in setup:
            config = setup[setup.find("[") + 1 : setup.find("]")]
            if config in ("16-mixed", "bf16-mixed", "32"):
                trainer_kwargs["plugins"] = nl.MegatronMixedPrecision(config)

        trainer = nl.Trainer(**trainer_kwargs)
        fabric = trainer.to_fabric()
        fabric.strategy.meta = meta
        return LightningEnv(fabric, can_load=can_load)

    raise ValueError("Cannot setup model on CPU")
