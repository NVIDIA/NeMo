from typing import Any, Dict, Union
import copy
from functools import partial
import torch
import torch.optim as optim
from omegaconf import DictConfig, OmegaConf

from nemo.core.config import OptimizerParams, get_optimizer_config, register_optimizer_params
from nemo.core.optim.adafactor import Adafactor
from nemo.core.optim.adan import Adan
from nemo.core.optim.novograd import Novograd
from nemo.utils.model_utils import maybe_update_config_version

# Define available optimizers
AVAILABLE_OPTIMIZERS = {
    'sgd': optim.SGD,
    'adam': optim.Adam,
    'adamw': optim.AdamW,
    'adadelta': optim.Adadelta,
    'adamax': optim.Adamax,
    'adagrad': optim.Adagrad,
    'rmsprop': optim.RMSprop,
    'rprop': optim.Rprop,
    'novograd': Novograd,
    'adafactor': Adafactor,
    'adan': Adan,
}

# Add support for APEX optimizers if available
try:
    from apex.optimizers import FusedAdam, FusedLAMB

    HAVE_APEX = True

    AVAILABLE_OPTIMIZERS['lamb'] = FusedLAMB
    AVAILABLE_OPTIMIZERS['fused_adam'] = FusedAdam
except ModuleNotFoundError:
    HAVE_APEX = False

# Check for APEX distributed Adam optimizer
if HAVE_APEX:
    try:
        from nemo.core.optim.distributed_adam import MegatronDistributedFusedAdam
        HAVE_APEX_DISTRIBUTED_ADAM = True
        AVAILABLE_OPTIMIZERS['distributed_fused_adam'] = MegatronDistributedFusedAdam
    except (ImportError, ModuleNotFoundError):
        HAVE_APEX_DISTRIBUTED_ADAM = False

    # Check for APEX FusedAdam optimizer
    try:
        from nemo.core.optim.megatron_fused_adam import MegatronFusedAdam
        AVAILABLE_OPTIMIZERS['megatron_fused_adam'] = MegatronFusedAdam
    except (ImportError, ModuleNotFoundError):
        pass

__all__ = ['get_optimizer', 'register_optimizer', 'parse_optimizer_args']


def parse_optimizer_args(
    optimizer_name: str, optimizer_kwargs: Union[DictConfig, Dict[str, Any]]
) -> Union[Dict[str, Any], DictConfig]:
    """
    Parses optimizer arguments into a dictionary for optimizer instantiation.

    Args:
        optimizer_name: Name of the optimizer.
        optimizer_kwargs: Optimizer keyword arguments.

    Returns:
        Parsed optimizer arguments.
    """
    kwargs = {}

    if optimizer_kwargs is None:
        return kwargs

    optimizer_kwargs = copy.deepcopy(optimizer_kwargs)
    optimizer_kwargs = maybe_update_config_version(optimizer_kwargs)

    if isinstance(optimizer_kwargs, DictConfig):
        optimizer_kwargs = OmegaConf.to_container(optimizer_kwargs, resolve=True)

    if hasattr(optimizer_kwargs, 'keys'):
        if '_target_' in optimizer_kwargs:
            optimizer_kwargs_config = OmegaConf.create(optimizer_kwargs)
            optimizer_instance = hydra.utils.instantiate(optimizer_kwargs_config)
            optimizer_instance = vars(optimizer_instance)
            return optimizer_instance

        if 'name' in optimizer_kwargs:
            if optimizer_kwargs['name'] == 'auto':
                optimizer_params_name = "{}_params".format(optimizer_name)
                optimizer_kwargs.pop('name')
            else:
                optimizer_params_name = optimizer_kwargs.pop('name')

            if 'params' in optimizer_kwargs:
                optimizer_params_override = optimizer_kwargs.get('params')
            else:
                optimizer_params_override = optimizer_kwargs

            if isinstance(optimizer_params_override, DictConfig):
                optimizer_params_override = OmegaConf.to_container(optimizer_params_override, resolve=True)

            optimizer_params_cls = get_optimizer_config(optimizer_params_name, **optimizer_params_override)

            if optimizer_params_name is None:
                optimizer_params = vars(optimizer_params_cls)
                return optimizer_params
            else:
                optimizer_params = optimizer_params_cls()
                optimizer_params = vars(optimizer_params)
                return optimizer_params

        return optimizer_kwargs

    return kwargs


def register_optimizer(name: str, optimizer: optim.Optimizer, optimizer_params: OptimizerParams):
    """
    Registers a custom optimizer.

    Args:
        name: Name of the optimizer.
        optimizer: Optimizer class.
        optimizer_params: Optimizer parameters.
    """
    if name in AVAILABLE_OPTIMIZERS:
        raise ValueError(f"Cannot override pre-existing optimizers. Conflicting optimizer name = {name}")

    AVAILABLE_OPTIMIZERS[name] = optimizer

    optim_name = "{}_params".format(optimizer.__name__)
    register_optimizer_params(name=optim_name, optimizer_params=optimizer_params)


def get_optimizer(name: str, **kwargs: Any) -> optim.Optimizer:
    """
    Obtains an optimizer class.

    Args:
        name: Name of the optimizer.
        kwargs: Optimizer keyword arguments.

    Returns:
        Optimizer instance.
    """
    if name not in AVAILABLE_OPTIMIZERS:
        raise ValueError(f"Cannot resolve optimizer '{name}'. Available optimizers are: {AVAILABLE_OPTIMIZERS.keys()}")

    if name == 'fused_adam':
        if not torch.cuda.is_available():
            raise ValueError(f'CUDA must be available to use fused_adam.')

    optimizer = AVAILABLE_OPTIMIZERS[name]
    optimizer = partial(optimizer, **kwargs)
    return optimizer


def init_optimizer_states(optimizer: optim.Optimizer):
    """
    Initializes optimizer states.

    Args:
        optimizer: Optimizer instance.
    """
    adam_nondist_optims = (optim.Adam, optim.AdamW)
    if HAVE_APEX:
        adam_nondist_optims += (FusedAdam,)
    if isinstance(optimizer, adam_nondist_optims):
        for group in optimizer.param_groups:
            for p in group['params']:
                state = optimizer.state[p]
               
