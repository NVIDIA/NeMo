from typing import Mapping

_HAS_HYDRA = True

try:
    import hydra
    from omegaconf import DictConfig
except ModuleNotFoundError:
    DictConfig = Mapping
    _HAS_HYDRA = False


def resolve_trainer_cfg(trainer_cfg: DictConfig) -> DictConfig:
    if not _HAS_HYDRA:
        return trainer_cfg
    if (strategy := trainer_cfg.get("strategy", None)) is not None and isinstance(strategy, Mapping):
        trainer_cfg["strategy"] = hydra.utils.instantiate(strategy)
    return trainer_cfg
