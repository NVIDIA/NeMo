from operator import attrgetter
from omegaconf import OmegaConf, DictConfig, ListConfig

from megatron.core.transformer.spec_utils import ModuleSpec, import_module


def override_spec(spec_root: ModuleSpec, cfg: DictConfig):
    overrides_cfg: ListConfig = cfg.get('spec_overrides')
    if overrides_cfg is None:
        return spec_root

    for override_cfg in overrides_cfg:
        submodule_path = override_cfg.get('submodule')
        if not submodule_path:
            parent = None
            obj_to_override = spec_root
        else:
            base_path, submodule_name = submodule_path.rsplit('.', 1)
            parent = attrgetter(base_path)(spec_root)
            obj_to_override = getattr(parent, submodule_name)

        # If user specified a class, import and create a new ModuleSpec and assign it to the submodule.
        new_cls = override_cfg.get('class')
        if new_cls is not None:
            # User specified class - create ModuleSpec
            cls = import_module(tuple(new_cls.rsplit('.', 1)))
            obj_to_override = ModuleSpec(cls)

        if isinstance(obj_to_override, ModuleSpec):
            # Possibly additional params configured
            # Support two options for this:
            #   1. "Redirect" to a separate config element via 'config_key' (easier to override from command line)
            #   2. 'params' sub-field in current override configuration element
            config_key = override_cfg.get('config_key')
            parent_cfg, key = (cfg, config_key) if config_key else (override_cfg, 'params')
            params = OmegaConf.to_container(parent_cfg.get(key, DictConfig({})))
            obj_to_override.params.update(params)

        if parent:
            setattr(parent, submodule_name, obj_to_override)
        else:
            spec_root = obj_to_override

    return spec_root
