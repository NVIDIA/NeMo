import hydra
import omegaconf


def convert_to_cli(cfg):
    result = ""
    if cfg.get("data_config") is not None:
        result += f"data_preparation={cfg['data_config']} "
    if cfg.get("training_config") is not None:
        result += f"training={cfg['training_config']} "
    if cfg.get("finetuning_config") is not None:
        result += f"finetuning={cfg['finetuning_config']} "
    if cfg.get("evaluation_config") is not None:
        result += f"evaluation={cfg['evaluation_config']} "
    if cfg.get("conversion_config") is not None:
        result += f"conversion={cfg['conversion_config']} "

    for k, v in cfg.items():
        if k in ["dgxa100_gpu2core", "dgxa100_gpu2mem"]:
            continue

        if isinstance(v, omegaconf.dictconfig.DictConfig):
            output = convert_to_cli(v).split(" ")
            result += " ".join([f"{k}.{x}" for x in output if x != ""]) + " "
        elif isinstance(v, omegaconf.listconfig.ListConfig):
            if k == "data_prefix":
                if v is None:
                    v = "null"
                else:
                    v = [x for x in v]  # Needed because of lazy omegaconf interpolation.
            result += f"{k}={str(v).replace(' ', '')} "
        elif isinstance(v, str) and "{" in v:
            continue
        elif k in ["splits_string", "file_numbers", "languages"]:
            result += f"{k}=\\'{v}\\' "
        elif k == "checkpoint_name":
            v = v.replace('=', '\=')
            result += f"{k}=\'{v}\' "
        else:
            result += f"{k}={convert_to_null(v)} "
    return result


def convert_to_null(val):
    if val is None:
        return "null"
    return val


def fake_submit(*args, **kwargs):
    print(args, kwargs)
    fake_id = 123456
    return str(fake_id).encode()


def add_container_mounts(container_mounts):
    mounts_str = ""
    if container_mounts is not None:
        assert isinstance(container_mounts, omegaconf.listconfig.ListConfig), "container_mounts must be a list."
        for mount in container_mounts:
            if mount is not None and isinstance(mount, str):
                mounts_str += f",{mount}" if ":" in mount else f",{mount}:{mount}"
    return mounts_str
